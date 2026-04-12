"""
USD/INR Next-Day Close Price Predictor — KNN Edition
=====================================================
KNN classifier replaces gamma/beta EMA threshold model for direction.
Everything else (GARCH vol, regime, price formula) is identical.

Features used for direction (StandardScaler applied before KNN):
  1.  feat            = EMA5 - EMA20
  2.  feat_change     = feat - feat.shift(1)
  3.  price_vs_ema20  = (close - ema20) / ema20
  4.  ret_1d          = close.pct_change(1)
  5.  ret_3d          = close.pct_change(3)
  6.  vol_ratio       = std5 / std25
  7.  intraday_body   = (close - open) / open
  8.  wick_upper      = (high - close) / (high - low + 1e-9)
  9.  close_in_range  = (close - low) / (high - low + 1e-9)
  10. prev_direction  = sign(close.diff().shift(1))

Scaler  : StandardScaler fitted on train only, applied to both train & test
k tuning: 5-fold CV over k = 3,5,7,9,11,15,21,31 (odd only, avoids ties)

Train : 2003-2020  (KNN fitted once, fixed)
Test  : 2021-2023  (zero lookahead, out-of-sample)

Outputs:
  usd_inr_knn_plots/index.html            (train summary)
  usd_inr_knn_plots/validation_index.html (test summary)
  usd_inr_knn_plots/plot_NN_YYYY.html     (per-year charts)
"""

import pandas as pd
import numpy as np
import os
import json
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# ──────────────────────────────────────────────────────────────────────────────
# 1. GARCH(1,1)
# ──────────────────────────────────────────────────────────────────────────────

def _garch_variance(r, omega, alpha, beta):
    n = len(r)
    h = np.full(n, max(float(np.var(r)), 1e-10))
    for t in range(1, n):
        h[t] = max(omega + alpha * r[t-1]**2 + beta * h[t-1], 1e-12)
    return h


def fit_garch11(returns):
    r  = returns.fillna(0).values
    uv = max(float(np.var(r)), 1e-10)
    best_nll, best_params = np.inf, None

    for a in np.linspace(0.04, 0.30, 10):
        for b in np.linspace(0.55, 0.92, 10):
            if a + b >= 0.9999:
                continue
            omega = uv * (1.0 - a - b)
            if omega <= 0:
                continue
            h   = _garch_variance(r, omega, a, b)
            nll = 0.5 * float(np.sum(np.log(h) + r**2 / h))
            if nll < best_nll:
                best_nll    = nll
                best_params = (omega, a, b)

    if best_params is None:
        best_params = (uv * 0.05, 0.10, 0.80)

    omega, alpha, beta = best_params
    h = _garch_variance(r, omega, alpha, beta)
    print(f"  GARCH(1,1): omega={omega:.2e}  alpha={alpha:.4f}  beta={beta:.4f}"
          f"  persistence={alpha+beta:.4f}")
    return np.sqrt(np.maximum(h, 1e-12))


# ──────────────────────────────────────────────────────────────────────────────
# 2. Volatility regime
# ──────────────────────────────────────────────────────────────────────────────

def classify_vol_regime(df):
    c = df['Close']
    df['std5']  = c.rolling(5).std()
    df['std14'] = c.rolling(14).std()
    df['std25'] = c.rolling(25).std()
    high = (df['std5'] > df['std14']) & (df['std14'] > df['std25'])
    low  = (df['std5'] < df['std14']) & (df['std14'] < df['std25'])
    df['vol_regime'] = 'medium'
    df.loc[high, 'vol_regime'] = 'high'
    df.loc[low,  'vol_regime'] = 'low'
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    'feat', 'feat_change', 'price_vs_ema20',
    'ret_1d', 'ret_3d', 'vol_ratio',
    'intraday_body', 'wick_upper', 'close_in_range', 'prev_direction',
]


def build_features(df):
    c = df['Close']
    o = df['Open']
    h = df['High']
    l = df['Low']

    df['ema_5']  = c.ewm(span=5,  adjust=False).mean()
    df['ema_20'] = c.ewm(span=20, adjust=False).mean()

    df['feat']           = df['ema_5'] - df['ema_20']
    df['feat_change']    = df['feat'].diff()
    df['price_vs_ema20'] = (c - df['ema_20']) / df['ema_20']
    df['ret_1d']         = c.pct_change(1)
    df['ret_3d']         = c.pct_change(3)
    df['vol_ratio']      = df['std5'] / (df['std25'] + 1e-9)
    df['intraday_body']  = (c - o) / (o + 1e-9)
    df['wick_upper']     = (h - c) / (h - l + 1e-9)
    df['close_in_range'] = (c - l) / (h - l + 1e-9)
    df['prev_direction'] = np.sign(c.diff().shift(1))

    # target: direction of NEXT day's close
    df['next_close']  = c.shift(-1)
    df['target']      = np.where(df['next_close'] > c, 1, -1)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. Train KNN
# ──────────────────────────────────────────────────────────────────────────────

def train_knn(df_train):
    valid = df_train.dropna(subset=FEATURE_COLS + ['target', 'next_close'])
    X_raw = valid[FEATURE_COLS].values
    y     = valid['target'].values

    # Fit scaler on train only
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("\n  Tuning k via 5-fold cross-validation ...")
    k_candidates = [3, 5, 7, 9, 11, 15, 21, 31]
    best_k, best_score = 5, -np.inf
    for k in k_candidates:
        clf    = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        mean_score = scores.mean()
        print(f"    k={k:3d}  cv_acc={mean_score:.4f}  ±{scores.std():.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_k     = k

    print(f"\n  Best k: {best_k}  (cv accuracy: {best_score:.4f})")
    clf = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=-1)
    clf.fit(X, y)

    train_acc = (clf.predict(X) == y).mean()
    print(f"  Train accuracy (in-sample): {train_acc:.4f}")
    return clf, scaler, best_k, best_score


# ──────────────────────────────────────────────────────────────────────────────
# 5. Build predictions
# ──────────────────────────────────────────────────────────────────────────────

def build_predictions(df, clf, scaler):
    close  = df['Close'].values
    gvol   = df['garch_vol'].values
    regime = df['vol_regime'].values
    n      = len(df)
    pred   = np.full(n, np.nan)
    dirs   = np.full(n, np.nan)

    feat_matrix_raw = df[FEATURE_COLS].values
    # Scale the entire matrix at once (faster than row-by-row)
    feat_matrix = scaler.transform(feat_matrix_raw)

    for i in range(n - 1):
        row = feat_matrix[i]
        if np.any(np.isnan(row)):
            continue
        d = clf.predict(row.reshape(1, -1))[0]
        c = close[i]
        g = gvol[i]

        if   regime[i] == 'low':    p = c
        elif regime[i] == 'high':   p = c + d * g * c
        else:                       p = c + d * 0.5 * g * c

        pred[i] = p
        dirs[i] = d

    df = df.copy()
    df['predicted_next'] = pred
    df['next_actual']    = df['Close'].shift(-1)
    df['direction_pred'] = dirs
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 6. Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary(df):
    rows = []
    for year, g in df.groupby(df.index.year):
        v = g.dropna(subset=['predicted_next', 'next_actual'])
        if len(v) == 0:
            continue
        err     = (v['predicted_next'] - v['next_actual']).abs()
        err_pct = err / v['next_actual'] * 100

        # direction accuracy
        actual_dir  = np.sign(v['next_actual'].values - v['Close'].values)
        pred_dir    = v['direction_pred'].values
        valid_mask  = ~np.isnan(pred_dir) & ~np.isnan(actual_dir)
        dir_acc     = (pred_dir[valid_mask] == actual_dir[valid_mask]).mean() if valid_mask.sum() > 0 else np.nan

        rows.append({
            'Year':             year,
            'Days':             len(v),
            'High Vol':         (v['vol_regime'] == 'high').sum(),
            'Low Vol':          (v['vol_regime'] == 'low').sum(),
            'Medium Vol':       (v['vol_regime'] == 'medium').sum(),
            'Dir Acc (%)':      round(float(dir_acc * 100) if not np.isnan(dir_acc) else 0, 2),
            'Avg Error (INR)':  round(float(err.mean()),     4),
            'Avg Error (%)':    round(float(err_pct.mean()), 4),
            '65th Pct Err (%)': round(float(err_pct.quantile(0.65)), 4),
            'Max Error (%)':    round(float(err_pct.max()),  4),
        })
    return pd.DataFrame(rows)


def print_summary(summary, df, label=''):
    print("\n" + "=" * 90)
    if label:
        print(f"  {label}")
    print(f"{'Year':<8} {'Days':>6} {'HiVol':>6} {'LoVol':>6} {'MedVol':>7} "
          f"{'DirAcc%':>8} {'AvgErrINR':>11} {'AvgErr%':>9} {'65pct%':>8} {'MaxErr%':>9}")
    print("-" * 90)
    for _, r in summary.iterrows():
        print(f"{int(r['Year']):<8} {int(r['Days']):>6} {int(r['High Vol']):>6} "
              f"{int(r['Low Vol']):>6} {int(r['Medium Vol']):>7} "
              f"{r['Dir Acc (%)']:>8.2f} "
              f"{r['Avg Error (INR)']:>11.4f} {r['Avg Error (%)']:>9.4f} "
              f"{r['65th Pct Err (%)']:>8.4f} {r['Max Error (%)']:>9.4f}")
    print("=" * 90)
    v  = df.dropna(subset=['predicted_next', 'next_actual'])
    ep = (v['predicted_next'] - v['next_actual']).abs() / v['next_actual'] * 100
    print(f"\nOVERALL ({len(v):,} days)")
    print(f"  Avg abs error   : {ep.mean():.4f}%")
    print(f"  Median error    : {ep.median():.4f}%")
    print(f"  65th pct error  : {ep.quantile(0.65):.4f}%")
    print(f"  90th pct error  : {ep.quantile(0.90):.4f}%")
    print(f"  Max error       : {ep.max():.4f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 7. HTML plots
# ──────────────────────────────────────────────────────────────────────────────

def _safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 4)


def plot_year_html(year_df, year, output_folder, plot_num, index_file='index.html'):
    v = year_df.dropna(subset=['predicted_next', 'next_actual'])

    labels      = [str(d.date()) for d in v.index]
    actual      = [_safe(x) for x in v['next_actual']]
    predicted   = [_safe(x) for x in v['predicted_next']]
    error_inr   = [_safe(x) for x in (v['predicted_next'] - v['next_actual'])]
    error_pct   = [_safe(x) for x in
                   ((v['predicted_next'] - v['next_actual']) / v['next_actual'] * 100)]
    abs_err_pct = [abs(x) for x in error_pct if x is not None]

    regime_colors = {
        'high':   'rgba(255,80,80,0.45)',
        'low':    'rgba(80,200,100,0.45)',
        'medium': 'rgba(100,140,255,0.35)',
    }
    point_colors = [regime_colors.get(r, 'grey') for r in v['vol_regime']]

    avg_err = round(np.mean(abs_err_pct), 3) if abs_err_pct else 0
    p65_err = round(float(np.percentile(abs_err_pct, 65)), 3) if abs_err_pct else 0
    n_high  = (v['vol_regime'] == 'high').sum()
    n_low   = (v['vol_regime'] == 'low').sum()
    n_med   = (v['vol_regime'] == 'medium').sum()

    # direction accuracy
    actual_dir = np.sign(v['next_actual'].values - v['Close'].values)
    pred_dir   = v['direction_pred'].values
    vm         = ~np.isnan(pred_dir)
    dir_acc    = round((pred_dir[vm] == actual_dir[vm]).mean() * 100, 2) if vm.sum() > 0 else 0

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>USD/INR {year} — KNN</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Syne:wght@700;800&display=swap');
  *    {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #080c10; color: #c9d1d9;
          font-family: 'JetBrains Mono', monospace; padding: 24px; }}
  h1   {{ text-align: center; color: #e2b714;
          font-family: 'Syne', sans-serif; font-size: 1.6em;
          letter-spacing: -0.02em; margin-bottom: 4px; }}
  .sub {{ text-align: center; color: #4a5568; font-size: 0.72em;
          margin-bottom: 16px; letter-spacing: 0.08em; text-transform: uppercase; }}
  .stats {{ display: flex; gap: 12px; justify-content: center;
            margin: 14px 0; flex-wrap: wrap; }}
  .sb  {{ background: #0d1117; border: 1px solid #1e2732;
          border-radius: 4px; padding: 10px 18px; text-align: center;
          position: relative; overflow: hidden; }}
  .sb::before {{ content: ''; position: absolute; top: 0; left: 0;
                 width: 100%; height: 2px; background: #e2b714; }}
  .sv  {{ font-size: 1.5em; color: #e2b714; font-weight: 700; }}
  .sl  {{ font-size: 0.65em; color: #4a5568; margin-top: 3px;
          text-transform: uppercase; letter-spacing: 0.1em; }}
  .acc .sv {{ color: #50c864; }}
  .acc::before {{ background: #50c864; }}
  .leg {{ display: flex; gap: 20px; justify-content: center; margin: 10px 0;
          font-size: 0.72em; flex-wrap: wrap; color: #4a5568; }}
  .li  {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
  .chart-wrap {{ background: #0d1117; border: 1px solid #1e2732;
                 border-radius: 6px; padding: 16px; margin: 10px 0; }}
  canvas {{ max-height: 300px; }}
  .back {{ text-align: center; margin-top: 18px; font-size: 0.8em; }}
  .back a {{ color: #e2b714; text-decoration: none; }}
</style>
</head>
<body>
<h1>USD/INR Next-Day Prediction — {year}</h1>
<div class="sub">KNN Direction Classifier · GARCH(1,1) Magnitude</div>

<div class="stats">
  <div class="sb"><div class="sv">{avg_err}%</div><div class="sl">Avg Abs Error</div></div>
  <div class="sb"><div class="sv">{p65_err}%</div><div class="sl">65th Pct Error</div></div>
  <div class="sb acc"><div class="sv">{dir_acc}%</div><div class="sl">Direction Accuracy</div></div>
  <div class="sb"><div class="sv">{len(v)}</div><div class="sl">Trading Days</div></div>
  <div class="sb"><div class="sv">{n_high}</div><div class="sl">High Vol</div></div>
  <div class="sb"><div class="sv">{n_low}</div><div class="sl">Low Vol</div></div>
  <div class="sb"><div class="sv">{n_med}</div><div class="sl">Medium Vol</div></div>
</div>

<div class="leg">
  <div class="li"><div class="dot" style="background:#ff5050"></div>High vol: full GARCH shift</div>
  <div class="li"><div class="dot" style="background:#50c864"></div>Low vol: flat</div>
  <div class="li"><div class="dot" style="background:#648cff"></div>Medium vol: half GARCH shift</div>
</div>

<div class="chart-wrap"><canvas id="c1"></canvas></div>
<div class="chart-wrap"><canvas id="c2"></canvas></div>
<div class="chart-wrap"><canvas id="c3"></canvas></div>

<div class="back"><a href="{index_file}">← Back to Index</a></div>

<script>
const labels    = {json.dumps(labels)};
const actual    = {json.dumps(actual)};
const predicted = {json.dumps(predicted)};
const errorInr  = {json.dumps(error_inr)};
const errorPct  = {json.dumps(error_pct)};
const ptColors  = {json.dumps(point_colors)};
const grid      = 'rgba(255,255,255,0.04)';
const tick      = '#4a5568';
const base = {{
  responsive: true, animation: false,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ labels: {{ color: '#8b949e',
    font: {{ family: 'JetBrains Mono', size: 11 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: tick, maxTicksLimit: 12, font: {{ size: 10 }} }},
           grid: {{ color: grid }} }},
    y: {{ ticks: {{ color: tick }}, grid: {{ color: grid }} }},
  }}
}};
new Chart(document.getElementById('c1'), {{
  type: 'line',
  data: {{ labels, datasets: [
    {{ label: 'Actual Close', data: actual, borderColor: '#26a69a',
       backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.1 }},
    {{ label: 'Predicted Close (KNN)', data: predicted, borderColor: '#e2b714',
       backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [4,3],
       pointRadius: 3, pointBackgroundColor: ptColors, tension: 0.1 }},
  ]}},
  options: {{ ...base, plugins: {{ ...base.plugins,
    title: {{ display: true, text: 'Actual vs Predicted Close Price',
              color: '#e2b714', font: {{ size: 13, family: 'Syne' }} }} }} }}
}});
const bc = errorInr.map(v => v === null ? 'grey' :
  v >= 0 ? 'rgba(38,166,154,0.7)' : 'rgba(239,83,80,0.7)');
new Chart(document.getElementById('c2'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'Error (INR)', data: errorInr,
    backgroundColor: bc, borderWidth: 0 }}] }},
  options: {{ ...base, plugins: {{ ...base.plugins,
    title: {{ display: true, text: 'Prediction Error: Predicted − Actual (INR)',
              color: '#e2b714', font: {{ size: 13, family: 'Syne' }} }} }} }}
}});
new Chart(document.getElementById('c3'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label: 'Abs Error (%)',
    data: errorPct.map(v => v === null ? null : Math.abs(v)),
    borderColor: '#e377c2', backgroundColor: 'rgba(227,119,194,0.08)',
    borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.1 }}] }},
  options: {{ ...base, plugins: {{ ...base.plugins,
    title: {{ display: true, text: 'Absolute Error (%)',
              color: '#e2b714', font: {{ size: 13, family: 'Syne' }} }} }} }}
}});
</script>
</body>
</html>"""

    fname = f"plot_{plot_num:02d}_{year}.html"
    with open(os.path.join(output_folder, fname), 'w', encoding='utf-8') as f:
        f.write(html)
    return fname


# ──────────────────────────────────────────────────────────────────────────────
# 8. Index pages
# ──────────────────────────────────────────────────────────────────────────────

def _overall_stats(df):
    valid = df.dropna(subset=['predicted_next', 'next_actual'])
    ep    = (valid['predicted_next'] - valid['next_actual']).abs() / valid['next_actual'] * 100
    actual_dir = np.sign(valid['next_actual'].values - valid['Close'].values)
    pred_dir   = valid['direction_pred'].values
    vm = ~np.isnan(pred_dir)
    dir_acc = (pred_dir[vm] == actual_dir[vm]).mean() if vm.sum() > 0 else 0
    return {
        'avg_err': float(ep.mean()),
        'med_err': float(ep.median()),
        'p65_err': float(ep.quantile(0.65)),
        'p90_err': float(ep.quantile(0.90)),
        'n_days':  len(valid),
        'dir_acc': float(dir_acc * 100),
    }


def _build_index_html(title, subtitle, plot_files, summary, overall_stats,
                      feature_importances=None, feature_names=None,
                      other_link=None, other_label=None):
    th    = ''.join(f'<th>{c}</th>' for c in summary.columns)
    tbody = ''
    for _, r in summary.iterrows():
        cells = ''.join(f'<td>{v}</td>' for v in r)
        tbody += f'<tr>{cells}</tr>\n'
    links = '\n'.join(f'<li><a href="{f}" target="_blank">{f}</a></li>' for f in plot_files)
    nav   = (f'<p style="text-align:center;margin:10px 0">'
             f'<a href="{other_link}" style="color:#e2b714">→ {other_label}</a></p>'
             if other_link else '')

    # Feature importance chart
    fi_chart = ''
    if feature_importances is not None and feature_names is not None:
        fi_chart = f"""
<h2>Feature Importances</h2>
<div class="chart-wrap" style="max-width:700px;margin:0 auto 20px">
  <canvas id="fi_chart" style="max-height:320px"></canvas>
</div>
<script>
(function() {{
  const names  = {json.dumps(feature_names)};
  const vals   = {json.dumps([round(float(x),4) for x in feature_importances])};
  const sorted = names.map((n,i) => [n, vals[i]])
                      .sort((a,b) => b[1]-a[1]);
  new Chart(document.getElementById('fi_chart'), {{
    type: 'bar',
    data: {{
      labels: sorted.map(x=>x[0]),
      datasets: [{{ label: 'Importance', data: sorted.map(x=>x[1]),
        backgroundColor: sorted.map((_,i) => `hsla(${{200+i*15}},70%,55%,0.75)`),
        borderWidth: 0 }}]
    }},
    options: {{
      indexAxis: 'y',
      responsive: true, animation: false,
      plugins: {{
        legend: {{ display: false }},
        title: {{ display: true, text: 'Feature Importances',
                  color: '#e2b714', font: {{ size: 14, family: 'Syne' }} }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#4a5568' }}, grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
        y: {{ ticks: {{ color: '#8b949e', font: {{ family: 'JetBrains Mono', size: 11 }} }},
               grid: {{ color: 'rgba(255,255,255,0.04)' }} }},
      }}
    }}
  }});
}})();
</script>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700;800&display=swap');
  * {{ box-sizing: border-box; }}
  body {{ background: #080c10; color: #c9d1d9;
          font-family: 'JetBrains Mono', monospace;
          max-width: 1300px; margin: 40px auto; padding: 0 24px; }}
  h1 {{ text-align: center; color: #e2b714;
        font-family: 'Syne', sans-serif; font-size: 2em;
        letter-spacing: -0.02em; margin-bottom: 4px; }}
  h2 {{ color: #e2b714; font-family: 'Syne', sans-serif;
        border-bottom: 1px solid #1e2732;
        padding-bottom: 6px; margin: 28px 0 12px; font-size: 1.1em; }}
  p  {{ text-align: center; color: #4a5568; margin: 6px 0; font-size: 0.8em;
        text-transform: uppercase; letter-spacing: 0.08em; }}
  .stats {{ display: flex; gap: 16px; justify-content: center;
            margin: 20px 0; flex-wrap: wrap; }}
  .sb {{ background: #0d1117; border: 1px solid #1e2732;
         border-radius: 4px; padding: 14px 22px; text-align: center;
         position: relative; overflow: hidden; }}
  .sb::before {{ content: ''; position: absolute; top: 0; left: 0;
                 width: 100%; height: 2px; background: #e2b714; }}
  .sb.acc::before {{ background: #50c864; }}
  .sv {{ font-size: 1.9em; color: #e2b714; font-weight: 700; }}
  .sb.acc .sv {{ color: #50c864; }}
  .sl {{ font-size: 0.7em; color: #4a5568; margin-top: 4px;
         text-transform: uppercase; letter-spacing: 0.1em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.78em; margin: 10px 0; }}
  th {{ background: #0d1117; color: #e2b714; padding: 10px 14px; text-align: right;
        border: 1px solid #1e2732; white-space: nowrap;
        font-family: 'Syne', sans-serif; }}
  td {{ padding: 8px 14px; text-align: right; border: 1px solid #1a2030; }}
  tr:nth-child(even) {{ background: #0d1117; }}
  tr:hover {{ background: #111820; }}
  ul {{ list-style: none; padding: 0; columns: 3; gap: 10px; }}
  li {{ margin: 8px 0; }}
  a  {{ color: #e2b714; text-decoration: none; }}
  a:hover {{ color: #26a69a; }}
  .chart-wrap {{ background: #0d1117; border: 1px solid #1e2732;
                 border-radius: 6px; padding: 16px; }}
  code {{ background: #0d1117; padding: 2px 6px; border-radius: 3px; color: #79c0ff; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>{subtitle}</p>
{nav}
<div class="stats">
  <div class="sb"><div class="sv">{overall_stats['avg_err']:.4f}%</div><div class="sl">Avg Abs Error</div></div>
  <div class="sb"><div class="sv">{overall_stats['med_err']:.4f}%</div><div class="sl">Median Error</div></div>
  <div class="sb"><div class="sv">{overall_stats['p65_err']:.4f}%</div><div class="sl">65th Pct Error</div></div>
  <div class="sb"><div class="sv">{overall_stats['p90_err']:.4f}%</div><div class="sl">90th Pct Error</div></div>
  <div class="sb acc"><div class="sv">{overall_stats['dir_acc']:.2f}%</div><div class="sl">Direction Accuracy</div></div>
  <div class="sb"><div class="sv">{overall_stats['n_days']:,}</div><div class="sl">Trading Days</div></div>
</div>
{fi_chart}
<h2>Yearly Summary</h2>
<table><thead><tr>{th}</tr></thead><tbody>{tbody}</tbody></table>
<h2>Per-Year Plots</h2>
<ul>{links}</ul>
</body>
</html>"""


def create_index(output_folder, plot_files, summary, overall_stats,
                 feature_importances=None, feature_names=None,
                 other_link=None, other_label=None):
    html = _build_index_html(
        title='USD/INR KNN — Train (2003–2020)',
        subtitle='KNN direction · StandardScaler · GARCH(1,1) magnitude · rolling-std regime · fixed model',
        plot_files=plot_files, summary=summary, overall_stats=overall_stats,
        feature_importances=feature_importances, feature_names=feature_names,
        other_link=other_link, other_label=other_label,
    )
    path = os.path.join(output_folder, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def create_validation_index(output_folder, plot_files, summary, overall_stats,
                             other_link=None, other_label=None):
    html = _build_index_html(
        title='USD/INR KNN — Test (2021–2023)',
        subtitle='Zero lookahead · KNN trained on 2003-2020 only · StandardScaler fitted on train · GARCH + regime identical',
        plot_files=plot_files, summary=summary, overall_stats=overall_stats,
        other_link=other_link, other_label=other_label,
    )
    path = os.path.join(output_folder, 'validation_index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────────────────────────────────────

def main(input_csv='USD_INR_Exchange.csv', output_folder='usd_inr_knn_plots'):
    print(f'\n{"="*65}')
    print('  USD/INR Next-Day Close Predictor — KNN Edition')
    print(f'{"="*65}')

    # Load
    print(f'\nLoading {input_csv} ...')
    raw = pd.read_csv(input_csv)
    raw.columns = raw.columns.str.strip()
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values('Date').reset_index(drop=True).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close'])

    df_train = raw[(raw.index.year >= 2003) & (raw.index.year <= 2020)].copy()
    df_test  = raw[(raw.index.year >= 2021) & (raw.index.year <= 2023)].copy()
    print(f'  Train : {len(df_train):,} days  ({df_train.index[0].date()} → {df_train.index[-1].date()})')
    print(f'  Test  : {len(df_test):,} days  ({df_test.index[0].date()} → {df_test.index[-1].date()})')

    # GARCH on full series
    print('\n[1] Fitting GARCH(1,1) on full series ...')
    full = raw[(raw.index.year >= 2003) & (raw.index.year <= 2023)].copy()
    full['garch_vol'] = fit_garch11(full['Close'].pct_change())

    # Regime
    print('\n[2] Volatility regimes ...')
    full = classify_vol_regime(full)

    # Features
    print('\n[3] Building features ...')
    full = build_features(full)

    feat_cols = ['garch_vol', 'std5', 'std14', 'std25', 'vol_regime'] + FEATURE_COLS
    for col in feat_cols:
        if col in full.columns:
            if col in df_train.columns:
                df_train[col] = full.loc[df_train.index, col]
            else:
                df_train[col] = full.loc[df_train.index, col]
            df_test[col] = full.loc[df_test.index, col]

    df_train['Close'] = full.loc[df_train.index, 'Close']
    df_test['Close']  = full.loc[df_test.index,  'Close']
    df_train['garch_vol'] = full.loc[df_train.index, 'garch_vol']
    df_test['garch_vol']  = full.loc[df_test.index,  'garch_vol']
    df_train['vol_regime'] = full.loc[df_train.index, 'vol_regime']
    df_test['vol_regime']  = full.loc[df_test.index,  'vol_regime']

    for col in FEATURE_COLS + ['target', 'next_close']:
        df_train[col] = full.loc[df_train.index, col]
        df_test[col]  = full.loc[df_test.index,  col]

    # Train KNN
    print('\n[4] Training KNN ...')
    clf, scaler, best_k, cv_acc = train_knn(df_train)

    # Predictions
    print('\n[5] Building train predictions ...')
    df_train = build_predictions(df_train, clf, scaler)

    print('\n[6] Building test predictions (zero lookahead, fixed KNN) ...')
    df_test = build_predictions(df_test, clf, scaler)

    # Results
    print('\n[7] Results — TRAIN (2003–2020)')
    summary_train = compute_summary(df_train)
    print_summary(summary_train, df_train, label='TRAIN SET')

    print('\n[7b] Results — TEST (2021–2023)')
    summary_test = compute_summary(df_test)
    print_summary(summary_test, df_test, label='TEST SET (out-of-sample)')

    # HTML
    print('\n[8] Generating HTML plots ...')
    os.makedirs(output_folder, exist_ok=True)

    train_files = []
    for i, year in enumerate(range(2003, 2021), 1):
        ydf = df_train[df_train.index.year == year].copy()
        if ydf.empty:
            continue
        print(f'  Train {year} ({len(ydf)} days)')
        fname = plot_year_html(ydf, year, output_folder, i, index_file='index.html')
        train_files.append(fname)

    test_files = []
    for i, year in enumerate(range(2021, 2024), 1):
        ydf = df_test[df_test.index.year == year].copy()
        if ydf.empty:
            continue
        print(f'  Test {year} ({len(ydf)} days)')
        fname = plot_year_html(ydf, year, output_folder, i + 100,
                               index_file='validation_index.html')
        test_files.append(fname)

    train_stats = _overall_stats(df_train)
    test_stats  = _overall_stats(df_test)

    create_index(
        output_folder, train_files, summary_train, train_stats,
        other_link='validation_index.html',
        other_label='Go to Test Results (2021–2023)',
    )
    create_validation_index(
        output_folder, test_files, summary_test, test_stats,
        other_link='index.html',
        other_label='Go to Train Results (2003–2020)',
    )

    print(f'\nDone.')
    print(f'  Train index : {output_folder}/index.html')
    print(f'  Test index  : {output_folder}/validation_index.html')
    print(f'{"="*65}\n')


if __name__ == '__main__':
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    out_folder = sys.argv[2] if len(sys.argv) > 2 else 'usd_inr_knn_plots'
    main(csv_path, out_folder)