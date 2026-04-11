"""
USD/INR Next-Day Close Price Predictor -- Gradient Boosting
===========================================================
Model  : HistGradientBoostingRegressor (sklearn)
Val    : Walk-forward -- train on years 2003..Y, predict year Y+1
Target : next day raw Close price

Features
--------
  garch_vol      : GARCH(1,1) conditional std (return units)
  std5/14/25     : rolling price std over 5, 14, 25 days
  feat           : EMA5 - EMA20
  regime_enc     : vol regime  high=2 / medium=1 / low=0
  ret_lag1/2/3   : lagged daily returns
  close_lag1/2/3 : lagged close prices
  rsi14          : RSI(14)
  atr14          : ATR(14)
  dow            : day-of-week (0=Mon, 4=Fri)

Usage
-----
  python gb_predict_usd_inr.py [csv_path] [output_folder]
  Default csv    : USD_INR_Exchange.csv
  Default output : gb_pred_plots/
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =============================================================================
# 1. GARCH(1,1)
# =============================================================================

def _garch_var(r, omega, alpha, beta):
    n = len(r)
    h = np.full(n, max(float(np.var(r)), 1e-10))
    for t in range(1, n):
        h[t] = max(omega + alpha * r[t-1]**2 + beta * h[t-1], 1e-12)
    return h


def fit_garch11(returns):
    r  = returns.fillna(0).values
    uv = max(float(np.var(r)), 1e-10)
    best_nll, best_p = np.inf, None
    for a in np.linspace(0.04, 0.30, 10):
        for b in np.linspace(0.55, 0.92, 10):
            if a + b >= 0.9999:
                continue
            omega = uv * (1 - a - b)
            if omega <= 0:
                continue
            h   = _garch_var(r, omega, a, b)
            nll = 0.5 * float(np.sum(np.log(h) + r**2 / h))
            if nll < best_nll:
                best_nll = nll
                best_p   = (omega, a, b)
    if best_p is None:
        best_p = (uv * 0.05, 0.10, 0.80)
    omega, alpha, beta = best_p
    h = _garch_var(r, omega, alpha, beta)
    print(f"  GARCH(1,1): omega={omega:.2e}  alpha={alpha:.4f}  "
          f"beta={beta:.4f}  persistence={alpha+beta:.4f}")
    return np.sqrt(np.maximum(h, 1e-12))


# =============================================================================
# 2. Direction thresholds (EMA5 - EMA20)
# =============================================================================

def _dir_objective(params, fv, tv, max_abs=0.05):
    gamma, beta = params
    if beta <= gamma:
        return 0.0
    up   = fv >  beta
    down = fv <  gamma
    n_ab = ((fv >= gamma) & (fv <= beta)).sum()
    pen  = max(0.0, (n_ab / len(fv) - max_abs) * 200)
    nc   = up.sum() + down.sum()
    if nc == 0:
        return 0.0
    correct = ((up & (tv == 1)).sum() + (down & (tv == -1)).sum())
    return -(correct / nc - pen)


def learn_thresholds(df):
    v  = df.dropna(subset=['feat', 'close_delta'])
    fv = v['feat'].values
    tv = v['target'].values
    lo, hi = fv.min(), fv.max()
    r  = hi - lo
    res = differential_evolution(
        _dir_objective, [(lo, lo + r*0.6), (lo + r*0.4, hi)],
        args=(fv, tv, 0.05), seed=42, maxiter=2000, popsize=20,
        tol=1e-9, mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )
    gamma, beta = res.x
    print(f"  Direction thresholds: gamma={gamma:.4f}  beta={beta:.4f}")
    return gamma, beta


# =============================================================================
# 3. Feature engineering
# =============================================================================

FEATURE_COLS = [
    'garch_vol',
    'std5', 'std14', 'std25',
    'feat',
    'regime_enc',
    'ret_lag1', 'ret_lag2', 'ret_lag3',
    'close_lag1', 'close_lag2', 'close_lag3',
    'rsi14', 'atr14',
    'dow',
]
TARGET_COL = 'next_close'


def build_features(df, gamma, beta):
    c = df['Close']
    h = df['High']
    l = df['Low']

    # Rolling std + regime
    df['std5']  = c.rolling(5).std()
    df['std14'] = c.rolling(14).std()
    df['std25'] = c.rolling(25).std()
    high = (df['std5'] > df['std14']) & (df['std14'] > df['std25'])
    low  = (df['std5'] < df['std14']) & (df['std14'] < df['std25'])
    df['regime_enc'] = 1
    df.loc[high, 'regime_enc'] = 2
    df.loc[low,  'regime_enc'] = 0

    # EMA feature
    df['ema_5']       = c.ewm(span=5,  adjust=False).mean()
    df['ema_20']      = c.ewm(span=20, adjust=False).mean()
    df['feat']        = df['ema_5'] - df['ema_20']
    df['close_delta'] = c.diff().shift(-1)
    df['target']      = np.where(df['close_delta'] > 0, 1, -1)

    # Lagged returns and prices
    ret = c.pct_change()
    for lag in [1, 2, 3]:
        df[f'ret_lag{lag}']   = ret.shift(lag)
        df[f'close_lag{lag}'] = c.shift(lag)

    # RSI(14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ATR(14)
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()

    # Day of week
    df['dow'] = df.index.dayofweek

    # Target
    df['next_close'] = c.shift(-1)

    return df


# =============================================================================
# 4. Walk-forward training & prediction
# =============================================================================

def walk_forward(df, start_train=2003, start_pred=2004, end_pred=2019):
    df = df.copy()
    df['gb_predicted'] = np.nan

    model_params = dict(
        max_iter=500,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False,
    )

    for pred_year in range(start_pred, end_pred + 1):
        train_mask = (df.index.year >= start_train) & (df.index.year < pred_year)
        pred_mask  =  df.index.year == pred_year

        train = df[train_mask].dropna(subset=FEATURE_COLS + [TARGET_COL])
        pred  = df[pred_mask].dropna(subset=FEATURE_COLS)

        if len(train) < 50 or len(pred) == 0:
            print(f"  {pred_year}: skipped (train={len(train)}, pred={len(pred)})")
            continue

        X_tr = train[FEATURE_COLS].values
        y_tr = train[TARGET_COL].values
        X_pr = pred[FEATURE_COLS].values

        model = HistGradientBoostingRegressor(**model_params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_pr)

        # write back using the index of pred rows that had no NaN features
        df.loc[pred.index, 'gb_predicted'] = preds

        act   = df.loc[pred.index, TARGET_COL].dropna()
        prd_a = df.loc[act.index, 'gb_predicted'].dropna()
        common = act.index.intersection(prd_a.index)
        mae = mean_absolute_error(act[common], prd_a[common]) if len(common) else float('nan')
        print(f"  {pred_year}: train={len(train):4d}  pred={len(pred):3d}  MAE={mae:.4f} INR")

    return df


# =============================================================================
# 5. Results
# =============================================================================

def compute_results(df):
    rows = []
    for year, g in df.groupby(df.index.year):
        if year < 2004:
            continue
        v = g.dropna(subset=['gb_predicted', TARGET_COL])
        if len(v) == 0:
            continue
        act = v[TARGET_COL].values
        prd = v['gb_predicted'].values
        err     = np.abs(prd - act)
        err_pct = err / act * 100
        rows.append({
            'Year':              year,
            'Days':              len(v),
            'MAE (INR)':         round(float(mean_absolute_error(act, prd)), 4),
            'RMSE (INR)':        round(float(np.sqrt(mean_squared_error(act, prd))), 4),
            'Avg Error (%)':     round(float(err_pct.mean()), 4),
            '65th Pct Err (%)':  round(float(np.percentile(err_pct, 65)), 4),
            '90th Pct Err (%)':  round(float(np.percentile(err_pct, 90)), 4),
            'Max Error (%)':     round(float(err_pct.max()), 4),
        })
    return pd.DataFrame(rows)


def print_results(results, df):
    print("\n" + "=" * 88)
    print(f"  {'Year':<6} {'Days':>5} {'MAE':>9} {'RMSE':>9} "
          f"{'AvgErr%':>9} {'65pct%':>8} {'90pct%':>8} {'MaxErr%':>9}")
    print("-" * 88)
    for _, r in results.iterrows():
        print(f"  {int(r['Year']):<6} {int(r['Days']):>5} "
              f"{r['MAE (INR)']:>9.4f} {r['RMSE (INR)']:>9.4f} "
              f"{r['Avg Error (%)']:>9.4f} {r['65th Pct Err (%)']:>8.4f} "
              f"{r['90th Pct Err (%)']:>8.4f} {r['Max Error (%)']:>9.4f}")
    print("=" * 88)

    v   = df[df.index.year >= 2004].dropna(subset=['gb_predicted', TARGET_COL])
    act = v[TARGET_COL].values
    prd = v['gb_predicted'].values
    ep  = np.abs(prd - act) / act * 100
    print(f"\n  OVERALL ({len(v):,} days | 2004-2019)")
    print(f"    MAE           : {mean_absolute_error(act, prd):.4f} INR")
    print(f"    RMSE          : {float(np.sqrt(mean_squared_error(act, prd))):.4f} INR")
    print(f"    Avg abs err   : {ep.mean():.4f}%")
    print(f"    Median err    : {float(np.median(ep)):.4f}%")
    print(f"    65th pct err  : {float(np.percentile(ep, 65)):.4f}%")
    print(f"    90th pct err  : {float(np.percentile(ep, 90)):.4f}%")
    print(f"    Max err       : {ep.max():.4f}%")
    print("=" * 88 + "\n")


# =============================================================================
# 6. HTML plots -- per year
# =============================================================================

def _j(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 4)


def plot_year_html(year_df, year, out_folder, plot_num):
    v = year_df.dropna(subset=['gb_predicted', TARGET_COL])
    if len(v) == 0:
        return None

    labels    = [str(d.date()) for d in v.index]
    actual    = [_j(x) for x in v[TARGET_COL]]
    predicted = [_j(x) for x in v['gb_predicted']]
    err_inr   = [_j(float(p) - float(a)) for p, a in zip(v['gb_predicted'], v[TARGET_COL])]
    err_pct   = [_j((float(p) - float(a)) / float(a) * 100)
                 for p, a in zip(v['gb_predicted'], v[TARGET_COL])]
    abs_ep    = [abs(x) for x in err_pct if x is not None]

    avg_err = round(float(np.mean(abs_ep)), 3)  if abs_ep else 0
    p65_err = round(float(np.percentile(abs_ep, 65)), 3) if abs_ep else 0
    mae_val = round(float(np.mean([abs(e) for e in err_inr if e is not None])), 4)

    rmap = {
        'high':   'rgba(255,80,80,0.75)',
        'low':    'rgba(80,200,100,0.75)',
        'medium': 'rgba(100,140,255,0.75)',
    }
    pt_colors = [rmap.get(str(r), 'grey') for r in v['vol_regime']]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GB USD/INR {year}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:'Courier New',monospace; padding:20px; }}
  h1   {{ text-align:center; color:#58a6ff; font-size:1.4em; margin-bottom:8px; }}
  .stats {{ display:flex; gap:14px; justify-content:center; margin:12px 0; flex-wrap:wrap; }}
  .sb  {{ background:#161b22; border:1px solid #30363d; border-radius:6px;
          padding:10px 18px; text-align:center; }}
  .sv  {{ font-size:1.5em; color:#ff8c00; font-weight:bold; }}
  .sl  {{ font-size:0.72em; color:#8b949e; margin-top:2px; }}
  .leg {{ display:flex; gap:18px; justify-content:center; margin:8px 0;
          font-size:0.78em; flex-wrap:wrap; }}
  .li  {{ display:flex; align-items:center; gap:5px; }}
  .dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  .cw  {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
          padding:14px; margin:10px 0; }}
  canvas {{ max-height:300px; }}
  .back {{ text-align:center; margin-top:14px; }}
  .back a {{ color:#58a6ff; text-decoration:none; }}
</style>
</head>
<body>
<h1>Gradient Boosting &mdash; USD/INR Next-Day Close &mdash; {year}</h1>

<div class="stats">
  <div class="sb"><div class="sv">{avg_err}%</div><div class="sl">Avg Abs Error</div></div>
  <div class="sb"><div class="sv">{p65_err}%</div><div class="sl">65th Pct Error</div></div>
  <div class="sb"><div class="sv">{mae_val}</div><div class="sl">MAE (INR)</div></div>
  <div class="sb"><div class="sv">{len(v)}</div><div class="sl">Days Predicted</div></div>
</div>

<div class="leg">
  <div class="li"><div class="dot" style="background:#26a69a"></div>Actual Close</div>
  <div class="li"><div class="dot" style="background:#ff8c00"></div>GB Predicted</div>
  <div class="li"><div class="dot" style="background:#ff5050"></div>High vol</div>
  <div class="li"><div class="dot" style="background:#50c864"></div>Low vol</div>
  <div class="li"><div class="dot" style="background:#648cff"></div>Medium vol</div>
</div>

<div class="cw"><canvas id="c1"></canvas></div>
<div class="cw"><canvas id="c2"></canvas></div>
<div class="cw"><canvas id="c3"></canvas></div>

<div class="back"><a href="index.html">&larr; Back to Index</a></div>

<script>
const labels    = {json.dumps(labels)};
const actual    = {json.dumps(actual)};
const predicted = {json.dumps(predicted)};
const errInr    = {json.dumps(err_inr)};
const errPct    = {json.dumps(err_pct)};
const ptColors  = {json.dumps(pt_colors)};

const base = {{
  responsive:true, animation:false,
  interaction:{{ mode:'index', intersect:false }},
  plugins:{{ legend:{{ labels:{{ color:'#c9d1d9', font:{{ family:'Courier New', size:11 }} }} }} }},
  scales:{{
    x:{{ ticks:{{ color:'#8b949e', maxTicksLimit:10, font:{{size:9}} }},
         grid:{{ color:'rgba(255,255,255,0.05)' }} }},
    y:{{ ticks:{{ color:'#8b949e' }}, grid:{{ color:'rgba(255,255,255,0.05)' }} }}
  }}
}};

new Chart(document.getElementById('c1'), {{
  type:'line',
  data:{{ labels, datasets:[
    {{ label:'Actual Close (INR)', data:actual,
       borderColor:'#26a69a', backgroundColor:'transparent',
       borderWidth:2, pointRadius:0, tension:0.1 }},
    {{ label:'GB Predicted (INR)', data:predicted,
       borderColor:'#ff8c00', backgroundColor:'transparent',
       borderWidth:1.5, borderDash:[5,3],
       pointRadius:3, pointBackgroundColor:ptColors, tension:0.1 }},
  ]}},
  options:{{ ...base, plugins:{{ ...base.plugins,
    title:{{ display:true, text:'Actual vs GB Predicted Close',
             color:'#58a6ff', font:{{size:13}} }} }} }}
}});

const barC = errInr.map(v => v === null ? 'grey' : v >= 0 ? 'rgba(38,166,154,0.7)' : 'rgba(239,83,80,0.7)');
new Chart(document.getElementById('c2'), {{
  type:'bar',
  data:{{ labels, datasets:[
    {{ label:'Error (INR)', data:errInr, backgroundColor:barC, borderWidth:0 }}
  ]}},
  options:{{ ...base, plugins:{{ ...base.plugins,
    title:{{ display:true, text:'Prediction Error: GB - Actual (INR)',
             color:'#58a6ff', font:{{size:13}} }} }} }}
}});

new Chart(document.getElementById('c3'), {{
  type:'line',
  data:{{ labels, datasets:[
    {{ label:'Abs Error (%)',
       data:errPct.map(v => v === null ? null : Math.abs(v)),
       borderColor:'#e377c2', backgroundColor:'rgba(227,119,194,0.12)',
       borderWidth:1.5, pointRadius:0, fill:true, tension:0.1 }}
  ]}},
  options:{{ ...base, plugins:{{ ...base.plugins,
    title:{{ display:true, text:'Absolute Error (%)',
             color:'#58a6ff', font:{{size:13}} }} }} }}
}});
</script>
</body>
</html>"""

    fname = f"plot_{plot_num:02d}_{year}.html"
    with open(os.path.join(out_folder, fname), 'w', encoding='utf-8') as f:
        f.write(html)
    return fname


# =============================================================================
# 7. Feature importance (permutation on 2019 holdout)
# =============================================================================

def plot_feature_importance(df, out_folder):
    from sklearn.inspection import permutation_importance

    train = df[(df.index.year >= 2003) & (df.index.year <= 2018)].dropna(
        subset=FEATURE_COLS + [TARGET_COL])
    test  = df[df.index.year == 2019].dropna(subset=FEATURE_COLS + [TARGET_COL])

    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=5,
        min_samples_leaf=20, l2_regularization=1.0,
        random_state=42, early_stopping=False,
    )
    model.fit(train[FEATURE_COLS].values, train[TARGET_COL].values)

    result = permutation_importance(
        model,
        test[FEATURE_COLS].values,
        test[TARGET_COL].values,
        n_repeats=20,
        random_state=42,
        scoring='neg_mean_absolute_error',
    )

    order        = np.argsort(result.importances_mean)[::-1]
    feats_sorted = [FEATURE_COLS[i] for i in order]
    imp_sorted   = result.importances_mean[order].tolist()
    std_sorted   = result.importances_std[order].tolist()

    print("\n  Permutation Feature Importance (2019 holdout):")
    for f, im, is_ in zip(feats_sorted, imp_sorted, std_sorted):
        print(f"    {f:<15} {im:+.6f}  (+/- {is_:.6f})")

    colors = ['#ff8c00' if v >= 0 else '#ef5350' for v in imp_sorted]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Feature Importance</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:'Courier New',monospace; padding:30px; }}
  h1   {{ text-align:center; color:#58a6ff; margin-bottom:6px; }}
  p    {{ text-align:center; color:#8b949e; font-size:0.82em; margin-bottom:20px; }}
  .cw  {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
          padding:20px; max-width:860px; margin:0 auto; }}
  canvas {{ max-height:440px; }}
  .back {{ text-align:center; margin-top:18px; }}
  .back a {{ color:#58a6ff; text-decoration:none; }}
</style>
</head>
<body>
<h1>Feature Importance &mdash; Permutation (2019 holdout)</h1>
<p>Mean MAE increase when each feature is randomly shuffled &mdash; higher = more important.</p>
<div class="cw"><canvas id="fi"></canvas></div>
<div class="back"><a href="index.html">&larr; Back to Index</a></div>
<script>
new Chart(document.getElementById('fi'), {{
  type:'bar',
  data:{{
    labels: {json.dumps(feats_sorted)},
    datasets: [{{
      label: 'Permutation Importance (MAE delta)',
      data:  {json.dumps([round(x, 6) for x in imp_sorted])},
      backgroundColor: {json.dumps(colors)},
      borderWidth: 0,
    }}]
  }},
  options:{{
    indexAxis:'y',
    responsive:true, animation:false,
    plugins:{{
      legend:{{ display:false }},
      title:{{ display:true, text:'Permutation Feature Importance (2019 holdout)',
               color:'#58a6ff', font:{{size:14}} }}
    }},
    scales:{{
      x:{{ ticks:{{ color:'#8b949e' }}, grid:{{ color:'rgba(255,255,255,0.06)' }},
           title:{{ display:true, text:'Mean MAE increase', color:'#8b949e' }} }},
      y:{{ ticks:{{ color:'#c9d1d9', font:{{ family:'Courier New', size:11 }} }},
           grid:{{ color:'rgba(255,255,255,0.04)' }} }},
    }}
  }}
}});
</script>
</body>
</html>"""

    fpath = os.path.join(out_folder, 'feature_importance.html')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  Saved: {fpath}")
    return 'feature_importance.html'


# =============================================================================
# 8. Index page
# =============================================================================

def create_index(out_folder, plot_files, fi_file, results, overall):
    th    = ''.join(f'<th>{c}</th>' for c in results.columns)
    tbody = ''
    for _, r in results.iterrows():
        cells = ''.join(f'<td>{v}</td>' for v in r)
        tbody += f'<tr>{cells}</tr>\n'

    links    = '\n'.join(
        f'<li><a href="{f}" target="_blank">{f}</a></li>' for f in plot_files)
    feat_lis = ''.join(f'<li><code>{f}</code></li>' for f in FEATURE_COLS)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GB USD/INR 2004-2019</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ background:#0d1117; color:#c9d1d9; font-family:'Courier New',monospace;
          max-width:1300px; margin:40px auto; padding:0 24px; }}
  h1   {{ text-align:center; color:#58a6ff; font-size:1.9em; margin-bottom:4px; }}
  h2   {{ color:#58a6ff; border-bottom:1px solid #30363d; padding-bottom:5px;
          margin:26px 0 12px; }}
  p    {{ text-align:center; color:#8b949e; margin:5px 0; font-size:0.87em; }}
  .stats {{ display:flex; gap:18px; justify-content:center; margin:20px 0; flex-wrap:wrap; }}
  .sb  {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
          padding:14px 22px; text-align:center; }}
  .sv  {{ font-size:1.8em; color:#ff8c00; font-weight:bold; }}
  .sl  {{ font-size:0.75em; color:#8b949e; margin-top:4px; }}
  table {{ border-collapse:collapse; width:100%; font-size:0.81em; margin:10px 0; }}
  th   {{ background:#161b22; color:#58a6ff; padding:9px 13px; text-align:right;
          border:1px solid #30363d; white-space:nowrap; }}
  td   {{ padding:7px 13px; text-align:right; border:1px solid #21262d; }}
  tr:nth-child(even) {{ background:#161b22; }}
  tr:hover {{ background:#1f2937; }}
  ul.plots {{ list-style:none; padding:0; columns:3; gap:10px; }}
  ul.feats {{ list-style:none; padding:0; display:flex; flex-wrap:wrap; gap:8px; margin:8px 0; }}
  li   {{ margin:5px 0; }}
  a    {{ color:#58a6ff; text-decoration:none; }}
  a:hover {{ color:#ff8c00; }}
  code {{ background:#161b22; padding:3px 8px; border-radius:4px;
          color:#79c0ff; font-size:0.88em; }}
  .fi-link {{ text-align:center; margin:10px 0; }}
</style>
</head>
<body>
<h1>USD/INR &mdash; Gradient Boosting Next-Day Close</h1>
<p>HistGradientBoostingRegressor &nbsp;&middot;&nbsp;
   Walk-forward validation 2004&ndash;2019 &nbsp;&middot;&nbsp;
   Train on all prior years, predict next year</p>

<div class="stats">
  <div class="sb"><div class="sv">{overall['mae']:.4f}</div><div class="sl">Overall MAE (INR)</div></div>
  <div class="sb"><div class="sv">{overall['rmse']:.4f}</div><div class="sl">Overall RMSE (INR)</div></div>
  <div class="sb"><div class="sv">{overall['avg_pct']:.4f}%</div><div class="sl">Avg Abs Error %</div></div>
  <div class="sb"><div class="sv">{overall['p65']:.4f}%</div><div class="sl">65th Pct Error %</div></div>
  <div class="sb"><div class="sv">{overall['p90']:.4f}%</div><div class="sl">90th Pct Error %</div></div>
  <div class="sb"><div class="sv">{overall['n']:,}</div><div class="sl">Total Days (2004-2019)</div></div>
</div>

<h2>Features ({len(FEATURE_COLS)} total)</h2>
<ul class="feats">{feat_lis}</ul>

<h2>Feature Importance</h2>
<div class="fi-link">
  <a href="{fi_file}" target="_blank">Open Feature Importance Chart &rarr;</a>
</div>

<h2>Yearly Results</h2>
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>{tbody}</tbody>
</table>

<h2>Per-Year Plots</h2>
<p>Actual vs Predicted &middot; Error bar &middot; Abs Error % &mdash; dots coloured by vol regime</p>
<ul class="plots">{links}</ul>
</body>
</html>"""

    path = os.path.join(out_folder, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


# =============================================================================
# 9. Main
# =============================================================================

def main(input_csv='USD_INR_Exchange.csv', out_folder='gb_pred_plots'):
    print(f'\n{"="*60}')
    print('  USD/INR Gradient Boosting Predictor')
    print(f'{"="*60}')

    # Load
    print(f'\nLoading {input_csv} ...')
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df[(df.index.year >= 2003) & (df.index.year <= 2019)]
    print(f'  {len(df):,} days  ({df.index[0].date()} -> {df.index[-1].date()})')

    # GARCH
    print('\n[1] Fitting GARCH(1,1) ...')
    df['garch_vol'] = fit_garch11(df['Close'].pct_change())

    # Direction thresholds
    print('\n[2] Learning direction thresholds ...')
    tmp = df.copy()
    tmp['ema_5']       = tmp['Close'].ewm(span=5,  adjust=False).mean()
    tmp['ema_20']      = tmp['Close'].ewm(span=20, adjust=False).mean()
    tmp['feat']        = tmp['ema_5'] - tmp['ema_20']
    tmp['close_delta'] = tmp['Close'].diff().shift(-1)
    tmp['target']      = np.where(tmp['close_delta'] > 0, 1, -1)
    gamma, beta = learn_thresholds(tmp)

    # Features
    print('\n[3] Building features ...')
    df = build_features(df, gamma, beta)
    rmap_inv = {2: 'high', 1: 'medium', 0: 'low'}
    df['vol_regime'] = df['regime_enc'].map(rmap_inv)
    rc = df['vol_regime'].value_counts()
    for r in ['high', 'medium', 'low']:
        n = rc.get(r, 0)
        print(f'  {r:8s}: {n:5d} days ({n/len(df)*100:.1f}%)')

    # Walk-forward
    print('\n[4] Walk-forward training & prediction (2004-2019) ...')
    df = walk_forward(df)

    # Results
    print('\n[5] Results ...')
    results = compute_results(df)
    print_results(results, df)

    v   = df[df.index.year >= 2004].dropna(subset=['gb_predicted', TARGET_COL])
    act = v[TARGET_COL].values
    prd = v['gb_predicted'].values
    ep  = np.abs(prd - act) / act * 100
    overall = {
        'mae':     float(mean_absolute_error(act, prd)),
        'rmse':    float(np.sqrt(mean_squared_error(act, prd))),
        'avg_pct': float(ep.mean()),
        'p65':     float(np.percentile(ep, 65)),
        'p90':     float(np.percentile(ep, 90)),
        'n':       len(v),
    }

    # Feature importance
    print('\n[6] Permutation feature importance ...')
    os.makedirs(out_folder, exist_ok=True)
    fi_file = plot_feature_importance(df, out_folder)

    # Plots
    print('\n[7] Generating HTML plots ...')
    plot_files = []
    for i, year in enumerate(range(2004, 2020), 1):
        ydf = df[df.index.year == year].copy()
        if ydf.empty:
            continue
        fname = plot_year_html(ydf, year, out_folder, i)
        if fname:
            plot_files.append(fname)
            print(f'  {year} -> {fname}')

    # Index
    idx = create_index(out_folder, plot_files, fi_file, results, overall)
    print(f'\nDone -- {len(plot_files)} plots + feature importance saved to "{out_folder}/"')
    print(f'Open : {idx}\n')


if __name__ == '__main__':
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    out_folder = sys.argv[2] if len(sys.argv) > 2 else 'gb_pred_plots'
    main(csv_path, out_folder)