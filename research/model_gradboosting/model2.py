"""
USD/INR Next-Day Close Price Predictor -- Gradient Boosting
===========================================================
Model   : HistGradientBoostingRegressor (sklearn)
Train   : 2003-2019  (gamma/beta also learned on this window)
Validate: 2020, 2021 (model has never seen these)

Key details:
  - Features for 2020-2021 are computed on the full series so rolling
    windows at Jan 2020 are correctly seeded by late-2019 prices.
  - Last row of 2019 (whose next_close bleeds into 2020) is dropped
    from the training target to avoid any lookahead.
  - gamma/beta for the EMA direction feature are fit on 2003-2019 only.

Features (15)
-------------
  garch_vol      GARCH(1,1) conditional std (return units)
  std5           rolling price std, 5 days
  std14          rolling price std, 14 days
  std25          rolling price std, 25 days
  feat           EMA5 - EMA20
  regime_enc     vol regime: high=2 / medium=1 / low=0
  ret_lag1/2/3   lagged daily returns (t-1, t-2, t-3)
  close_lag1/2/3 lagged close prices  (t-1, t-2, t-3)
  rsi14          RSI(14)
  atr14          ATR(14)
  dow            day-of-week (0=Mon, 4=Fri)

Usage
-----
  python gb_predict_usd_inr.py [csv_path] [output_folder]
  Defaults: USD_INR_Exchange.csv   gb_pred_plots/
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
# 1.  GARCH(1,1)
# =============================================================================

def _garch_var(r, omega, alpha, beta):
    n = len(r)
    h = np.full(n, max(float(np.var(r)), 1e-10))
    for t in range(1, n):
        h[t] = max(omega + alpha * r[t - 1] ** 2 + beta * h[t - 1], 1e-12)
    return h


def fit_garch11(returns):
    """Return conditional std-dev series in return-fraction units."""
    r  = returns.fillna(0).values
    uv = max(float(np.var(r)), 1e-10)
    best_nll, best_p = np.inf, None

    for a in np.linspace(0.04, 0.30, 10):
        for b in np.linspace(0.55, 0.92, 10):
            if a + b >= 0.9999:
                continue
            omega = uv * (1.0 - a - b)
            if omega <= 0:
                continue
            h   = _garch_var(r, omega, a, b)
            nll = 0.5 * float(np.sum(np.log(h) + r ** 2 / h))
            if nll < best_nll:
                best_nll = nll
                best_p   = (omega, a, b)

    if best_p is None:
        best_p = (uv * 0.05, 0.10, 0.80)

    omega, alpha, beta = best_p
    h = _garch_var(r, omega, alpha, beta)
    print(f"  GARCH(1,1):  omega={omega:.2e}  alpha={alpha:.4f}  "
          f"beta={beta:.4f}  persistence={alpha + beta:.4f}")
    return np.sqrt(np.maximum(h, 1e-12))


# =============================================================================
# 2.  Direction thresholds  (EMA5 - EMA20, learned on 2003-2019 only)
# =============================================================================

def _dir_obj(params, fv, tv, max_abs=0.05):
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
    correct = (up & (tv == 1)).sum() + (down & (tv == -1)).sum()
    return -(correct / nc - pen)


def learn_thresholds(df_train):
    """Fit gamma/beta on the supplied slice (2003-2019)."""
    v  = df_train.dropna(subset=['feat', 'close_delta'])
    fv = v['feat'].values
    tv = v['target'].values
    lo, hi = fv.min(), fv.max()
    rng    = hi - lo
    result = differential_evolution(
        _dir_obj,
        bounds=[(lo, lo + rng * 0.6), (lo + rng * 0.4, hi)],
        args=(fv, tv, 0.05),
        seed=42, maxiter=2000, popsize=20, tol=1e-9,
        mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )
    gamma, beta = result.x
    print(f"  Direction thresholds:  gamma={gamma:.4f}  beta={beta:.4f}")
    return gamma, beta


# =============================================================================
# 3.  Feature engineering  (run on FULL series so rolling windows are correct)
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


def build_features(df):
    """Add all feature columns to df in-place (operates on full date range)."""
    c = df['Close']
    h = df['High']
    l = df['Low']

    # -- Rolling std + vol regime --
    df['std5']  = c.rolling(5).std()
    df['std14'] = c.rolling(14).std()
    df['std25'] = c.rolling(25).std()
    high = (df['std5'] > df['std14']) & (df['std14'] > df['std25'])
    low  = (df['std5'] < df['std14']) & (df['std14'] < df['std25'])
    df['regime_enc'] = 1          # medium default
    df.loc[high, 'regime_enc'] = 2
    df.loc[low,  'regime_enc'] = 0

    # -- EMA direction feature --
    df['ema_5']       = c.ewm(span=5,  adjust=False).mean()
    df['ema_20']      = c.ewm(span=20, adjust=False).mean()
    df['feat']        = df['ema_5'] - df['ema_20']
    df['close_delta'] = c.diff().shift(-1)   # used only for threshold learning
    df['target']      = np.where(df['close_delta'] > 0, 1, -1)

    # -- Lagged returns & prices --
    ret = c.pct_change()
    for lag in [1, 2, 3]:
        df[f'ret_lag{lag}']   = ret.shift(lag)
        df[f'close_lag{lag}'] = c.shift(lag)

    # -- RSI(14) --
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi14'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # -- ATR(14) --
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()

    # -- Day of week --
    df['dow'] = df.index.dayofweek

    # -- Target --
    df[TARGET_COL] = c.shift(-1)

    return df


# =============================================================================
# 4.  Train on 2003-2019, predict 2020-2021
# =============================================================================

def train_and_predict(df):
    """
    Train one GB model on 2003-2019 (excluding the last row whose
    next_close bleeds into 2020).  Predict 2020 and 2021 separately.
    """
    train_raw = df[df.index.year <= 2019].copy()

    # Drop last row of 2019 -- its next_close is the first 2020 price
    train_raw = train_raw.iloc[:-1]

    train = train_raw.dropna(subset=FEATURE_COLS + [TARGET_COL])

    model = HistGradientBoostingRegressor(
        max_iter        = 600,
        learning_rate   = 0.05,
        max_depth       = 5,
        min_samples_leaf= 20,
        l2_regularization = 1.0,
        random_state    = 42,
        early_stopping  = False,
    )
    print(f"  Training on {len(train):,} rows  "
          f"({train.index[0].date()} -> {train.index[-1].date()})")
    model.fit(train[FEATURE_COLS].values, train[TARGET_COL].values)

    df = df.copy()
    df['gb_predicted'] = np.nan

    for year in [2020, 2021]:
        val = df[df.index.year == year].dropna(subset=FEATURE_COLS)
        if val.empty:
            print(f"  {year}: no data found -- skipping")
            continue
        preds = model.predict(val[FEATURE_COLS].values)
        df.loc[val.index, 'gb_predicted'] = preds

        # quick per-year MAE at predict time
        act_vals = df.loc[val.index, TARGET_COL].dropna()
        common   = act_vals.index.intersection(val.index)
        if len(common):
            mae = mean_absolute_error(act_vals[common],
                                      df.loc[common, 'gb_predicted'])
            print(f"  {year}: {len(val):3d} days predicted  |  MAE = {mae:.4f} INR")
        else:
            print(f"  {year}: {len(val):3d} days predicted  |  (no actuals to compare)")

    return df, model


# =============================================================================
# 5.  Feature importance  (permutation on 2020-2021 validation set)
# =============================================================================

def compute_feature_importance(model, df, out_folder):
    from sklearn.inspection import permutation_importance

    val = df[df.index.year.isin([2020, 2021])].dropna(
        subset=FEATURE_COLS + [TARGET_COL])

    if len(val) == 0:
        print("  No validation data for feature importance -- skipping")
        return None

    result = permutation_importance(
        model,
        val[FEATURE_COLS].values,
        val[TARGET_COL].values,
        n_repeats   = 30,
        random_state= 42,
        scoring     = 'neg_mean_absolute_error',
    )

    order        = np.argsort(result.importances_mean)[::-1]
    feats_sorted = [FEATURE_COLS[i] for i in order]
    imp_sorted   = result.importances_mean[order].tolist()
    std_sorted   = result.importances_std[order].tolist()

    print("\n  Permutation Feature Importance (2020-2021 validation):")
    for f, im, is_ in zip(feats_sorted, imp_sorted, std_sorted):
        bar = '+' * max(1, int(abs(im) * 300))
        print(f"    {f:<15}  {im:+.6f}  (+/- {is_:.6f})  {bar}")

    colors = ['#ff8c00' if v >= 0 else '#ef5350' for v in imp_sorted]

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Feature Importance</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:#0d1117; color:#c9d1d9;
          font-family:'Courier New',monospace; padding:30px; }}
  h1 {{ text-align:center; color:#58a6ff; margin-bottom:6px; }}
  p  {{ text-align:center; color:#8b949e; font-size:0.82em; margin-bottom:20px; }}
  .cw {{ background:#161b22; border:1px solid #30363d; border-radius:8px;
         padding:20px; max-width:880px; margin:0 auto; }}
  canvas {{ max-height:460px; }}
  .back {{ text-align:center; margin-top:18px; }}
  .back a {{ color:#58a6ff; text-decoration:none; }}
  .back a:hover {{ color:#ff8c00; }}
</style>
</head>
<body>
<h1>Feature Importance &mdash; Permutation (2020&ndash;2021 validation)</h1>
<p>Mean MAE increase when each feature is randomly shuffled.
   Higher = more important. Evaluated on out-of-sample 2020&ndash;2021 data.</p>
<div class="cw"><canvas id="fi"></canvas></div>
<div class="back"><a href="index.html">&larr; Back to Index</a></div>
<script>
new Chart(document.getElementById('fi'), {{
  type: 'bar',
  data: {{
    labels:   {json.dumps(feats_sorted)},
    datasets: [{{
      label: 'Permutation Importance (MAE delta)',
      data:  {json.dumps([round(x, 6) for x in imp_sorted])},
      backgroundColor: {json.dumps(colors)},
      borderWidth: 0,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    animation:  false,
    plugins: {{
      legend: {{ display: false }},
      title:  {{ display: true,
                 text: 'Permutation Feature Importance (2020-2021 OOS)',
                 color: '#58a6ff', font: {{ size: 14 }} }}
    }},
    scales: {{
      x: {{ ticks: {{ color:'#8b949e' }},
             grid:  {{ color:'rgba(255,255,255,0.06)' }},
             title: {{ display:true, text:'Mean MAE increase', color:'#8b949e' }} }},
      y: {{ ticks: {{ color:'#c9d1d9',
                      font:{{ family:'Courier New', size:11 }} }},
             grid:  {{ color:'rgba(255,255,255,0.04)' }} }}
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
# 6.  Results summary
# =============================================================================

def compute_results(df):
    rows = []
    for year in [2020, 2021, 'combined']:
        if year == 'combined':
            v = df[df.index.year.isin([2020, 2021])].dropna(
                subset=['gb_predicted', TARGET_COL])
            yr_label = 'Combined'
        else:
            v = df[df.index.year == year].dropna(
                subset=['gb_predicted', TARGET_COL])
            yr_label = year

        if len(v) == 0:
            continue

        act     = v[TARGET_COL].values
        prd     = v['gb_predicted'].values
        err_inr = np.abs(prd - act)
        err_pct = err_inr / act * 100

        rows.append({
            'Period':            yr_label,
            'Days':              len(v),
            'MAE (INR)':         round(float(mean_absolute_error(act, prd)), 4),
            'RMSE (INR)':        round(float(np.sqrt(mean_squared_error(act, prd))), 4),
            'Avg Error (%)':     round(float(err_pct.mean()), 4),
            'Median Error (%)':  round(float(np.median(err_pct)), 4),
            '65th Pct Err (%)':  round(float(np.percentile(err_pct, 65)), 4),
            '90th Pct Err (%)':  round(float(np.percentile(err_pct, 90)), 4),
            'Max Error (%)':     round(float(err_pct.max()), 4),
        })
    return pd.DataFrame(rows)


def print_results(results):
    print("\n" + "=" * 95)
    print(f"  {'Period':<10} {'Days':>5} {'MAE':>9} {'RMSE':>9} "
          f"{'Avg%':>7} {'Med%':>7} {'65pct%':>8} {'90pct%':>8} {'Max%':>8}")
    print("-" * 95)
    for _, r in results.iterrows():
        print(f"  {str(r['Period']):<10} {int(r['Days']):>5} "
              f"{r['MAE (INR)']:>9.4f} {r['RMSE (INR)']:>9.4f} "
              f"{r['Avg Error (%)']:>7.4f} {r['Median Error (%)']:>7.4f} "
              f"{r['65th Pct Err (%)']:>8.4f} {r['90th Pct Err (%)']:>8.4f} "
              f"{r['Max Error (%)']:>8.4f}")
    print("=" * 95 + "\n")


# =============================================================================
# 7.  HTML plots  (one per validation year)
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
    err_inr   = [_j(float(p) - float(a))
                 for p, a in zip(v['gb_predicted'], v[TARGET_COL])]
    err_pct   = [_j((float(p) - float(a)) / float(a) * 100)
                 for p, a in zip(v['gb_predicted'], v[TARGET_COL])]
    abs_ep    = [abs(x) for x in err_pct if x is not None]

    avg_err = round(float(np.mean(abs_ep)), 3)            if abs_ep else 0
    p65_err = round(float(np.percentile(abs_ep, 65)), 3)  if abs_ep else 0
    mae_val = round(float(np.mean(
                [abs(e) for e in err_inr if e is not None])), 4)
    rmse_val= round(float(np.sqrt(np.mean(
                [(e**2) for e in err_inr if e is not None]))), 4)

    rmap = {
        'high':   'rgba(255,80,80,0.80)',
        'low':    'rgba(80,200,100,0.80)',
        'medium': 'rgba(100,140,255,0.80)',
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
  body {{ background:#0d1117; color:#c9d1d9;
          font-family:'Courier New',monospace; padding:20px; }}
  h1  {{ text-align:center; color:#58a6ff; font-size:1.4em; margin-bottom:8px; }}
  .tag {{ display:inline-block; background:#1f3a5f; color:#79c0ff;
          border:1px solid #2d5986; border-radius:4px;
          font-size:0.75em; padding:2px 8px; margin-left:8px; vertical-align:middle; }}
  .stats {{ display:flex; gap:14px; justify-content:center;
            margin:12px 0; flex-wrap:wrap; }}
  .sb {{ background:#161b22; border:1px solid #30363d; border-radius:6px;
         padding:10px 18px; text-align:center; }}
  .sv {{ font-size:1.5em; color:#ff8c00; font-weight:bold; }}
  .sl {{ font-size:0.72em; color:#8b949e; margin-top:2px; }}
  .leg {{ display:flex; gap:18px; justify-content:center;
          margin:8px 0; font-size:0.78em; flex-wrap:wrap; }}
  .li {{ display:flex; align-items:center; gap:5px; }}
  .dot {{ width:11px; height:11px; border-radius:50%; flex-shrink:0; }}
  .cw {{ background:#161b22; border:1px solid #30363d;
         border-radius:8px; padding:14px; margin:10px 0; }}
  canvas {{ max-height:300px; }}
  .back {{ text-align:center; margin-top:14px; }}
  .back a {{ color:#58a6ff; text-decoration:none; }}
  .back a:hover {{ color:#ff8c00; }}
</style>
</head>
<body>
<h1>
  USD/INR Next-Day Close &mdash; {year}
  <span class="tag">OUT-OF-SAMPLE VALIDATION</span>
</h1>

<div class="stats">
  <div class="sb"><div class="sv">{avg_err}%</div><div class="sl">Avg Abs Error</div></div>
  <div class="sb"><div class="sv">{p65_err}%</div><div class="sl">65th Pct Error</div></div>
  <div class="sb"><div class="sv">{mae_val}</div><div class="sl">MAE (INR)</div></div>
  <div class="sb"><div class="sv">{rmse_val}</div><div class="sl">RMSE (INR)</div></div>
  <div class="sb"><div class="sv">{len(v)}</div><div class="sl">Days Validated</div></div>
</div>

<div class="leg">
  <div class="li"><div class="dot" style="background:#26a69a"></div>Actual Next Close</div>
  <div class="li"><div class="dot" style="background:#ff8c00"></div>GB Predicted</div>
  <div class="li"><div class="dot" style="background:#ff5050"></div>High vol day</div>
  <div class="li"><div class="dot" style="background:#50c864"></div>Low vol day</div>
  <div class="li"><div class="dot" style="background:#648cff"></div>Medium vol day</div>
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
  responsive: true, animation: false,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ labels: {{ color:'#c9d1d9',
                                    font:{{ family:'Courier New', size:11 }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color:'#8b949e', maxTicksLimit:10, font:{{size:9}} }},
           grid:  {{ color:'rgba(255,255,255,0.05)' }} }},
    y: {{ ticks: {{ color:'#8b949e' }},
           grid:  {{ color:'rgba(255,255,255,0.05)' }} }}
  }}
}};

// Chart 1 -- Actual vs Predicted
new Chart(document.getElementById('c1'), {{
  type: 'line',
  data: {{ labels, datasets: [
    {{ label: 'Actual Next Close (INR)', data: actual,
       borderColor:'#26a69a', backgroundColor:'transparent',
       borderWidth:2, pointRadius:0, tension:0.1 }},
    {{ label: 'GB Predicted (INR)', data: predicted,
       borderColor:'#ff8c00', backgroundColor:'transparent',
       borderWidth:1.5, borderDash:[5,3],
       pointRadius:3, pointBackgroundColor:ptColors, tension:0.1 }},
  ]}},
  options: {{ ...base, plugins: {{ ...base.plugins,
    title: {{ display:true,
              text:'Actual vs GB Predicted Close Price (Out-of-Sample)',
              color:'#58a6ff', font:{{size:13}} }} }} }}
}});

// Chart 2 -- Error bar
const barC = errInr.map(
  v => v === null ? 'grey' : v >= 0 ? 'rgba(38,166,154,0.7)' : 'rgba(239,83,80,0.7)');
new Chart(document.getElementById('c2'), {{
  type: 'bar',
  data: {{ labels, datasets: [
    {{ label:'Prediction Error (INR)', data:errInr,
       backgroundColor:barC, borderWidth:0 }}
  ]}},
  options: {{ ...base, plugins: {{ ...base.plugins,
    title: {{ display:true, text:'Prediction Error: GB - Actual (INR)',
              color:'#58a6ff', font:{{size:13}} }} }} }}
}});

// Chart 3 -- Abs error %
new Chart(document.getElementById('c3'), {{
  type: 'line',
  data: {{ labels, datasets: [
    {{ label:'Abs Error (%)',
       data: errPct.map(v => v === null ? null : Math.abs(v)),
       borderColor:'#e377c2', backgroundColor:'rgba(227,119,194,0.12)',
       borderWidth:1.5, pointRadius:0, fill:true, tension:0.1 }}
  ]}},
  options: {{ ...base, plugins: {{ ...base.plugins,
    title: {{ display:true, text:'Absolute Error (%)',
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
# 8.  Index page
# =============================================================================

def create_index(out_folder, plot_files, fi_file, results, train_info):
    th    = ''.join(f'<th>{c}</th>' for c in results.columns)
    tbody = ''
    for _, r in results.iterrows():
        cells = ''.join(f'<td>{v}</td>' for v in r)
        tbody += f'<tr>{cells}</tr>\n'

    links    = '\n'.join(
        f'<li><a href="{f}" target="_blank">{f}</a></li>'
        for f in plot_files)
    feat_lis = ''.join(f'<li><code>{f}</code></li>' for f in FEATURE_COLS)
    fi_block = (f'<div class="fi-link"><a href="{fi_file}" target="_blank">'
                f'Open Feature Importance Chart &rarr;</a></div>'
                if fi_file else '<p>Feature importance not available.</p>')

    # pull combined row for headline stats
    comb = results[results['Period'] == 'Combined']
    mae_h  = comb['MAE (INR)'].values[0]     if len(comb) else 'N/A'
    rmse_h = comb['RMSE (INR)'].values[0]    if len(comb) else 'N/A'
    avg_h  = comb['Avg Error (%)'].values[0] if len(comb) else 'N/A'
    p65_h  = comb['65th Pct Err (%)'].values[0] if len(comb) else 'N/A'
    p90_h  = comb['90th Pct Err (%)'].values[0] if len(comb) else 'N/A'
    nd_h   = int(comb['Days'].values[0])     if len(comb) else 'N/A'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GB USD/INR Validation 2020-2021</title>
<style>
  * {{ box-sizing:border-box; }}
  body {{ background:#0d1117; color:#c9d1d9;
          font-family:'Courier New',monospace;
          max-width:1300px; margin:40px auto; padding:0 24px; }}
  h1   {{ text-align:center; color:#58a6ff; font-size:1.85em; margin-bottom:4px; }}
  h2   {{ color:#58a6ff; border-bottom:1px solid #30363d;
          padding-bottom:5px; margin:26px 0 12px; }}
  p    {{ text-align:center; color:#8b949e; margin:5px 0; font-size:0.87em; }}
  .pill {{ display:inline-block; background:#1f3a5f; color:#79c0ff;
           border:1px solid #2d5986; border-radius:12px;
           font-size:0.78em; padding:3px 12px; margin:0 4px; }}
  .stats {{ display:flex; gap:18px; justify-content:center;
            margin:20px 0; flex-wrap:wrap; }}
  .sb  {{ background:#161b22; border:1px solid #30363d;
          border-radius:8px; padding:14px 22px; text-align:center; }}
  .sv  {{ font-size:1.75em; color:#ff8c00; font-weight:bold; }}
  .sl  {{ font-size:0.75em; color:#8b949e; margin-top:4px; }}
  .train-info {{ background:#161b22; border:1px solid #30363d;
                 border-radius:8px; padding:14px 20px;
                 margin:16px 0; font-size:0.85em; color:#8b949e; }}
  .train-info span {{ color:#c9d1d9; }}
  table {{ border-collapse:collapse; width:100%;
           font-size:0.81em; margin:10px 0; }}
  th   {{ background:#161b22; color:#58a6ff; padding:9px 13px;
          text-align:right; border:1px solid #30363d; white-space:nowrap; }}
  td   {{ padding:7px 13px; text-align:right; border:1px solid #21262d; }}
  tr:nth-child(even) {{ background:#161b22; }}
  tr:hover {{ background:#1f2937; }}
  ul.plots {{ list-style:none; padding:0; display:flex;
              gap:16px; flex-wrap:wrap; margin:10px 0; }}
  ul.feats {{ list-style:none; padding:0; display:flex;
              flex-wrap:wrap; gap:8px; margin:8px 0; }}
  li   {{ margin:4px 0; }}
  a    {{ color:#58a6ff; text-decoration:none; }}
  a:hover {{ color:#ff8c00; }}
  code {{ background:#161b22; padding:3px 8px; border-radius:4px;
          color:#79c0ff; font-size:0.88em; }}
  .fi-link {{ text-align:center; margin:10px 0; font-size:0.9em; }}
</style>
</head>
<body>

<h1>USD/INR &mdash; Gradient Boosting Validation</h1>
<p>
  <span class="pill">Train: 2003&ndash;2019</span>
  <span class="pill">Validate: 2020&ndash;2021 (out-of-sample)</span>
  <span class="pill">HistGradientBoostingRegressor</span>
</p>

<div class="train-info">
  Model trained on <span>{train_info['n_train']:,} rows</span>
  &nbsp;|&nbsp;
  GARCH + thresholds fit on <span>2003&ndash;2019</span>
  &nbsp;|&nbsp;
  gamma = <span>{train_info['gamma']:.4f}</span>
  &nbsp;|&nbsp;
  beta = <span>{train_info['beta']:.4f}</span>
  &nbsp;|&nbsp;
  Features: <span>{len(FEATURE_COLS)}</span>
</div>

<div class="stats">
  <div class="sb"><div class="sv">{mae_h}</div><div class="sl">Combined MAE (INR)</div></div>
  <div class="sb"><div class="sv">{rmse_h}</div><div class="sl">Combined RMSE (INR)</div></div>
  <div class="sb"><div class="sv">{avg_h}%</div><div class="sl">Avg Abs Error %</div></div>
  <div class="sb"><div class="sv">{p65_h}%</div><div class="sl">65th Pct Error %</div></div>
  <div class="sb"><div class="sv">{p90_h}%</div><div class="sl">90th Pct Error %</div></div>
  <div class="sb"><div class="sv">{nd_h}</div><div class="sl">Total Days Validated</div></div>
</div>

<h2>Features ({len(FEATURE_COLS)} total)</h2>
<ul class="feats">{feat_lis}</ul>

<h2>Feature Importance</h2>
{fi_block}

<h2>Validation Results</h2>
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>{tbody}</tbody>
</table>

<h2>Per-Year Validation Plots</h2>
<p>Actual vs Predicted &middot; Error bar &middot;
   Abs Error % &mdash; prediction dots coloured by vol regime</p>
<ul class="plots">{links}</ul>

</body>
</html>"""

    path = os.path.join(out_folder, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


# =============================================================================
# 9.  Main
# =============================================================================

def main(input_csv='USD_INR_Exchange.csv', out_folder='gb_pred_plots'):
    print(f'\n{"=" * 62}')
    print('  USD/INR GB Predictor  |  Train 2003-2019  |  Val 2020-2021')
    print(f'{"=" * 62}')

    # -- Load full dataset (2003-2021) ----------------------------------------
    print(f'\nLoading {input_csv} ...')
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
    df = df[df.index.year <= 2021]

    years_found = sorted(df.index.year.unique())
    print(f'  {len(df):,} days  |  years: {years_found[0]} -> {years_found[-1]}')

    val_years = [y for y in [2020, 2021] if y in years_found]
    if not val_years:
        print("\n  WARNING: No 2020 or 2021 data found in CSV.")
        print("  Check your file has rows beyond 2019.")
        sys.exit(1)

    # -- GARCH on full series (seeds rolling correctly for 2020-2021) ---------
    print('\n[1] Fitting GARCH(1,1) on full series ...')
    df['garch_vol'] = fit_garch11(df['Close'].pct_change())

    # -- Build all features on full series ------------------------------------
    print('\n[2] Building features on full series ...')
    df = build_features(df)

    # -- Learn gamma/beta on 2003-2019 ONLY -----------------------------------
    print('\n[3] Learning direction thresholds on 2003-2019 ...')
    train_slice = df[df.index.year <= 2019].copy()
    gamma, beta = learn_thresholds(train_slice)

    # Vol regime string for plot colouring
    rmap_inv = {2: 'high', 1: 'medium', 0: 'low'}
    df['vol_regime'] = df['regime_enc'].map(rmap_inv)
    rc = df[df.index.year <= 2019]['vol_regime'].value_counts()
    print(f"  Regime breakdown (train):")
    for r in ['high', 'medium', 'low']:
        n = rc.get(r, 0)
        tot = len(df[df.index.year <= 2019])
        print(f"    {r:8s}: {n:5d} days ({n/tot*100:.1f}%)")

    # -- Train on 2003-2019, predict 2020-2021 --------------------------------
    print('\n[4] Training GB model & predicting 2020-2021 ...')
    df, model = train_and_predict(df)

    # -- Results --------------------------------------------------------------
    print('\n[5] Computing results ...')
    results = compute_results(df)
    print_results(results)

    # -- Feature importance ---------------------------------------------------
    print('\n[6] Computing permutation feature importance ...')
    os.makedirs(out_folder, exist_ok=True)
    fi_file = compute_feature_importance(model, df, out_folder)

    # -- Plots ----------------------------------------------------------------
    print('\n[7] Generating validation plots ...')
    plot_files = []
    for i, year in enumerate(val_years, 1):
        ydf = df[df.index.year == year].copy()
        fname = plot_year_html(ydf, year, out_folder, i)
        if fname:
            plot_files.append(fname)
            print(f'  {year} -> {fname}')

    # count actual training rows used
    train_used = df[df.index.year <= 2019].iloc[:-1].dropna(
        subset=FEATURE_COLS + [TARGET_COL])

    train_info = {
        'n_train': len(train_used),
        'gamma':   gamma,
        'beta':    beta,
    }

    # -- Index ----------------------------------------------------------------
    idx = create_index(out_folder, plot_files, fi_file, results, train_info)
    print(f'\nDone -- outputs in "{out_folder}/"')
    print(f'Open : {idx}\n')


if __name__ == '__main__':
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    out_folder = sys.argv[2] if len(sys.argv) > 2 else 'gb_pred_plots'
    main(csv_path, out_folder)