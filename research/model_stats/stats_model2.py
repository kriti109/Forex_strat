"""
USD/INR Next-Day Close Price Predictor  —  Walk-Forward Edition (Fixed)
=======================================================================
Changes from original:
  1. GARCH parameters are now fitted on 2003-2020 ONLY (not the full 2003-2023
     series). This eliminates the forward bias in GARCH estimation.
  2. For the 2021-2023 test period, GARCH parameters are updated every 7
     trading days using a walk-forward expanding window (all data strictly
     before the current prediction day). Identical approach to the Decision
     Tree edition.
  3. All other logic (EMA features, gamma/beta walk-forward, vol regime,
     prediction formula) is unchanged and was already forward-bias-free.

Remaining known issues (not forward bias, but worth noting):
  - gamma/beta are tuned via differential_evolution on the full 2003-2020
    training set. The optimiser sees the entire train set at once, which is
    standard and acceptable for an initial fit.
  - Vol regime rolling std (std5/std14/std25) is computed on the combined
    series. Since these are purely backward-looking rolling windows, this is
    causal and correct.

Logic:
  - Fit GARCH(1,1) on 2003-2020 -> initial parameters (omega, alpha, beta)
  - Apply those parameters to get conditional vol for 2003-2020 train set
  - Walk-forward GARCH for 2021-2023:
      Every 7 trading days, refit GARCH on all data seen so far
      (2003-2020 + elapsed test days, strictly before today).
      Apply current parameters via full recursion from t=0 to get today's vol.
  - Direction from optimised EMA5-EMA20 threshold model (gamma/beta)
  - Regime classification (using rolling std):
      HIGH vol  : std5 > std14 > std25
      LOW  vol  : std5 < std14 < std25
      MEDIUM    : everything else
  - Prediction for day t+1:
      HIGH   -> close_t + direction * garch_vol_t * close_t
      LOW    -> close_t  (same price, no change)
      MEDIUM -> close_t + direction * 0.5 * garch_vol_t * close_t
  - Walk-forward gamma/beta update:
      Every 7 trading days, re-run optimiser on all data strictly before today.

Outputs:
  - usd_inr_pred_plots/index.html             (train 2003-2020 summary + links)
  - usd_inr_pred_plots/plot_NN_YYYY.html      (one per year, actual vs predicted)
  - usd_inr_pred_plots/validation_index.html  (2021-2023 walk-forward summary + links)

Usage:
  python predict_usd_inr_fixed.py [path_to_csv] [output_folder]
  Default csv : USD_INR_Exchange.csv
  Default out : usd_inr_pred_plots/
"""

import pandas as pd
import numpy as np
import os
import json
import sys
from scipy.optimize import differential_evolution


# ─────────────────────────────────────────────────────────────────────────────
# 1. GARCH(1,1) — separated into fit (returns params) and apply (returns vol)
# ─────────────────────────────────────────────────────────────────────────────

def _garch_variance(r, omega, alpha, beta):
    """Run GARCH(1,1) variance recursion. Pure math — no lookahead possible."""
    n = len(r)
    h = np.full(n, max(float(np.var(r)), 1e-10))
    for t in range(1, n):
        h[t] = max(omega + alpha * r[t-1]**2 + beta * h[t-1], 1e-12)
    return h


def fit_garch11(returns, label=''):
    """
    Fit GARCH(1,1) parameters on `returns` using grid-search MLE.
    Returns (omega, alpha, beta) — the raw parameters only.
    Does NOT return a vol series; call apply_garch_vol() separately.

    Separating fit from apply is the key fix: the caller decides which
    data to fit on (train-only) and which data to apply to (any window).
    """
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
    tag = f"  [{label}] " if label else "  "
    print(f"{tag}GARCH(1,1): omega={omega:.2e}  alpha={alpha:.4f}  beta={beta:.4f}"
          f"  persistence={alpha+beta:.4f}")
    return omega, alpha, beta


def apply_garch_vol(returns, omega, alpha, beta):
    """
    Given fixed GARCH parameters, run the variance recursion on `returns`
    and return the conditional std-dev series (same length as returns).
    """
    r = returns.fillna(0).values
    h = _garch_variance(r, omega, alpha, beta)
    return np.sqrt(np.maximum(h, 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Walk-forward GARCH vol for the test period (2021-2023)
# ─────────────────────────────────────────────────────────────────────────────

def build_walkforward_garch_vol(full_raw, df_test,
                                 init_omega, init_alpha, init_beta,
                                 update_every=7):
    """
    Produce a conditional vol Series for every day in df_test with zero
    lookahead.

    For each test day i:
      - If i > 0 and i % update_every == 0:
          Refit GARCH parameters on ALL returns strictly before today
          (expanding window = 2003-2020 train + test days elapsed so far).
          Uses full_raw.index < day, so today's return is NEVER seen.
      - Run the full GARCH recursion from t=0 on all history up to and
          including today (full_raw.index <= day) with current parameters,
          and read off the last value as today's conditional vol.
          Reading today's vol to predict tomorrow is causal — vol clustering
          means today's variance is a valid predictor of tomorrow's.

    Parameters
    ----------
    full_raw : DataFrame
        The complete price series (2003-2023), used only for building the
        expanding return history.
    df_test : DataFrame
        Test rows (2021-2023). Index must be a subset of full_raw.index.
    init_omega, init_alpha, init_beta : float
        GARCH parameters pre-fitted on 2003-2020 train data.
    update_every : int
        How many test days to wait between parameter refits.

    Returns
    -------
    pd.Series aligned to df_test.index
    """
    val_idx = df_test.index
    n_val   = len(val_idx)

    omega, alpha, beta = init_omega, init_alpha, init_beta
    garch_vol_list = []

    print(f"\n  Walk-forward GARCH: {n_val} test days  |  refit every {update_every} days")
    print(f"  Initial params: omega={omega:.2e}  alpha={alpha:.4f}  beta={beta:.4f}")

    for i in range(n_val):
        day = val_idx[i]

        # --- Refit: use ALL history strictly before today -------------------
        if i > 0 and i % update_every == 0:
            history_rets = full_raw.loc[full_raw.index < day, 'Close'].pct_change()
            if len(history_rets.dropna()) > 50:
                omega, alpha, beta = fit_garch11(
                    history_rets,
                    label=f"refit @ {day.date()}  n={len(history_rets)}"
                )

        # --- Apply: run recursion on all history up to and including today --
        history_rets = full_raw.loc[full_raw.index <= day, 'Close'].pct_change()
        vol_series   = apply_garch_vol(history_rets, omega, alpha, beta)
        garch_vol_list.append(vol_series[-1])   # today's conditional vol

    return pd.Series(garch_vol_list, index=val_idx, name='garch_vol')


# ─────────────────────────────────────────────────────────────────────────────
# 3. Volatility regime (causal rolling std — no lookahead)
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 4. Direction model  (EMA5 - EMA20 threshold, optimised via diff-evolution)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ema_feature(df):
    c = df['Close']
    df['ema_5']       = c.ewm(span=5,  adjust=False).mean()
    df['ema_20']      = c.ewm(span=20, adjust=False).mean()
    df['feat']        = df['ema_5'] - df['ema_20']
    df['close_delta'] = c.diff().shift(-1)   # next-day change (NaN on last row)
    df['target']      = np.where(df['close_delta'] > 0, 1, -1)
    return df


def _objective(params, feat_vals, target_vals, max_abstain=0.05):
    gamma, beta = params
    if beta <= gamma:
        return 0.0
    mask_up   = feat_vals >  beta
    mask_down = feat_vals <  gamma
    n_total   = len(feat_vals)
    n_abstain = ((feat_vals >= gamma) & (feat_vals <= beta)).sum()
    penalty   = max(0.0, (n_abstain / n_total - max_abstain) * 200)
    n_counted = mask_up.sum() + mask_down.sum()
    if n_counted == 0:
        return 0.0
    correct = (
        (mask_up   & (target_vals ==  1)).sum() +
        (mask_down & (target_vals == -1)).sum()
    )
    return -(correct / n_counted - penalty)


def learn_thresholds(df, label=''):
    """Learn gamma/beta from df using only rows where close_delta is known."""
    valid = df.dropna(subset=['feat', 'close_delta'])
    fv    = valid['feat'].values
    tv    = valid['target'].values
    f_min, f_max = fv.min(), fv.max()
    r     = f_max - f_min
    bounds = [(f_min, f_min + r * 0.6), (f_min + r * 0.4, f_max)]
    result = differential_evolution(
        _objective, bounds, args=(fv, tv, 0.05),
        seed=42, maxiter=2000, popsize=20, tol=1e-9,
        mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )
    gamma, beta = result.x
    if label:
        print(f"  [{label}] gamma={gamma:.4f}  beta={beta:.4f}")
    return gamma, beta


# ─────────────────────────────────────────────────────────────────────────────
# 5. Predictions
# ─────────────────────────────────────────────────────────────────────────────

def build_train_predictions(df, beta, gamma):
    """Standard next-day prediction for the train set (fixed thresholds)."""
    close  = df['Close'].values
    gvol   = df['garch_vol'].values
    feat   = df['feat'].values
    regime = df['vol_regime'].values
    n      = len(df)
    pred   = np.full(n, np.nan)

    for i in range(n - 1):
        c, g, f = close[i], gvol[i], feat[i]
        if   f > beta:   d =  1
        elif f < gamma:  d = -1
        else:            d =  0
        if   regime[i] == 'low':    pred[i] = c
        elif regime[i] == 'high':   pred[i] = c + d * g * c
        else:                       pred[i] = c + d * 0.5 * g * c

    df = df.copy()
    df['predicted_next'] = pred
    df['next_actual']    = df['Close'].shift(-1)
    df['gamma_used']     = gamma
    df['beta_used']      = beta
    return df


def build_walkforward_predictions(df_full, df_val, initial_gamma, initial_beta,
                                  update_every=7):
    """
    Walk-forward prediction for df_val (2021-2023).

    On day i (val index):
      - If i > 0 and i % update_every == 0:
          Re-run gamma/beta optimiser on ALL of df_full strictly before today
          (expanding window — 2003-2020 + val days seen so far).
          close_delta at row k needs close at k+1, so the last row of history
          naturally has NaN close_delta and is excluded by learn_thresholds.
      - Predict close for day i+1 using current gamma/beta and today's features.
    Zero lookahead guaranteed.
    """
    val_idx = df_val.index
    n_val   = len(val_idx)
    gamma, beta = initial_gamma, initial_beta

    pred_list  = []
    gamma_list = []
    beta_list  = []

    print(f"\n  Walk-forward gamma/beta: {n_val} days  |  update every {update_every} days")
    print(f"  Initial gamma={gamma:.4f}  beta={beta:.4f}")

    for i in range(n_val - 1):   # last row: no next_actual
        day = val_idx[i]

        # Update thresholds using all data strictly before today
        if i > 0 and i % update_every == 0:
            history = df_full.loc[df_full.index < day]
            if len(history) > 50:
                gamma, beta = learn_thresholds(
                    history,
                    label=f"gamma/beta update @ {day.date()}  (n={len(history)} days)"
                )

        # Predict next day
        row    = df_val.iloc[i]
        c      = row['Close']
        g      = row['garch_vol']
        f      = row['feat']
        regime = row['vol_regime']

        if   f > beta:   d =  1
        elif f < gamma:  d = -1
        else:            d =  0

        if   regime == 'low':    p = c
        elif regime == 'high':   p = c + d * g * c
        else:                    p = c + d * 0.5 * g * c

        pred_list.append(p)
        gamma_list.append(gamma)
        beta_list.append(beta)

    # Last row — no prediction
    pred_list.append(np.nan)
    gamma_list.append(gamma)
    beta_list.append(beta)

    df_out = df_val.copy()
    df_out['predicted_next'] = pred_list
    df_out['next_actual']    = df_val['Close'].shift(-1)
    df_out['gamma_used']     = gamma_list
    df_out['beta_used']      = beta_list
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(df):
    rows = []
    for year, g in df.groupby(df.index.year):
        v = g.dropna(subset=['predicted_next', 'next_actual'])
        if len(v) == 0:
            continue
        err     = (v['predicted_next'] - v['next_actual']).abs()
        err_pct = err / v['next_actual'] * 100
        rows.append({
            'Year':             year,
            'Days':             len(v),
            'High Vol':         (v['vol_regime'] == 'high').sum(),
            'Low Vol':          (v['vol_regime'] == 'low').sum(),
            'Medium Vol':       (v['vol_regime'] == 'medium').sum(),
            'Avg Error (INR)':  round(float(err.mean()),     4),
            'Avg Error (%)':    round(float(err_pct.mean()), 4),
            '65th Pct Err (%)': round(float(err_pct.quantile(0.65)), 4),
            'Max Error (%)':    round(float(err_pct.max()),  4),
        })
    return pd.DataFrame(rows)


def print_summary(summary, df, label=''):
    print("\n" + "=" * 80)
    if label:
        print(f"  {label}")
    print(f"{'Year':<8} {'Days':>6} {'HiVol':>6} {'LoVol':>6} {'MedVol':>7} "
          f"{'AvgErrINR':>11} {'AvgErr%':>9} {'65pct%':>8} {'MaxErr%':>9}")
    print("-" * 80)
    for _, r in summary.iterrows():
        print(f"{int(r['Year']):<8} {int(r['Days']):>6} {int(r['High Vol']):>6} "
              f"{int(r['Low Vol']):>6} {int(r['Medium Vol']):>7} "
              f"{r['Avg Error (INR)']:>11.4f} {r['Avg Error (%)']:>9.4f} "
              f"{r['65th Pct Err (%)']:>8.4f} {r['Max Error (%)']:>9.4f}")
    print("=" * 80)
    v  = df.dropna(subset=['predicted_next', 'next_actual'])
    ep = (v['predicted_next'] - v['next_actual']).abs() / v['next_actual'] * 100
    print(f"\nOVERALL ({len(v):,} days)")
    print(f"  Avg abs error   : {ep.mean():.4f}%")
    print(f"  Median error    : {ep.median():.4f}%")
    print(f"  65th pct error  : {ep.quantile(0.65):.4f}%")
    print(f"  90th pct error  : {ep.quantile(0.90):.4f}%")
    print(f"  Max error       : {ep.max():.4f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 7. HTML plots
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 4)


def plot_year_html(year_df, year, output_folder, plot_num,
                   index_file='index.html', show_thresholds=False):
    v = year_df.dropna(subset=['predicted_next', 'next_actual'])

    labels      = [str(d.date()) for d in v.index]
    actual      = [_safe(x) for x in v['next_actual']]
    predicted   = [_safe(x) for x in v['predicted_next']]
    error_inr   = [_safe(x) for x in (v['predicted_next'] - v['next_actual'])]
    error_pct   = [_safe(x) for x in
                   ((v['predicted_next'] - v['next_actual']) / v['next_actual'] * 100)]
    abs_err_pct = [abs(x) for x in error_pct if x is not None]

    regime_colors = {
        'high':   'rgba(255,80,80,0.35)',
        'low':    'rgba(80,200,100,0.35)',
        'medium': 'rgba(100,140,255,0.25)',
    }
    point_colors = [regime_colors.get(r, 'grey') for r in v['vol_regime']]

    avg_err = round(np.mean(abs_err_pct), 3) if abs_err_pct else 0
    p65_err = round(float(np.percentile(abs_err_pct, 65)), 3) if abs_err_pct else 0
    n_high  = (v['vol_regime'] == 'high').sum()
    n_low   = (v['vol_regime'] == 'low').sum()
    n_med   = (v['vol_regime'] == 'medium').sum()

    threshold_chart = ''
    if show_thresholds and 'gamma_used' in v.columns:
        gamma_vals = [_safe(x) for x in v['gamma_used']]
        beta_vals  = [_safe(x) for x in v['beta_used']]
        threshold_chart = f"""
<div class="chart-wrap"><canvas id="c4"></canvas></div>
<script>
(function(){{
  new Chart(document.getElementById('c4'), {{
    type: 'line',
    data: {{
      labels: {json.dumps(labels)},
      datasets: [
        {{ label: 'gamma (lower threshold)', data: {json.dumps(gamma_vals)},
           borderColor: '#ff5050', backgroundColor: 'transparent',
           borderWidth: 1.5, pointRadius: 0, tension: 0.1 }},
        {{ label: 'beta (upper threshold)',  data: {json.dumps(beta_vals)},
           borderColor: '#50c864', backgroundColor: 'transparent',
           borderWidth: 1.5, pointRadius: 0, tension: 0.1 }},
      ]
    }},
    options: {{
      responsive: true, animation: false,
      interaction: {{ mode: 'index', intersect: false }},
      plugins: {{
        legend: {{ labels: {{ color: '#c9d1d9', font: {{ family: 'Courier New' }} }} }},
        title: {{ display: true, text: 'Walk-Forward gamma / beta (updated every 7 days)',
                  color: '#58a6ff', font: {{ size: 14 }} }}
      }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 12, font: {{ size: 10 }} }},
               grid: {{ color: 'rgba(255,255,255,0.06)' }} }},
        y: {{ ticks: {{ color: '#8b949e' }},
               grid: {{ color: 'rgba(255,255,255,0.06)' }} }}
      }}
    }}
  }});
}})();
</script>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>USD/INR {year} Prediction</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace; padding: 20px; }}
  h1   {{ text-align: center; color: #58a6ff; font-size: 1.5em; margin-bottom: 6px; }}
  .stats {{ display: flex; gap: 16px; justify-content: center; margin: 14px 0; flex-wrap: wrap; }}
  .sb  {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 20px; text-align: center; }}
  .sv  {{ font-size: 1.6em; color: #ff8c00; font-weight: bold; }}
  .sl  {{ font-size: 0.75em; color: #8b949e; margin-top: 2px; }}
  .leg {{ display: flex; gap: 20px; justify-content: center; margin: 8px 0; font-size: 0.8em; flex-wrap: wrap; }}
  .li  {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .chart-wrap {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 12px 0; }}
  canvas {{ max-height: 320px; }}
  .back {{ text-align: center; margin-top: 16px; }}
  .back a {{ color: #58a6ff; text-decoration: none; }}
</style>
</head>
<body>
<h1>USD/INR Next-Day Close Prediction — {year}</h1>
<div class="stats">
  <div class="sb"><div class="sv">{avg_err}%</div><div class="sl">Avg Abs Error</div></div>
  <div class="sb"><div class="sv">{p65_err}%</div><div class="sl">65th Pct Error</div></div>
  <div class="sb"><div class="sv">{len(v)}</div><div class="sl">Trading Days</div></div>
  <div class="sb"><div class="sv">{n_high}</div><div class="sl">High Vol Days</div></div>
  <div class="sb"><div class="sv">{n_low}</div><div class="sl">Low Vol Days</div></div>
  <div class="sb"><div class="sv">{n_med}</div><div class="sl">Medium Vol Days</div></div>
</div>
<div class="leg">
  <div class="li"><div class="dot" style="background:#ff5050"></div>High vol: full GARCH shift</div>
  <div class="li"><div class="dot" style="background:#50c864"></div>Low vol: flat (same price)</div>
  <div class="li"><div class="dot" style="background:#648cff"></div>Medium vol: half GARCH shift</div>
</div>
<div class="chart-wrap"><canvas id="c1"></canvas></div>
<div class="chart-wrap"><canvas id="c2"></canvas></div>
<div class="chart-wrap"><canvas id="c3"></canvas></div>
{threshold_chart}
<div class="back"><a href="{index_file}">← Back to Index</a></div>
<script>
const labels    = {json.dumps(labels)};
const actual    = {json.dumps(actual)};
const predicted = {json.dumps(predicted)};
const errorInr  = {json.dumps(error_inr)};
const errorPct  = {json.dumps(error_pct)};
const ptColors  = {json.dumps(point_colors)};
const gridColor = 'rgba(255,255,255,0.06)';
const tickColor = '#8b949e';
const baseOpts  = {{
  responsive: true, animation: false,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ labels: {{ color: '#c9d1d9', font: {{ family: 'Courier New' }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: tickColor, maxTicksLimit: 12, font: {{ size: 10 }} }}, grid: {{ color: gridColor }} }},
    y: {{ ticks: {{ color: tickColor }}, grid: {{ color: gridColor }} }},
  }}
}};
new Chart(document.getElementById('c1'), {{
  type: 'line',
  data: {{ labels, datasets: [
    {{ label: 'Actual Close (INR)', data: actual, borderColor: '#26a69a',
       backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.1 }},
    {{ label: 'Predicted Close (INR)', data: predicted, borderColor: '#ff8c00',
       backgroundColor: 'transparent', borderWidth: 1.5, borderDash: [4,3],
       pointRadius: 3, pointBackgroundColor: ptColors, tension: 0.1 }},
  ]}},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins,
    title: {{ display: true, text: 'Actual vs Predicted Close Price', color: '#58a6ff', font: {{ size: 14 }} }} }} }}
}});
const barColors = errorInr.map(v => v === null ? 'grey' : v >= 0 ? 'rgba(38,166,154,0.75)' : 'rgba(239,83,80,0.75)');
new Chart(document.getElementById('c2'), {{
  type: 'bar',
  data: {{ labels, datasets: [{{ label: 'Prediction Error (INR)', data: errorInr,
    backgroundColor: barColors, borderWidth: 0 }}] }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins,
    title: {{ display: true, text: 'Error: Predicted - Actual (INR)', color: '#58a6ff', font: {{ size: 14 }} }} }} }}
}});
new Chart(document.getElementById('c3'), {{
  type: 'line',
  data: {{ labels, datasets: [{{ label: 'Absolute Error (%)',
    data: errorPct.map(v => v === null ? null : Math.abs(v)),
    borderColor: '#e377c2', backgroundColor: 'rgba(227,119,194,0.12)',
    borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.1 }}] }},
  options: {{ ...baseOpts, plugins: {{ ...baseOpts.plugins,
    title: {{ display: true, text: 'Absolute Error (%)', color: '#58a6ff', font: {{ size: 14 }} }} }} }}
}});
</script>
</body>
</html>"""

    fname = f"plot_{plot_num:02d}_{year}.html"
    with open(os.path.join(output_folder, fname), 'w', encoding='utf-8') as f:
        f.write(html)
    return fname


# ─────────────────────────────────────────────────────────────────────────────
# 8. Index pages
# ─────────────────────────────────────────────────────────────────────────────

def _overall_stats(df):
    valid = df.dropna(subset=['predicted_next', 'next_actual'])
    ep    = (valid['predicted_next'] - valid['next_actual']).abs() / valid['next_actual'] * 100
    return {
        'avg_err': float(ep.mean()),
        'med_err': float(ep.median()),
        'p65_err': float(ep.quantile(0.65)),
        'p90_err': float(ep.quantile(0.90)),
        'n_days':  len(valid),
    }


def _build_index_html(title, subtitle, plot_files, summary, overall_stats,
                      other_link=None, other_label=None):
    th    = ''.join(f'<th>{c}</th>' for c in summary.columns)
    tbody = ''
    for _, r in summary.iterrows():
        cells = ''.join(f'<td>{v}</td>' for v in r)
        tbody += f'<tr>{cells}</tr>\n'
    links = '\n'.join(f'<li><a href="{f}" target="_blank">{f}</a></li>' for f in plot_files)
    nav   = (f'<p style="text-align:center;margin-top:10px">'
             f'<a href="{other_link}">-> {other_label}</a></p>'
             if other_link else '')

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace;
          max-width: 1300px; margin: 40px auto; padding: 0 24px; }}
  h1 {{ text-align: center; color: #58a6ff; font-size: 2em; }}
  h2 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 6px; margin: 28px 0 12px; }}
  p  {{ text-align: center; color: #8b949e; margin: 6px 0; }}
  .stats {{ display: flex; gap: 20px; justify-content: center; margin: 20px 0; flex-wrap: wrap; }}
  .sb {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px 24px; text-align: center; }}
  .sv {{ font-size: 2em; color: #ff8c00; font-weight: bold; }}
  .sl {{ font-size: 0.8em; color: #8b949e; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.82em; margin: 10px 0; }}
  th {{ background: #161b22; color: #58a6ff; padding: 10px 14px; text-align: right;
        border: 1px solid #30363d; white-space: nowrap; }}
  td {{ padding: 8px 14px; text-align: right; border: 1px solid #21262d; }}
  tr:nth-child(even) {{ background: #161b22; }}
  tr:hover {{ background: #1f2937; }}
  ul {{ list-style: none; padding: 0; columns: 3; gap: 10px; }}
  li {{ margin: 8px 0; }}
  a  {{ color: #58a6ff; text-decoration: none; }}
  a:hover {{ color: #ff8c00; }}
  code {{ background: #161b22; padding: 2px 6px; border-radius: 4px; color: #79c0ff; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>{subtitle}</p>
{nav}
<div class="stats">
  <div class="sb"><div class="sv">{overall_stats['avg_err']:.4f}%</div><div class="sl">Overall Avg Abs Error</div></div>
  <div class="sb"><div class="sv">{overall_stats['med_err']:.4f}%</div><div class="sl">Overall Median Error</div></div>
  <div class="sb"><div class="sv">{overall_stats['p65_err']:.4f}%</div><div class="sl">Overall 65th Pct Error</div></div>
  <div class="sb"><div class="sv">{overall_stats['p90_err']:.4f}%</div><div class="sl">Overall 90th Pct Error</div></div>
  <div class="sb"><div class="sv">{overall_stats['n_days']:,}</div><div class="sl">Total Trading Days</div></div>
</div>
<h2>Yearly Summary</h2>
<table><thead><tr>{th}</tr></thead><tbody>{tbody}</tbody></table>
<h2>Per-Year Interactive Plots</h2>
<ul>{links}</ul>
</body>
</html>"""


def create_index(output_folder, plot_files, summary, overall_stats,
                 other_link=None, other_label=None):
    html = _build_index_html(
        title='USD/INR Prediction — Train (2003–2020)',
        subtitle='GARCH(1,1) fitted on 2003-2020 only · EMA direction · rolling-std regime · Fixed gamma/beta',
        plot_files=plot_files, summary=summary, overall_stats=overall_stats,
        other_link=other_link, other_label=other_label,
    )
    path = os.path.join(output_folder, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def create_validation_index(output_folder, plot_files, summary, overall_stats,
                             other_link=None, other_label=None):
    html = _build_index_html(
        title='USD/INR Prediction — Walk-Forward Test (2021–2023)',
        subtitle='Zero lookahead · GARCH params walk-forward updated every 7 days · gamma/beta updated every 7 days',
        plot_files=plot_files, summary=summary, overall_stats=overall_stats,
        other_link=other_link, other_label=other_label,
    )
    path = os.path.join(output_folder, 'validation_index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────────────────────────────────────

def main(input_csv='USD_INR_Exchange.csv', output_folder='usd_inr_pred_plots'):
    print(f'\n{"="*65}')
    print('  USD/INR Next-Day Close Predictor  —  Walk-Forward (Fixed)')
    print(f'{"="*65}')

    # ── Load data ─────────────────────────────────────────────────────────────
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

    # ── GARCH: fit on train ONLY (the key fix) ────────────────────────────────
    # FIX: the original code called fit_garch11() on the full 2003-2023 series,
    # which meant the initial GARCH parameters were informed by 2021-2023 data.
    # We now fit only on 2003-2020 and use a walk-forward refit for the test set.
    print('\n[1] Fitting GARCH(1,1) on 2003-2020 only (train set) ...')
    train_returns = df_train['Close'].pct_change()
    garch_omega, garch_alpha, garch_beta = fit_garch11(
        train_returns, label='initial 2003-2020'
    )

    # Apply fixed params to the train series
    train_vol = apply_garch_vol(train_returns, garch_omega, garch_alpha, garch_beta)
    df_train['garch_vol'] = train_vol

    # ── GARCH walk-forward for the test period ────────────────────────────────
    print('\n[2] Walk-forward GARCH vol for 2021-2023 ...')
    # We pass the full raw series so the GARCH recursion can condition on all
    # available history (growing expanding window). No test data is ever used
    # for parameter fitting before the prediction day.
    full_raw = raw[(raw.index.year >= 2003) & (raw.index.year <= 2023)].copy()
    test_vol = build_walkforward_garch_vol(
        full_raw=full_raw,
        df_test=df_test,
        init_omega=garch_omega,
        init_alpha=garch_alpha,
        init_beta=garch_beta,
        update_every=7,
    )
    df_test['garch_vol'] = test_vol

    # ── Regime + features: compute on full series (all causal) ───────────────
    print('\n[3] Computing volatility regimes ...')
    full = raw[(raw.index.year >= 2003) & (raw.index.year <= 2023)].copy()
    full = classify_vol_regime(full)

    print('\n[4] Computing EMA direction features ...')
    full = compute_ema_feature(full)

    feat_cols = ['std5', 'std14', 'std25', 'vol_regime',
                 'ema_5', 'ema_20', 'feat', 'close_delta', 'target']
    for col in feat_cols:
        df_train[col] = full.loc[df_train.index, col]
        df_test[col]  = full.loc[df_test.index,  col]

    # Build a combined train+test df for use as the walk-forward history base.
    # close_delta and target are only used in learn_thresholds() which always
    # calls dropna() on close_delta, so the last row's NaN is handled safely.
    full_feat = full.loc[
        (full.index.year >= 2003) & (full.index.year <= 2023),
        feat_cols + ['Close']
    ].copy()
    full_feat['garch_vol'] = pd.concat([df_train['garch_vol'], df_test['garch_vol']])

    # ── Learn initial gamma/beta on train only ────────────────────────────────
    print('\n[5] Learning initial gamma/beta on 2003-2020 ...')
    gamma0, beta0 = learn_thresholds(df_train, label='initial (2003-2020)')
    print(f'  Frozen for train predictions: gamma={gamma0:.4f}  beta={beta0:.4f}')

    # ── Train predictions ─────────────────────────────────────────────────────
    print('\n[6] Building train predictions (2003-2020, fixed thresholds) ...')
    df_train = build_train_predictions(df_train, beta0, gamma0)

    # ── Walk-forward predictions 2021-2023 ────────────────────────────────────
    print('\n[7] Walk-forward predictions (2021-2023, update every 7 days) ...')
    df_test = build_walkforward_predictions(
        df_full=full_feat,
        df_val=df_test,
        initial_gamma=gamma0,
        initial_beta=beta0,
        update_every=7,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    print('\n[8] Results — TRAIN (2003-2020)')
    summary_train = compute_summary(df_train)
    print_summary(summary_train, df_train, label='TRAIN SET')

    print('\n[8b] Results — TEST / WALK-FORWARD (2021-2023)')
    summary_test = compute_summary(df_test)
    print_summary(summary_test, df_test, label='TEST SET (zero lookahead)')

    # ── HTML plots ────────────────────────────────────────────────────────────
    print('\n[9] Generating HTML plots ...')
    os.makedirs(output_folder, exist_ok=True)

    train_plot_files = []
    for i, year in enumerate(range(2003, 2021), 1):
        ydf = df_train[df_train.index.year == year].copy()
        if ydf.empty:
            continue
        print(f'  Train {year} ({len(ydf)} days) ...')
        fname = plot_year_html(ydf, year, output_folder, i,
                               index_file='index.html', show_thresholds=False)
        train_plot_files.append(fname)

    test_plot_files = []
    for i, year in enumerate(range(2021, 2024), 1):
        ydf = df_test[df_test.index.year == year].copy()
        if ydf.empty:
            continue
        print(f'  Test {year} ({len(ydf)} days) ...')
        fname = plot_year_html(ydf, year, output_folder, i + 100,
                               index_file='validation_index.html',
                               show_thresholds=True)
        test_plot_files.append(fname)

    train_stats = _overall_stats(df_train)
    test_stats  = _overall_stats(df_test)

    create_index(
        output_folder, train_plot_files, summary_train, train_stats,
        other_link='validation_index.html',
        other_label='Go to Walk-Forward Test Results (2021–2023)',
    )
    create_validation_index(
        output_folder, test_plot_files, summary_test, test_stats,
        other_link='index.html',
        other_label='Go to Train Results (2003–2020)',
    )

    print(f'\nDone.')
    print(f'  Train index : {output_folder}/index.html')
    print(f'  Test index  : {output_folder}/validation_index.html')
    print(f'{"="*65}\n')


if __name__ == '__main__':
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    out_folder = sys.argv[2] if len(sys.argv) > 2 else 'usd_inr_pred_plots'
    main(csv_path, out_folder)