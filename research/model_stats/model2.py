"""
USD/INR Next-Day Close Price Predictor
=======================================
Logic:
  - Fit GARCH(1,1) -> conditional volatility (sigma_t, in return units)
  - Rolling std of last 5, 14, 25 days (price std, for regime detection)
  - Direction from optimised EMA5-EMA20 threshold model (beta/gamma)

  Regime classification (using rolling std):
    HIGH vol  : std5 > std14 > std25
    LOW  vol  : std5 < std14 < std25
    MEDIUM    : everything else

  Prediction for day t+1:
    HIGH   -> close_t + direction * garch_vol_t * close_t
    LOW    -> close_t  (same price, no change)
    MEDIUM -> close_t + direction * 0.5 * garch_vol_t * close_t

  Abstain (direction=None): treat as direction=0 (flat) for high/medium

  TRAIN  : 2003 – 2021  (gamma/beta learned here, frozen after)
  VALIDATE: 2022 – 2023 (same frozen gamma/beta applied)

Outputs:
  - usd_inr_pred_plots/index.html        (summary table + links, train years)
  - usd_inr_pred_plots/plot_NN_YYYY.html (one per year, actual vs predicted)
  - usd_inr_pred_plots/validation_index.html  (2022-2023 summary + links)

Usage:
  python predict_usd_inr.py [path_to_csv] [output_folder]
  Default csv : USD_INR_Exchange.csv
  Default out : usd_inr_pred_plots/
"""

import pandas as pd
import numpy as np
import os
import json
import sys
from scipy.optimize import differential_evolution


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
    """Return conditional std-dev series (in return fraction units)."""
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
# 3. Direction model  (EMA5 – EMA20 threshold, optimised via diff-evolution)
# ──────────────────────────────────────────────────────────────────────────────

def compute_ema_feature(df):
    c = df['Close']
    df['ema_5']       = c.ewm(span=5,  adjust=False).mean()
    df['ema_20']      = c.ewm(span=20, adjust=False).mean()
    df['feat']        = df['ema_5'] - df['ema_20']
    df['close_delta'] = c.diff().shift(-1)           # next-day change
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


def learn_thresholds(df):
    """Learn gamma/beta from the provided dataframe (training data only)."""
    valid     = df.dropna(subset=['feat', 'close_delta'])
    fv        = valid['feat'].values
    tv        = valid['target'].values
    f_min, f_max = fv.min(), fv.max()
    r         = f_max - f_min
    bounds    = [(f_min, f_min + r * 0.6), (f_min + r * 0.4, f_max)]
    result    = differential_evolution(
        _objective, bounds, args=(fv, tv, 0.05),
        seed=42, maxiter=2000, popsize=20, tol=1e-9,
        mutation=(0.5, 1.5), recombination=0.9, polish=True,
    )
    gamma, beta = result.x
    print(f"  Direction thresholds: gamma={gamma:.4f}  beta={beta:.4f}")
    return gamma, beta


# ──────────────────────────────────────────────────────────────────────────────
# 4. Build next-day prediction
# ──────────────────────────────────────────────────────────────────────────────

def build_predictions(df, beta, gamma):
    """
    For each row i, predict close[i+1].
    Uses the frozen beta/gamma (learned on train set only).
    """
    close  = df['Close'].values
    gvol   = df['garch_vol'].values
    feat   = df['feat'].values
    regime = df['vol_regime'].values
    n      = len(df)

    pred = np.full(n, np.nan)

    for i in range(n - 1):
        c = close[i]
        g = gvol[i]
        f = feat[i]

        # direction — using frozen thresholds
        if   f > beta:   d =  1
        elif f < gamma:  d = -1
        else:            d =  0       # abstain → flat

        if   regime[i] == 'low':    pred[i] = c
        elif regime[i] == 'high':   pred[i] = c + d * g * c
        else:                       pred[i] = c + d * 0.5 * g * c   # medium

    df = df.copy()
    df['predicted_next'] = pred
    df['next_actual']    = df['Close'].shift(-1)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 5. Summary statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_summary(df):
    rows = []
    for year, g in df.groupby(df.index.year):
        v = g.dropna(subset=['predicted_next', 'next_actual'])
        if len(v) == 0:
            continue
        err     = (v['predicted_next'] - v['next_actual']).abs()
        err_pct = err / v['next_actual'] * 100
        rows.append({
            'Year':               year,
            'Days':               len(v),
            'High Vol':           (v['vol_regime'] == 'high').sum(),
            'Low Vol':            (v['vol_regime'] == 'low').sum(),
            'Medium Vol':         (v['vol_regime'] == 'medium').sum(),
            'Avg Error (INR)':    round(float(err.mean()),     4),
            'Avg Error (%)':      round(float(err_pct.mean()), 4),
            '65th Pct Err (%)':   round(float(err_pct.quantile(0.65)), 4),
            'Max Error (%)':      round(float(err_pct.max()),  4),
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

    v       = df.dropna(subset=['predicted_next', 'next_actual'])
    ep      = (v['predicted_next'] - v['next_actual']).abs() / v['next_actual'] * 100
    print(f"\nOVERALL ({len(v):,} days)")
    print(f"  Avg abs error   : {ep.mean():.4f}%")
    print(f"  Median error    : {ep.median():.4f}%")
    print(f"  65th pct error  : {ep.quantile(0.65):.4f}%")
    print(f"  90th pct error  : {ep.quantile(0.90):.4f}%")
    print(f"  Max error       : {ep.max():.4f}%")


# ──────────────────────────────────────────────────────────────────────────────
# 6. HTML plot (one per year)
# ──────────────────────────────────────────────────────────────────────────────

def _safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), 4)


def plot_year_html(year_df, year, output_folder, plot_num, index_file='index.html'):
    v = year_df.dropna(subset=['predicted_next', 'next_actual'])

    labels       = [str(d.date()) for d in v.index]
    actual       = [_safe(x) for x in v['next_actual']]
    predicted    = [_safe(x) for x in v['predicted_next']]
    error_inr    = [_safe(x) for x in (v['predicted_next'] - v['next_actual'])]
    error_pct    = [_safe(x) for x in
                    ((v['predicted_next'] - v['next_actual']) / v['next_actual'] * 100)]
    abs_err_pct  = [abs(x) for x in error_pct if x is not None]

    regime_colors = {
        'high':   'rgba(255,80,80,0.35)',
        'low':    'rgba(80,200,100,0.35)',
        'medium': 'rgba(100,140,255,0.25)',
    }
    point_colors = [regime_colors.get(r, 'grey') for r in v['vol_regime']]

    avg_err  = round(np.mean(abs_err_pct), 3) if abs_err_pct else 0
    p65_err  = round(float(np.percentile(abs_err_pct, 65)), 3) if abs_err_pct else 0
    n_high   = (v['vol_regime'] == 'high').sum()
    n_low    = (v['vol_regime'] == 'low').sum()
    n_med    = (v['vol_regime'] == 'medium').sum()

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>USD/INR {year} Prediction</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Courier New', monospace;
          padding: 20px; }}
  h1   {{ text-align: center; color: #58a6ff; font-size: 1.5em; margin-bottom: 6px; }}
  .stats {{ display: flex; gap: 16px; justify-content: center; margin: 14px 0; flex-wrap: wrap; }}
  .sb  {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px;
          padding: 10px 20px; text-align: center; }}
  .sv  {{ font-size: 1.6em; color: #ff8c00; font-weight: bold; }}
  .sl  {{ font-size: 0.75em; color: #8b949e; margin-top: 2px; }}
  .leg {{ display: flex; gap: 20px; justify-content: center; margin: 8px 0;
          font-size: 0.8em; flex-wrap: wrap; }}
  .li  {{ display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .chart-wrap {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 16px; margin: 12px 0; }}
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

<div class="chart-wrap">
  <canvas id="c1"></canvas>
</div>
<div class="chart-wrap">
  <canvas id="c2"></canvas>
</div>
<div class="chart-wrap">
  <canvas id="c3"></canvas>
</div>

<div class="back"><a href="{index_file}">← Back to Index</a></div>

<script>
const labels    = {json.dumps(labels)};
const actual    = {json.dumps(actual)};
const predicted = {json.dumps(predicted)};
const errorInr  = {json.dumps(error_inr)};
const errorPct  = {json.dumps(error_pct)};
const ptColors  = {json.dumps(point_colors)};

const gridColor  = 'rgba(255,255,255,0.06)';
const tickColor  = '#8b949e';
const baseOpts   = {{
  responsive: true,
  animation: false,
  interaction: {{ mode: 'index', intersect: false }},
  plugins: {{ legend: {{ labels: {{ color: '#c9d1d9', font: {{ family: 'Courier New' }} }} }} }},
  scales: {{
    x: {{ ticks: {{ color: tickColor, maxTicksLimit: 12, font: {{ size: 10 }} }},
           grid: {{ color: gridColor }} }},
    y: {{ ticks: {{ color: tickColor }}, grid: {{ color: gridColor }} }},
  }}
}};

new Chart(document.getElementById('c1'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{ label: 'Actual Close (INR)', data: actual,
         borderColor: '#26a69a', backgroundColor: 'transparent',
         borderWidth: 2, pointRadius: 0, tension: 0.1 }},
      {{ label: 'Predicted Close (INR)', data: predicted,
         borderColor: '#ff8c00', backgroundColor: 'transparent',
         borderWidth: 1.5, borderDash: [4, 3], pointRadius: 3,
         pointBackgroundColor: ptColors, tension: 0.1 }},
    ]
  }},
  options: {{ ...baseOpts,
    plugins: {{ ...baseOpts.plugins,
      title: {{ display: true, text: 'Actual vs Predicted Close Price',
                color: '#58a6ff', font: {{ size: 14 }} }} }} }}
}});

const barColors = errorInr.map(v => v === null ? 'grey' : v >= 0 ? 'rgba(38,166,154,0.75)' : 'rgba(239,83,80,0.75)');
new Chart(document.getElementById('c2'), {{
  type: 'bar',
  data: {{
    labels,
    datasets: [{{ label: 'Prediction Error (INR)', data: errorInr,
                  backgroundColor: barColors, borderWidth: 0 }}]
  }},
  options: {{ ...baseOpts,
    plugins: {{ ...baseOpts.plugins,
      title: {{ display: true, text: 'Error: Predicted − Actual (INR)',
                color: '#58a6ff', font: {{ size: 14 }} }} }} }}
}});

new Chart(document.getElementById('c3'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [{{ label: 'Absolute Error (%)', data: errorPct.map(v => v === null ? null : Math.abs(v)),
                  borderColor: '#e377c2', backgroundColor: 'rgba(227,119,194,0.12)',
                  borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.1 }}]
  }},
  options: {{ ...baseOpts,
    plugins: {{ ...baseOpts.plugins,
      title: {{ display: true, text: 'Absolute Error (%)',
                color: '#58a6ff', font: {{ size: 14 }} }} }} }}
}});
</script>
</body>
</html>"""

    fname = f"plot_{plot_num:02d}_{year}.html"
    with open(os.path.join(output_folder, fname), 'w', encoding='utf-8') as f:
        f.write(html)
    return fname


# ──────────────────────────────────────────────────────────────────────────────
# 7. Index pages
# ──────────────────────────────────────────────────────────────────────────────

def _build_index_html(title, subtitle, plot_files, summary, overall_stats, other_link=None, other_label=None):
    th = ''.join(f'<th>{c}</th>' for c in summary.columns)
    tbody = ''
    for _, r in summary.iterrows():
        cells = ''.join(f'<td>{v}</td>' for v in r)
        tbody += f'<tr>{cells}</tr>\n'

    links = '\n'.join(
        f'<li><a href="{f}" target="_blank">{f}</a></li>'
        for f in plot_files
    )

    nav = ''
    if other_link and other_label:
        nav = f'<p style="text-align:center;margin-top:10px"><a href="{other_link}">→ {other_label}</a></p>'

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
  .sb {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
         padding: 14px 24px; text-align: center; }}
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
  .legend {{ display: flex; gap: 24px; justify-content: center; margin: 14px 0;
             font-size: 0.82em; flex-wrap: wrap; }}
  .li  {{ display: flex; align-items: center; gap: 7px; }}
  .dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
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

<div class="legend">
  <div class="li"><div class="dot" style="background:#ff5050"></div>
    <b>High vol</b>: <code>std5 &gt; std14 &gt; std25</code> &rarr; <code>close + dir &times; &sigma;_GARCH &times; close</code></div>
  <div class="li"><div class="dot" style="background:#50c864"></div>
    <b>Low vol</b>: <code>std5 &lt; std14 &lt; std25</code> &rarr; same price as today</div>
  <div class="li"><div class="dot" style="background:#648cff"></div>
    <b>Medium vol</b>: everything else &rarr; <code>close + dir &times; 0.5&sigma; &times; close</code></div>
</div>

<h2>Yearly Summary</h2>
<table>
  <thead><tr>{th}</tr></thead>
  <tbody>{tbody}</tbody>
</table>

<h2>Per-Year Interactive Plots</h2>
<p style="font-size:0.82em">Each plot: actual vs predicted close &middot; error bar chart &middot; abs error % &mdash; coloured by regime</p>
<ul>{links}</ul>
</body>
</html>"""


def create_index(output_folder, plot_files, summary, overall_stats, other_link=None, other_label=None):
    html = _build_index_html(
        title='USD/INR Next-Day Close Prediction (Train: 2003–2021)',
        subtitle='GARCH(1,1) volatility + EMA direction model + rolling-std regime',
        plot_files=plot_files,
        summary=summary,
        overall_stats=overall_stats,
        other_link=other_link,
        other_label=other_label,
    )
    path = os.path.join(output_folder, 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def create_validation_index(output_folder, plot_files, summary, overall_stats, other_link=None, other_label=None):
    html = _build_index_html(
        title='USD/INR Next-Day Close Prediction (Validation: 2022–2023)',
        subtitle='Frozen gamma/beta learned on 2003–2021 | GARCH + regime computed on full series',
        plot_files=plot_files,
        summary=summary,
        overall_stats=overall_stats,
        other_link=other_link,
        other_label=other_label,
    )
    path = os.path.join(output_folder, 'validation_index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


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


# ──────────────────────────────────────────────────────────────────────────────
# 8. Main
# ──────────────────────────────────────────────────────────────────────────────

def main(input_csv='USD_INR_Exchange.csv', output_folder='usd_inr_pred_plots'):
    print(f'\n{"="*60}')
    print('  USD/INR Next-Day Close Predictor')
    print(f'{"="*60}')

    # ── Load full dataset ─────────────────────────────────────────────────────
    print(f'\nLoading {input_csv} ...')
    raw = pd.read_csv(input_csv)
    raw.columns = raw.columns.str.strip()
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values('Date').reset_index(drop=True).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Split into train (2003-2019) and validation (2020-2021)
    df_train = raw[(raw.index.year >= 2003) & (raw.index.year <= 2021)].copy()
    df_val   = raw[(raw.index.year >= 2022) & (raw.index.year <= 2023)].copy()

    print(f'  Train : {len(df_train):,} days  ({df_train.index[0].date()} → {df_train.index[-1].date()})')
    print(f'  Val   : {len(df_val):,} days  ({df_val.index[0].date()} → {df_val.index[-1].date()})')

    # ── GARCH(1,1) — fit on train, apply to both ──────────────────────────────
    # Note: GARCH is fit on train returns; for validation we continue the
    # variance recursion forward using train-learned parameters.
    print('\n[1] Fitting GARCH(1,1) on train data (2003-2019) ...')
    train_ret = df_train['Close'].pct_change()
    df_train['garch_vol'] = fit_garch11(train_ret)

    # For validation: re-run GARCH on the full series up to end of val
    # so the variance recursion is properly initialised from history.
    # We use the SAME alpha/beta/omega found on training data.
    print('\n[1b] Extending GARCH to validation period ...')
    full_series = raw[(raw.index.year >= 2003) & (raw.index.year <= 2023)].copy()
    full_ret    = full_series['Close'].pct_change()
    full_series['garch_vol'] = fit_garch11(full_ret)
    df_val['garch_vol'] = full_series.loc[df_val.index, 'garch_vol']

    # ── Rolling-std regime ────────────────────────────────────────────────────
    # Compute regimes on the full series so rolling windows at 2020 boundary
    # are properly populated from 2019 history.
    print('\n[2] Computing rolling-std volatility regimes ...')
    full_series = classify_vol_regime(full_series)
    df_train[['std5','std14','std25','vol_regime']] = full_series.loc[df_train.index, ['std5','std14','std25','vol_regime']]
    df_val  [['std5','std14','std25','vol_regime']] = full_series.loc[df_val.index,   ['std5','std14','std25','vol_regime']]

    for label, dff in [('Train', df_train), ('Val', df_val)]:
        rc    = dff['vol_regime'].value_counts()
        total = len(dff)
        print(f'  {label}:')
        for regime in ['high', 'medium', 'low']:
            n = rc.get(regime, 0)
            print(f'    {regime:8s}: {n:5d} days  ({n/total*100:.1f}%)')

    # ── EMA feature — compute on full series for continuity ──────────────────
    print('\n[3] Computing EMA5-EMA20 direction feature ...')
    full_series = compute_ema_feature(full_series)
    df_train[['ema_5','ema_20','feat','close_delta','target']] = \
        full_series.loc[df_train.index, ['ema_5','ema_20','feat','close_delta','target']]
    df_val[['ema_5','ema_20','feat','close_delta','target']] = \
        full_series.loc[df_val.index, ['ema_5','ema_20','feat','close_delta','target']]

    # ── Learn thresholds ON TRAIN ONLY ───────────────────────────────────────
    print('\n[4] Optimising direction thresholds on TRAIN data only (2003-2021) ...')
    gamma, beta = learn_thresholds(df_train)
    print(f'  *** Thresholds FROZEN: gamma={gamma:.4f}  beta={beta:.4f} ***')

    # ── Predictions (same frozen gamma/beta for both sets) ────────────────────
    print('\n[5] Building predictions ...')
    print('  Train (2003-2019) ...')
    df_train = build_predictions(df_train, beta, gamma)
    print('  Validation (2020-2021) ...')
    df_val   = build_predictions(df_val,   beta, gamma)

    # ── Results ───────────────────────────────────────────────────────────────
    print('\n[6] Results — TRAIN (2003-2021)')
    summary_train = compute_summary(df_train)
    print_summary(summary_train, df_train, label='TRAIN SET')

    print('\n[6b] Results — VALIDATION (2022-2023)')
    summary_val = compute_summary(df_val)
    print_summary(summary_val, df_val, label='VALIDATION SET')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\n[7] Generating HTML plots ...')
    os.makedirs(output_folder, exist_ok=True)

    # Train plots
    train_plot_files = []
    for i, year in enumerate(range(2003, 2022), 1):
        ydf = df_train[df_train.index.year == year].copy()
        if ydf.empty:
            continue
        print(f'  Train {year} ({len(ydf)} days) ...')
        fname = plot_year_html(ydf, year, output_folder, i, index_file='index.html')
        train_plot_files.append(fname)

    # Validation plots
    val_plot_files = []
    for i, year in enumerate(range(2022, 2024), 1):
        ydf = df_val[df_val.index.year == year].copy()
        if ydf.empty:
            continue
        print(f'  Val {year} ({len(ydf)} days) ...')
        fname = plot_year_html(ydf, year, output_folder, i + 100, index_file='validation_index.html')
        val_plot_files.append(fname)

    # Index pages (cross-linked)
    train_stats = _overall_stats(df_train)
    val_stats   = _overall_stats(df_val)

    create_index(
        output_folder, train_plot_files, summary_train, train_stats,
        other_link='validation_index.html', other_label='Go to Validation Results (2022-2023)'
    )
    create_validation_index(
        output_folder, val_plot_files, summary_val, val_stats,
        other_link='index.html', other_label='Go to Train Results (2003-2019)'
    )

    print(f'\nDone.')
    print(f'  Train index   : {output_folder}/index.html')
    print(f'  Val index     : {output_folder}/validation_index.html')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    out_folder = sys.argv[2] if len(sys.argv) > 2 else 'usd_inr_pred_plots'
    main(csv_path, out_folder)