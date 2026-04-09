import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


# ============================================================================
# GARCH(p, q) — pure-numpy implementation
#   p : number of lagged squared-return (ARCH) terms
#   q : number of lagged variance         (GARCH) terms
#
#   h_t = omega
#         + sum_{i=1}^{p} alpha_i * r_{t-i}^2
#         + sum_{j=1}^{q} beta_j  * h_{t-j}
#
#   Estimation: coarse grid over scalar alpha / beta totals,
#               each distributed evenly across p / q lags.
#   Returns:
#     cond_vol  : np.ndarray  conditional std dev  (same length as returns)
#     rt        : np.ndarray  standardised residuals  r_t / sigma_t
#     params    : dict        fitted omega, alphas, betas, persistence
# ============================================================================

def _garch_variance(r, omega, alphas, betas):
    """Compute GARCH(p,q) conditional variance series."""
    p   = len(alphas)
    q   = len(betas)
    n   = len(r)
    lag = max(p, q)

    h = np.full(n, max(np.var(r), 1e-10))

    for t in range(lag, n):
        val = omega
        for i in range(p):
            val += alphas[i] * r[t - 1 - i] ** 2
        for j in range(q):
            val += betas[j] * h[t - 1 - j]
        h[t] = max(val, 1e-12)

    return h


def _neg_log_lik(r, omega, alphas, betas):
    h = _garch_variance(r, omega, alphas, betas)
    return 0.5 * np.sum(np.log(h) + r ** 2 / h)


def fit_garch(returns, p=1, q=1):
    """
    Fit GARCH(p,q) via coarse grid search over total alpha / beta persistence.
    Each lag shares equal weight (alpha_i = alpha_total / p, etc.).

    Returns
    -------
    cond_vol : np.ndarray   conditional std dev (daily, in return units)
    rt       : np.ndarray   standardised residuals  r_t / sigma_t
    params   : dict         fitted omega, alphas, betas, persistence
    """
    r   = returns.fillna(0).values
    uv  = max(np.var(r), 1e-10)

    best_nll    = np.inf
    best_params = None

    for a_sum in np.linspace(0.04, 0.30, 7):
        for b_sum in np.linspace(0.55, 0.92, 8):
            if a_sum + b_sum >= 0.9999:
                continue
            omega = uv * (1.0 - a_sum - b_sum)
            if omega <= 0:
                continue
            alphas = np.full(p, a_sum / p)
            betas  = np.full(q, b_sum / q)
            nll = _neg_log_lik(r, omega, alphas, betas)
            if nll < best_nll:
                best_nll    = nll
                best_params = (omega, alphas.copy(), betas.copy())

    if best_params is None:
        omega  = uv * 0.05
        alphas = np.full(p, 0.10 / p)
        betas  = np.full(q, 0.80 / q)
        best_params = (omega, alphas, betas)

    omega, alphas, betas = best_params
    h        = _garch_variance(r, omega, alphas, betas)
    cond_vol = np.sqrt(np.maximum(h, 1e-12))
    rt       = r / cond_vol

    params = dict(omega=omega, alphas=alphas, betas=betas,
                  p=p, q=q,
                  persistence=float(alphas.sum() + betas.sum()))
    return cond_vol, rt, params


# ============================================================================
# Technical indicators
# ============================================================================

def compute_indicators(df):
    c = df['Close']
    h = df['High']
    l = df['Low']

    for w in [5, 20, 40, 75]:
        df[f'sma_{w}'] = c.rolling(w).mean()
    for w in [20, 40, 75]:
        df[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    return df


# ============================================================================
# Fit all three GARCH models and attach columns to df
# ============================================================================

def add_garch_models(df):
    """
    Fit GARCH(1,1), (2,2), (3,3) on percentage-return series.
    Columns added per model (tag = garch11 / garch22 / garch33):
        {tag}_vol   conditional standard deviation (in return units)
        {tag}_rt    standardised residuals  r_t / sigma_t
    Extra columns for price-band plotting (GARCH(1,1) only):
        g11_upper1 / g11_lower1   ±1σ  (in price units)
        g11_upper2 / g11_lower2   ±2σ  (in price units)
    """
    pct_ret = df['Close'].pct_change()

    for p in [1, 2, 3]:
        tag = f'garch{p}{p}'
        print(f'  Fitting GARCH({p},{p}) ...')
        cond_vol, rt, params = fit_garch(pct_ret, p=p, q=p)

        df[f'{tag}_vol'] = cond_vol
        df[f'{tag}_rt']  = rt

        a_str = '  '.join(f'α{i+1}={v:.4f}' for i, v in enumerate(params['alphas']))
        b_str = '  '.join(f'β{i+1}={v:.4f}' for i, v in enumerate(params['betas']))
        print(f'    ω={params["omega"]:.2e}  {a_str}  {b_str}  '
              f'persistence={params["persistence"]:.4f}')

    # Price-space bands from GARCH(1,1)
    v11 = df['garch11_vol']
    df['g11_upper1'] = df['Close'] * (1 + v11)
    df['g11_lower1'] = df['Close'] * (1 - v11)
    df['g11_upper2'] = df['Close'] * (1 + 2 * v11)
    df['g11_lower2'] = df['Close'] * (1 - 2 * v11)

    return df


# ============================================================================
# Per-year plot
# ============================================================================

GARCH_VOL_STYLES = {
    'garch11': dict(color='#FF8C00', dash='solid', width=1.5, label='GARCH(1,1) σ_t'),
    'garch22': dict(color='#00BFFF', dash='dash',  width=1.5, label='GARCH(2,2) σ_t'),
    'garch33': dict(color='#7CFC00', dash='dot',   width=1.8, label='GARCH(3,3) σ_t'),
}

GARCH_RT_STYLES = {
    'garch11': dict(color='rgba(255,140,0,0.8)',  dash='solid', width=1.2, label='r_t  GARCH(1,1)'),
    'garch22': dict(color='rgba(0,191,255,0.8)',  dash='dash',  width=1.2, label='r_t  GARCH(2,2)'),
    'garch33': dict(color='rgba(124,252,0,0.85)', dash='dot',   width=1.4, label='r_t  GARCH(3,3)'),
}


def plot_year(year_df, year, plot_num, output_folder):
    x = year_df.index

    # ── 4-panel layout ──────────────────────────────────────────────────────
    # Row 1 : OHLC + MAs + GARCH(1,1) ±1σ / ±2σ price bands
    # Row 2 : Conditional volatility σ_t — all three models
    # Row 3 : Standardised residuals r_t  (left y)  +  Close price (right y)
    # Row 4 : RSI (left y)  +  ATR (right y)

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        subplot_titles=(
            f'USD/INR {year} — OHLC, Moving Averages & GARCH(1,1) ±1σ / ±2σ Bands',
            'Conditional Volatility σ_t — GARCH(1,1) vs (2,2) vs (3,3)',
            'Standardised Residuals r_t  &  Close Price',
            'RSI(14)  |  ATR(14)',
        ),
        row_heights=[0.40, 0.18, 0.24, 0.18],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": True}],
        ]
    )

    # ── Row 1 : Candlestick + MAs + GARCH bands ─────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x,
        open=year_df['Open'], high=year_df['High'],
        low=year_df['Low'],   close=year_df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a', increasing_fillcolor='#26a69a',
        decreasing_line_color='#ef5350', decreasing_fillcolor='#ef5350',
        line=dict(width=1)
    ), row=1, col=1)

    sma_colors = {5: '#1f77b4', 20: '#ff7f0e', 40: '#9467bd', 75: '#8c564b'}
    for w, col in sma_colors.items():
        fig.add_trace(go.Scatter(
            x=x, y=year_df[f'sma_{w}'],
            name=f'SMA {w}', mode='lines',
            line=dict(color=col, width=1.2)
        ), row=1, col=1)

    ema_colors = {20: '#d62728', 40: '#2ca02c', 75: '#e377c2'}
    for w, col in ema_colors.items():
        fig.add_trace(go.Scatter(
            x=x, y=year_df[f'ema_{w}'],
            name=f'EMA {w}', mode='lines',
            line=dict(color=col, width=1.2, dash='dash')
        ), row=1, col=1)

    # ±2σ outer band (lightest fill)
    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(year_df['g11_upper2']) + list(year_df['g11_lower2'][::-1]),
        fill='toself', fillcolor='rgba(255,140,0,0.07)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        name='GARCH(1,1) ±2σ band', hoverinfo='skip'
    ), row=1, col=1)

    # ±1σ inner band (slightly more opaque)
    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(year_df['g11_upper1']) + list(year_df['g11_lower1'][::-1]),
        fill='toself', fillcolor='rgba(255,140,0,0.16)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        name='GARCH(1,1) ±1σ band', hoverinfo='skip'
    ), row=1, col=1)

    # ±1σ boundary lines (thin dotted)
    for col_key in ('g11_upper1', 'g11_lower1'):
        fig.add_trace(go.Scatter(
            x=x, y=year_df[col_key], mode='lines',
            line=dict(color='rgba(255,140,0,0.55)', width=0.8, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ), row=1, col=1)

    # ── Row 2 : Conditional volatility σ_t ──────────────────────────────────
    for tag, style in GARCH_VOL_STYLES.items():
        col_name = f'{tag}_vol'
        if col_name not in year_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=year_df[col_name],
            name=style['label'], mode='lines',
            line=dict(color=style['color'], width=style['width'], dash=style['dash'])
        ), row=2, col=1)

    # ── Row 3 : Standardised residuals r_t  +  Close price ──────────────────
    for tag, style in GARCH_RT_STYLES.items():
        col_name = f'{tag}_rt'
        if col_name not in year_df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=x, y=year_df[col_name],
            name=style['label'], mode='lines',
            line=dict(color=style['color'], width=style['width'], dash=style['dash'])
        ), row=3, col=1, secondary_y=False)

    # Reference lines for r_t
    fig.add_hline(y=0,  line_dash='dot',  line_color='rgba(160,160,160,0.5)', line_width=1, row=3, col=1)
    fig.add_hline(y=2,  line_dash='dash', line_color='rgba(255,80,80,0.45)',  line_width=1, row=3, col=1)
    fig.add_hline(y=-2, line_dash='dash', line_color='rgba(80,200,80,0.45)',  line_width=1, row=3, col=1)

    # Close price on secondary y-axis (right)
    fig.add_trace(go.Scatter(
        x=x, y=year_df['Close'],
        name='Close Price (INR)',
        mode='lines',
        line=dict(color='rgba(200,200,200,0.85)', width=1.4)
    ), row=3, col=1, secondary_y=True)

    # ── Row 4 : RSI (left) + ATR (right) ────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=year_df['rsi'],
        name='RSI(14)', mode='lines',
        line=dict(color='#e377c2', width=1.6)
    ), row=4, col=1, secondary_y=False)

    fig.add_hline(y=70, line_dash='dash', line_color='red',   line_width=1, row=4, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', line_width=1, row=4, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=year_df['atr'],
        name='ATR(14)', mode='lines',
        line=dict(color='#ff7f0e', width=1.6, dash='dot')
    ), row=4, col=1, secondary_y=True)

    # ── Global layout ────────────────────────────────────────────────────────
    rb = [dict(bounds=['sat', 'mon'])]

    fig.update_layout(
        title=dict(text=f'USD/INR Daily Analysis — {year}',
                   x=0.5, xanchor='center', font=dict(size=18)),
        height=1250,
        width=1750,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', x=1.02, y=1, font=dict(size=10), tracegroupgap=4),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        xaxis=dict(rangebreaks=rb),
        xaxis2=dict(rangebreaks=rb),
        xaxis3=dict(rangebreaks=rb),
        xaxis4=dict(rangebreaks=rb),
    )

    fig.update_yaxes(title_text='Price (INR)',         row=1, col=1)
    fig.update_yaxes(title_text='σ_t (daily %)',       row=2, col=1)
    fig.update_yaxes(title_text='r_t',                 row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text='Close (INR)',          row=3, col=1, secondary_y=True, showgrid=False)
    fig.update_yaxes(title_text='RSI', range=[0, 100], row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text='ATR (INR)',            row=4, col=1, secondary_y=True, showgrid=False)

    filename = f'plot_{plot_num:02d}_{year}.html'
    filepath = os.path.join(output_folder, filename)
    fig.write_html(filepath)
    print(f'  Saved: {filepath}')
    return filename


# ============================================================================
# Index page
# ============================================================================

def create_index(output_folder, plot_files):
    links = '\n'.join(
        f'<li><a href="{f}" target="_blank">{f}</a></li>'
        for f in plot_files
    )
    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>USD/INR Analysis 2003-2019</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 60px auto; }}
    h1   {{ text-align: center; }}
    p    {{ text-align: center; color: #555; }}
    li   {{ margin: 10px 0; font-size: 16px; }}
    a    {{ color: #0066cc; }}
  </style>
</head>
<body>
  <h1>USD/INR — Yearly Plots 2003–2019</h1>
  <p>
    Panel 1: OHLC + MAs + GARCH(1,1) ±σ bands &nbsp;|&nbsp;
    Panel 2: σ_t — all three models &nbsp;|&nbsp;
    Panel 3: r_t + Close &nbsp;|&nbsp;
    Panel 4: RSI / ATR
  </p>
  <ul>{links}</ul>
</body>
</html>"""
    path = os.path.join(output_folder, 'index.html')
    with open(path, 'w') as f:
        f.write(html)
    print(f'Index saved: {path}')


# ============================================================================
# Main
# ============================================================================

def main(input_csv, output_folder='usd_inr_plots'):
    print(f'Loading {input_csv} ...')
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.set_index('Date')

    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    print('Computing technical indicators ...')
    df = compute_indicators(df)

    print('Fitting GARCH(1,1), (2,2), (3,3) on full return series ...')
    df = add_garch_models(df)

    os.makedirs(output_folder, exist_ok=True)

    plot_files = []
    for i, year in enumerate(range(2003, 2020), 1):
        year_df = df[df.index.year == year].copy()
        if year_df.empty:
            print(f'No data for {year}, skipping.')
            continue
        print(f'Plotting {year} — {len(year_df)} trading days ...')
        fname = plot_year(year_df, year, i, output_folder)
        plot_files.append(fname)

    create_index(output_folder, plot_files)
    print(f'\nDone — {len(plot_files)} plots saved to "{output_folder}/"')


if __name__ == '__main__':
    import sys
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    out_folder = sys.argv[2] if len(sys.argv) > 2 else 'usd_inr_plots'
    main(csv_path, out_folder)