"""
USD/INR 2004 — Presentation Plot
==================================
4 panels, dark theme, publication quality:
  Panel 1 (large) : OHLC candlesticks + EMA 5/20/40/75 + GARCH(1,1) ±1σ / ±2σ bands
  Panel 2 (thin)  : feat = EMA5 − EMA20  (with zero-line)
  Panel 3 (thin)  : Rolling std — 5, 14, 25 days

Usage:
  python presentation_plot_2004.py [csv_path]
  Default csv: USD_INR_Exchange.csv
  Output     : usd_inr_2004_presentation.html
"""

import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────
# Palette  (refined dark — presentation grade)
# ─────────────────────────────────────────────────────────────
BG        = '#09090f'
PANEL_BG  = '#0e0e18'
GRID      = 'rgba(255,255,255,0.04)'
TICK_COL  = '#5a5a7a'
LABEL_COL = '#9090b0'
TITLE_COL = '#ddddf0'

UP_COL    = '#00c9a0'
DOWN_COL  = '#ff4d6d'

EMA_COLS  = {
    5:  '#f0c040',
    20: '#60a8ff',
    40: '#c084fc',
    75: '#fb923c',
}

BAND_FILL_2 = 'rgba(96,168,255,0.04)'
BAND_FILL_1 = 'rgba(96,168,255,0.09)'
BAND_LINE   = 'rgba(96,168,255,0.30)'

FEAT_COL    = '#f0c040'
STD_COLS    = {
    5:  '#00c9a0',
    14: '#60a8ff',
    25: '#ff4d6d',
}


# ─────────────────────────────────────────────────────────────
# GARCH(1,1)
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# Build features
# ─────────────────────────────────────────────────────────────

def build(df):
    c = df['Close']
    for w in [5, 20, 40, 75]:
        df[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()
    df['feat']  = df['ema_5'] - df['ema_20']
    df['std5']  = c.rolling(5).std()
    df['std14'] = c.rolling(14).std()
    df['std25'] = c.rolling(25).std()
    pct_ret     = c.pct_change()
    gvol        = fit_garch11(pct_ret)
    df['garch_vol'] = gvol
    df['g_upper2']  = c * (1 + 2 * gvol)
    df['g_lower2']  = c * (1 - 2 * gvol)
    df['g_upper1']  = c * (1 + gvol)
    df['g_lower1']  = c * (1 - gvol)
    return df


# ─────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────

def make_plot(df_2004, out_path):
    x = df_2004.index

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.0,
        row_heights=[0.65, 0.18, 0.17],
    )

    # ── Panel 1 : GARCH bands ────────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(df_2004['g_upper2']) + list(df_2004['g_lower2'][::-1]),
        fill='toself', fillcolor=BAND_FILL_2,
        line=dict(color='rgba(0,0,0,0)', width=0),
        name='GARCH ±2σ', hoverinfo='skip', showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(df_2004['g_upper1']) + list(df_2004['g_lower1'][::-1]),
        fill='toself', fillcolor=BAND_FILL_1,
        line=dict(color='rgba(0,0,0,0)', width=0),
        name='GARCH ±1σ', hoverinfo='skip', showlegend=True,
    ), row=1, col=1)

    for col_key in ['g_upper1', 'g_lower1']:
        fig.add_trace(go.Scatter(
            x=x, y=df_2004[col_key], mode='lines',
            line=dict(color=BAND_LINE, width=0.8, dash='dot'),
            showlegend=False, hoverinfo='skip',
        ), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=x,
        open=df_2004['Open'], high=df_2004['High'],
        low=df_2004['Low'],   close=df_2004['Close'],
        name='OHLC',
        increasing_line_color=UP_COL,   increasing_fillcolor=UP_COL,
        decreasing_line_color=DOWN_COL, decreasing_fillcolor=DOWN_COL,
        line=dict(width=1),
        whiskerwidth=0.3,
    ), row=1, col=1)

    ema_dash  = {5: 'solid', 20: 'solid', 40: 'dash', 75: 'dot'}
    ema_width = {5: 1.2, 20: 1.4, 40: 1.2, 75: 1.2}
    for w in [75, 40, 20, 5]:
        fig.add_trace(go.Scatter(
            x=x, y=df_2004[f'ema_{w}'],
            name=f'EMA {w}', mode='lines',
            line=dict(color=EMA_COLS[w], width=ema_width[w], dash=ema_dash[w]),
        ), row=1, col=1)

    # ── Panel 2 : feat = EMA5 − EMA20 ───────────────────────
    feat = df_2004['feat']
    pos  = feat.clip(lower=0)
    neg  = feat.clip(upper=0)

    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(pos) + [0] * len(x),
        fill='toself', fillcolor='rgba(0,201,160,0.18)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        name='feat > 0', hoverinfo='skip', showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(x) + list(x[::-1]),
        y=list(neg) + [0] * len(x),
        fill='toself', fillcolor='rgba(255,77,109,0.18)',
        line=dict(color='rgba(0,0,0,0)', width=0),
        name='feat < 0', hoverinfo='skip', showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=feat,
        name='feat (EMA5−EMA20)', mode='lines',
        line=dict(color=FEAT_COL, width=1.5),
    ), row=2, col=1)

    fig.add_hline(y=0, line_color='rgba(255,255,255,0.15)',
                  line_width=1, line_dash='dot', row=2, col=1)

    # ── Panel 3 : Rolling std ────────────────────────────────
    for w, lw, dash in [(25, 1.2, 'dot'), (14, 1.4, 'dash'), (5, 1.6, 'solid')]:
        fig.add_trace(go.Scatter(
            x=x, y=df_2004[f'std{w}'],
            name=f'Std {w}d', mode='lines',
            line=dict(color=STD_COLS[w], width=lw, dash=dash),
        ), row=3, col=1)

    # ── Tight y-axis range for price panel ───────────────────
    price_min = df_2004[['Low', 'g_lower2']].min().min()
    price_max = df_2004[['High', 'g_upper2']].max().max()
    pad = (price_max - price_min) * 0.05

    # ── Layout — scaled for laptop screen (~1366×768) ────────
    rangebreaks = [dict(bounds=['sat', 'mon'])]

    fig.update_layout(
        title=dict(
            text='USD / INR  ·  2004',
            x=0.5, xanchor='center',
            font=dict(size=16, color=TITLE_COL, family='Georgia, serif'),
        ),
        height=680,        # fits inside browser on a 768p laptop
        width=1280,        # safe for 1366-wide screens
        paper_bgcolor=BG,
        plot_bgcolor=PANEL_BG,
        showlegend=True,
        legend=dict(
            orientation='v',
            x=1.01, y=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=9, color=LABEL_COL, family='Courier New, monospace'),
            tracegroupgap=1,
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1a1a2e',
            bordercolor='#3a3a5c',
            font=dict(color='#ddddf0', size=10, family='Courier New, monospace'),
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=55, r=140, t=45, b=35),
        xaxis=dict(rangebreaks=rangebreaks),
        xaxis2=dict(rangebreaks=rangebreaks),
        xaxis3=dict(rangebreaks=rangebreaks),
    )

    axis_style = dict(
        gridcolor=GRID,
        zerolinecolor='rgba(255,255,255,0.08)',
        tickfont=dict(color=TICK_COL, size=9, family='Courier New, monospace'),
        title_font=dict(color=LABEL_COL, size=10, family='Courier New, monospace'),
        showgrid=True,
        ticks='outside',
        tickcolor='rgba(255,255,255,0.10)',
        linecolor='rgba(255,255,255,0.06)',
        showline=True,
    )

    fig.update_yaxes(
        title_text='Price (INR)', row=1, col=1,
        range=[price_min - pad, price_max + pad],
        **axis_style
    )
    fig.update_yaxes(title_text='EMA5−EMA20', row=2, col=1, **axis_style)
    fig.update_yaxes(title_text='Rolling Std', row=3, col=1, **axis_style)

    fig.update_xaxes(**{k: v for k, v in axis_style.items()
                        if k not in ('title_text',)},
                     showticklabels=False, row=1, col=1)
    fig.update_xaxes(**{k: v for k, v in axis_style.items()
                        if k not in ('title_text',)},
                     showticklabels=False, row=2, col=1)
    fig.update_xaxes(
        **{k: v for k, v in axis_style.items() if k not in ('title_text',)},
        tickformat='%b %Y',
        dtick='M1',
        tickangle=-30,
        row=3, col=1,
    )

    for row in [1, 2]:
        fig.add_hline(
            y=0, line_color='rgba(255,255,255,0.06)',
            line_width=1, row=row, col=1,
        )

    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f'  Saved: {out_path}')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main(input_csv='USD_INR_Exchange.csv'):
    print(f'Loading {input_csv} ...')
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    df_full = df[df.index.year.isin([2003, 2004])].copy()
    print('Building features ...')
    df_full = build(df_full)

    df_2004 = df_full[df_full.index.year == 2004].copy()
    print(f'  2004: {len(df_2004)} trading days')

    out_path = 'usd_inr_2004_presentation.html'
    print('Rendering plot ...')
    make_plot(df_2004, out_path)
    print('Done.')


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    main(csv_path)