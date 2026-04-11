import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


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

    return df


# ============================================================================
# Heikin-Ashi candles
# ============================================================================

def compute_ha(df):
    ha = pd.DataFrame(index=df.index)

    ha['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    ha_open = np.zeros(len(df))
    ha_open[0] = (df['Open'].iloc[0] + df['Close'].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + ha['Close'].iloc[i - 1]) / 2
    ha['Open'] = ha_open

    ha['High'] = pd.concat([df['High'], ha['Open'], ha['Close']], axis=1).max(axis=1)
    ha['Low']  = pd.concat([df['Low'],  ha['Open'], ha['Close']], axis=1).min(axis=1)

    return ha


# ============================================================================
# Kalman filter
# Q : process noise  — higher = filter tracks price more aggressively
# R : observation noise — higher = smoother output, less reactive
# ============================================================================

def kalman_filter(prices, Q=1e-5, R=1e-2):
    n        = len(prices)
    filtered = np.zeros(n)
    gains    = np.zeros(n)

    x = prices[0]
    P = 1.0

    for i, z in enumerate(prices):
        P = P + Q
        K = P / (P + R)
        x = x + K * (z - x)
        P = (1 - K) * P

        filtered[i] = x
        gains[i]    = K

    return filtered, gains


# ============================================================================
# Per-year plot
# ============================================================================

def plot_year(year_df, year, plot_num, output_folder, Q=1e-5, R=1e-2):
    x      = year_df.index
    prices = year_df['Close'].values

    kalman_line, kalman_gain = kalman_filter(prices, Q=Q, R=R)
    ha = compute_ha(year_df)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            f'USD/INR {year} — OHLC, HA, Moving Averages & Kalman Filter',
            'Kalman Gain',
        ),
        row_heights=[0.75, 0.25],
    )

    # ── Regular candlestick ──────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=x,
        open=year_df['Open'], high=year_df['High'],
        low=year_df['Low'],   close=year_df['Close'],
        name='OHLC',
        increasing_line_color='#26a69a', increasing_fillcolor='#26a69a',
        decreasing_line_color='#ef5350', decreasing_fillcolor='#ef5350',
        line=dict(width=1),
        opacity=0.6,
    ), row=1, col=1)

    # ── Heikin-Ashi candles (yellow=bullish, purple=bearish) ─────────────────
    ha_bull = ha['Close'] >= ha['Open']

    fig.add_trace(go.Candlestick(
        x=x[ha_bull],
        open=ha.loc[ha_bull,  'Open'],
        high=ha.loc[ha_bull,  'High'],
        low=ha.loc[ha_bull,   'Low'],
        close=ha.loc[ha_bull, 'Close'],
        name='HA Bullish',
        increasing_line_color='#fdd835', increasing_fillcolor='#fdd835',
        decreasing_line_color='#fdd835', decreasing_fillcolor='#fdd835',
        line=dict(width=1),
        opacity=0.75,
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Candlestick(
        x=x[~ha_bull],
        open=ha.loc[~ha_bull,  'Open'],
        high=ha.loc[~ha_bull,  'High'],
        low=ha.loc[~ha_bull,   'Low'],
        close=ha.loc[~ha_bull, 'Close'],
        name='HA Bearish',
        increasing_line_color='#8e24aa', increasing_fillcolor='#8e24aa',
        decreasing_line_color='#8e24aa', decreasing_fillcolor='#8e24aa',
        line=dict(width=1),
        opacity=0.75,
        showlegend=True,
    ), row=1, col=1)

    # ── Close price line ─────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=year_df['Close'],
        name='Close', mode='lines',
        line=dict(color='#ffffff', width=1.2, dash='dot'),
    ), row=1, col=1)

    # ── SMAs ─────────────────────────────────────────────────────────────────
    sma_colors = {5: '#1f77b4', 20: '#ff7f0e', 40: '#9467bd', 75: '#8c564b'}
    for w, col in sma_colors.items():
        fig.add_trace(go.Scatter(
            x=x, y=year_df[f'sma_{w}'],
            name=f'SMA {w}', mode='lines',
            line=dict(color=col, width=1.2)
        ), row=1, col=1)

    # ── EMAs ─────────────────────────────────────────────────────────────────
    ema_colors = {20: '#d62728', 40: '#2ca02c', 75: '#e377c2'}
    for w, col in ema_colors.items():
        fig.add_trace(go.Scatter(
            x=x, y=year_df[f'ema_{w}'],
            name=f'EMA {w}', mode='lines',
            line=dict(color=col, width=1.2, dash='dash')
        ), row=1, col=1)

    # ── Kalman overlay ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=kalman_line,
        name='Kalman (filtered)', mode='lines',
        line=dict(color='#00bcd4', width=2.2),
    ), row=1, col=1)

    # ── Kalman gain panel ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=kalman_gain,
        name='Kalman Gain (K)', mode='lines',
        line=dict(color='#ff9800', width=1.6),
        fill='tozeroy',
        fillcolor='rgba(255,152,0,0.12)'
    ), row=2, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    rb = [dict(bounds=['sat', 'mon'])]

    fig.update_layout(
        title=dict(
            text=f'USD/INR Daily Analysis — {year}  (Kalman Q={Q}, R={R})',
            x=0.5, xanchor='center', font=dict(size=18)
        ),
        height=850,
        width=1750,
        template='plotly_dark',        # dark bg makes yellow/purple HA pop
        showlegend=True,
        legend=dict(orientation='v', x=1.02, y=1, font=dict(size=10), tracegroupgap=4),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        xaxis=dict(rangebreaks=rb),
        xaxis2=dict(rangebreaks=rb),
    )

    fig.update_yaxes(title_text='Price (INR)', row=1, col=1)
    fig.update_yaxes(title_text='Gain (K)',    row=2, col=1)

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
  <title>USD/INR Analysis 2004-2005</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 700px; margin: 60px auto; }}
    h1   {{ text-align: center; }}
    p    {{ text-align: center; color: #555; }}
    li   {{ margin: 10px 0; font-size: 16px; }}
    a    {{ color: #0066cc; }}
  </style>
</head>
<body>
  <h1>USD/INR — Yearly Plots 2004–2005</h1>
  <p>
    Panel 1: OHLC + HA Candles + Close Line + Moving Averages + Kalman overlay &nbsp;|&nbsp;
    Panel 2: Kalman Gain
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

    os.makedirs(output_folder, exist_ok=True)

    plot_files = []
    for i, year in enumerate(range(2004, 2006), 1):
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