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

    # RSI(5)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(5).mean()
    loss  = (-delta.clip(upper=0)).rolling(5).mean()
    df['rsi5'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # ATR(5)
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr5'] = tr.rolling(5).mean()

    return df


# ============================================================================
# Per-year plot
# ============================================================================

def plot_year(year_df, year, plot_num, output_folder):
    x = year_df.index

    # ── 3-panel layout ──────────────────────────────────────────────────────
    # Row 1 : OHLC + MAs
    # Row 2 : RSI(5)
    # Row 3 : ATR(5)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            f'USD/INR {year} — OHLC & Moving Averages',
            'RSI(5)',
            'ATR(5)',
        ),
        row_heights=[0.60, 0.20, 0.20],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ]
    )

    # ── Row 1 : Candlestick + MAs ────────────────────────────────────────────
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

    # ── Row 2 : RSI(5) ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=year_df['rsi5'],
        name='RSI(5)', mode='lines',
        line=dict(color='#e377c2', width=1.6)
    ), row=2, col=1)

    fig.add_hline(y=70, line_dash='dash', line_color='red',   line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', line_width=1, row=2, col=1)
    fig.add_hline(y=50, line_dash='dot',  line_color='rgba(160,160,160,0.5)', line_width=1, row=2, col=1)

    # ── Row 3 : ATR(5) ───────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=year_df['atr5'],
        name='ATR(5)', mode='lines',
        line=dict(color='#ff7f0e', width=1.6)
    ), row=3, col=1)

    # ── Global layout ────────────────────────────────────────────────────────
    rb = [dict(bounds=['sat', 'mon'])]

    fig.update_layout(
        title=dict(text=f'USD/INR Daily Analysis — {year}',
                   x=0.5, xanchor='center', font=dict(size=18)),
        height=900,
        width=1750,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', x=1.02, y=1, font=dict(size=10), tracegroupgap=4),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        xaxis=dict(rangebreaks=rb),
        xaxis2=dict(rangebreaks=rb),
        xaxis3=dict(rangebreaks=rb),
    )

    fig.update_yaxes(title_text='Price (INR)',  row=1, col=1)
    fig.update_yaxes(title_text='RSI', range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text='ATR (INR)',    row=3, col=1)

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
    Panel 1: OHLC + Moving Averages &nbsp;|&nbsp;
    Panel 2: RSI(5) &nbsp;|&nbsp;
    Panel 3: ATR(5)
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