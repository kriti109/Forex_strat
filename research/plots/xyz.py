import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def compute_indicators(df):
    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']

    for w in [5, 20, 40, 75]:
        df[f'sma_{w}'] = c.rolling(w).mean()

    for w in [20, 40, 75]:
        df[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()

    df['hl_diff']   = h - l
    df['co_diff']   = c - o
    df['co_ratio']  = (c - o) / (h - l + 1e-9)

    df['raw_return'] = c.diff()

    for w in [5, 20, 40, 75]:
        df[f'rolling_std_{w}'] = c.rolling(w).std()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    ema12           = c.ewm(span=12, adjust=False).mean()
    ema26           = c.ewm(span=26, adjust=False).mean()
    df['macd']      = ema12 - ema26
    df['macd_sig']  = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']

    return df


def plot_year(year_df, year, plot_num, output_folder):
    x = year_df.index

    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            f'USD/INR {year} — Daily OHLC & Moving Averages',
            'HL Diff  |  CO Diff  |  CO Ratio',
            'Raw Return  &  Rolling Std',
            'RSI (14)  &  ATR (14)',
            'MACD (12, 26, 9)'
        ),
        row_heights=[0.42, 0.12, 0.16, 0.15, 0.15]
    )

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

    fig.add_trace(go.Scatter(x=x, y=year_df['hl_diff'],
        name='HL Diff', mode='lines', line=dict(color='#17becf', width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=year_df['co_diff'],
        name='CO Diff', mode='lines', line=dict(color='#bcbd22', width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=year_df['co_ratio'],
        name='CO Ratio', mode='lines', line=dict(color='#7f7f7f', width=1, dash='dot')), row=2, col=1)

    colors_ret = ['#26a69a' if v >= 0 else '#ef5350' for v in year_df['raw_return'].fillna(0)]
    fig.add_trace(go.Bar(x=x, y=year_df['raw_return'],
        name='Raw Return', marker_color=colors_ret, opacity=0.6), row=3, col=1)

    std_colors = {5: '#1f77b4', 20: '#ff7f0e', 40: '#9467bd', 75: '#8c564b'}
    for w, col in std_colors.items():
        fig.add_trace(go.Scatter(
            x=x, y=year_df[f'rolling_std_{w}'],
            name=f'Std {w}', mode='lines',
            line=dict(color=col, width=1, dash='dot')
        ), row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=year_df['rsi'],
        name='RSI(14)', mode='lines', line=dict(color='#e377c2', width=1.5)), row=4, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='red',   line_width=1, row=4, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='green', line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=year_df['atr'],
        name='ATR(14)', mode='lines',
        line=dict(color='#ff7f0e', width=1.5, dash='dot')), row=4, col=1)

    fig.add_trace(go.Scatter(x=x, y=year_df['macd'],
        name='MACD', mode='lines', line=dict(color='#1f77b4', width=1.5)), row=5, col=1)
    fig.add_trace(go.Scatter(x=x, y=year_df['macd_sig'],
        name='Signal', mode='lines', line=dict(color='#ff7f0e', width=1.5, dash='dash')), row=5, col=1)
    colors_hist = ['#26a69a' if v >= 0 else '#ef5350' for v in year_df['macd_hist'].fillna(0)]
    fig.add_trace(go.Bar(x=x, y=year_df['macd_hist'],
        name='Histogram', marker_color=colors_hist, opacity=0.6), row=5, col=1)

    fig.update_layout(
        title=dict(text=f'USD/INR Daily Analysis — {year}', x=0.5, xanchor='center', font=dict(size=18)),
        height=1000,
        width=1600,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation='v', x=1.01, y=1, font=dict(size=10)),
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        xaxis=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),
        xaxis2=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),
        xaxis3=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),
        xaxis4=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),
        xaxis5=dict(rangebreaks=[dict(bounds=['sat', 'mon'])]),
    )
    fig.update_yaxes(title_text='Price (INR)',  row=1, col=1)
    fig.update_yaxes(title_text='Spread',       row=2, col=1)
    fig.update_yaxes(title_text='Return/Std',   row=3, col=1)
    fig.update_yaxes(title_text='RSI / ATR',    row=4, col=1)
    fig.update_yaxes(title_text='MACD',         row=5, col=1)

    filename = f'plot_{plot_num:02d}_{year}.html'
    filepath = os.path.join(output_folder, filename)
    fig.write_html(filepath)
    print(f'Saved: {filepath}')
    return filename


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
    li   {{ margin: 10px 0; font-size: 16px; }}
    a    {{ color: #0066cc; }}
  </style>
</head>
<body>
  <h1>USD/INR — Yearly Plots 2003–2019</h1>
  <ul>{links}</ul>
</body>
</html>"""
    path = os.path.join(output_folder, 'index.html')
    with open(path, 'w') as f:
        f.write(html)
    print(f'Index saved: {path}')


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

    print('Computing indicators on full series ...')
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