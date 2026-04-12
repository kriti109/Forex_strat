"""
USD/INR Next-Day Close Price Predictor  —  Two-Stage Model (LDA + Ridge)
===============================================================================
Model Architecture:
  Stage 1 (Direction): Linear Discriminant Analysis (LDA) predicts market sign (+1 / -1).
  Stage 2 (Magnitude): RidgeCV predicts the absolute size of the return.
  Synthesis: Return = Direction * Magnitude -> Reconstruct Next-Day Close.

Feature Split:
  Direction Features: Momentum, returns, and moving averages.
  Magnitude Features: Volatility, range statistics, and intraday microstructure.

Walk-forward (2021-2023):
  - Fit initially on 2003-2020.
  - Refit every 7 trading days on a 1000-day ROLLING window.
  - Zero lookahead guaranteed.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. GARCH(1,1)  -- for vol_regime classification only
# ------------------------------------------------------------------------------

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
            if a + b >= 0.9999: continue
            omega = uv * (1.0 - a - b)
            if omega <= 0: continue
            h   = _garch_variance(r, omega, a, b)
            nll = 0.5 * float(np.sum(np.log(h) + r**2 / h))
            if nll < best_nll:
                best_nll    = nll
                best_params = (omega, a, b)
    if best_params is None:
        best_params = (uv * 0.05, 0.10, 0.80)
    omega, alpha, beta = best_params
    h = _garch_variance(r, omega, alpha, beta)
    print(f"  GARCH(1,1): omega={omega:.2e}  alpha={alpha:.4f}  beta={beta:.4f}")
    return np.sqrt(np.maximum(h, 1e-12))

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

# ------------------------------------------------------------------------------
# 2. Feature engineering & Split
# ------------------------------------------------------------------------------

# We intentionally split features so models don't get confused by conflicting signals
DIR_FEATURE_COLS = ['feat', 'price_vs_ema20', 'ret_1d', 'ret_3d', 'prev_direction']
MAG_FEATURE_COLS = ['feat_change', 'vol_ratio', 'intraday_body', 'wick_upper', 'close_in_range']
ALL_FEATURES = DIR_FEATURE_COLS + MAG_FEATURE_COLS

def build_features(df):
    c, o, h, l = df['Close'], df['Open'], df['High'], df['Low']

    ema5  = c.ewm(span=5,  adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()

    # Features
    df['feat']           = (ema5 - ema20) / (ema20 + 1e-9) 
    df['feat_change']    = df['feat'].diff()
    df['price_vs_ema20'] = (c - ema20) / (ema20 + 1e-9)
    df['ret_1d']         = c.pct_change(1)
    df['ret_3d']         = c.pct_change(3)
    df['vol_ratio']      = df['std5'] / (df['std25'] + 1e-9)
    df['intraday_body']  = np.abs(c - o) / (o + 1e-9) # Absolute body size for magnitude
    df['wick_upper']     = (h - c) / (h - l + 1e-9)
    df['close_in_range'] = (c - l) / (h - l + 1e-9)
    df['prev_direction'] = np.sign(c.diff().shift(1)).astype(float)

    # TWO TARGETS
    returns = c.pct_change(1).shift(-1)
    # Target 1: Direction (+1 or -1)
    df['target_direction'] = np.where(returns >= 0, 1.0, -1.0)
    # Target 2: Magnitude (Absolute size of the return)
    df['target_magnitude'] = np.abs(returns)
    
    # Ground truth for evaluation
    df['next_actual'] = c.shift(-1)
    return df

# ------------------------------------------------------------------------------
# 3. Two-Stage Model Architecture
# ------------------------------------------------------------------------------

def build_direction_model():
    return Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis())])

def build_magnitude_model():
    # Notice we drop the polynomial features to prevent overfitting the volatility
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5))
    ])

def fit_two_stage(df_history):
    valid = df_history.dropna(subset=ALL_FEATURES + ['target_direction', 'target_magnitude'])
    
    # Train LDA on direction features
    X_dir = valid[DIR_FEATURE_COLS].astype(float).values
    y_dir = valid['target_direction'].astype(float).values
    lda_model = build_direction_model()
    lda_model.fit(X_dir, y_dir)

    # Train Ridge on magnitude features
    X_mag = valid[MAG_FEATURE_COLS].astype(float).values
    y_mag = valid['target_magnitude'].astype(float).values
    ridge_model = build_magnitude_model()
    ridge_model.fit(X_mag, y_mag)

    return {'direction': lda_model, 'magnitude': ridge_model}

def predict_two_stage(models, row):
    try:
        x_dir = row[DIR_FEATURE_COLS].astype(float).values.reshape(1, -1)
        x_mag = row[MAG_FEATURE_COLS].astype(float).values.reshape(1, -1)
    except (ValueError, TypeError):
        return np.nan
        
    if np.any(np.isnan(x_dir)) or np.any(np.isnan(x_mag)):
        return np.nan
        
    # Predict stages
    pred_sign = models['direction'].predict(x_dir)[0]
    pred_size = models['magnitude'].predict(x_mag)[0]
    
    # Mathematically clamp size so model doesn't predict negative volatility
    pred_size = max(0.0, float(pred_size))
    
    return float(pred_sign * pred_size)

# ------------------------------------------------------------------------------
# 4. Prediction Engines
# ------------------------------------------------------------------------------

def build_train_predictions(df_train, models):
    preds_close = []
    for i in range(len(df_train) - 1):
        current_close = float(df_train['Close'].iloc[i])
        pred_return = predict_two_stage(models, df_train.iloc[i])
        
        if np.isnan(pred_return):
            preds_close.append(np.nan)
        else:
            preds_close.append(current_close * (1.0 + pred_return))
            
    preds_close.append(np.nan)
    out = df_train.copy()
    out['predicted_next'] = preds_close
    return out

def build_walkforward_predictions(df_full, df_val, initial_models, update_every=7, lookback_days=1000):
    val_idx = df_val.index
    n_val   = len(val_idx)
    models  = initial_models
    preds_close = []

    print(f"\n  Walk-forward: {n_val} days  |  refit every {update_every} days (Rolling {lookback_days}d)")

    for i in range(n_val - 1):
        day = val_idx[i]

        if i > 0 and i % update_every == 0:
            history = df_full.loc[df_full.index < day].tail(lookback_days)
            n_valid = len(history.dropna(subset=ALL_FEATURES + ['target_direction']))
            
            if n_valid > 100:
                models = fit_two_stage(history)
                alpha = models['magnitude'].named_steps['ridge'].alpha_
                print(f"    Refit @ {day.date()}  history={len(history)}d  valid={n_valid}  ridge_alpha={alpha:.2f}")

        current_close = float(df_val['Close'].iloc[i])
        pred_return = predict_two_stage(models, df_val.iloc[i])
        
        if np.isnan(pred_return):
            preds_close.append(np.nan)
        else:
            preds_close.append(current_close * (1.0 + pred_return))

    preds_close.append(np.nan)
    out = df_val.copy()
    out['predicted_next'] = preds_close
    return out

# ------------------------------------------------------------------------------
# 5. Summaries & Output
# ------------------------------------------------------------------------------

def compute_summary(df):
    rows = []
    for year, g in df.groupby(df.index.year):
        v = g.dropna(subset=['predicted_next', 'next_actual'])
        if len(v) == 0: continue
        err     = (v['predicted_next'] - v['next_actual']).abs()
        err_pct = err / v['next_actual'] * 100

        pred_dir   = np.sign(v['predicted_next'].values - v['Close'].values)
        actual_dir = np.sign(v['next_actual'].values    - v['Close'].values)
        vm         = ~np.isnan(pred_dir) & ~np.isnan(actual_dir)
        dir_acc    = (pred_dir[vm] == actual_dir[vm]).mean() if vm.sum() > 0 else np.nan

        rows.append({
            'Year':             year,
            'Days':             len(v),
            'Dir Acc (%)':      round(float(dir_acc * 100) if not np.isnan(dir_acc) else 0, 2),
            'Avg Error (INR)':  round(float(err.mean()),     4),
            'Avg Error (%)':    round(float(err_pct.mean()), 4),
            '65th Pct Err (%)': round(float(err_pct.quantile(0.65)), 4),
            'Max Error (%)':    round(float(err_pct.max()),  4),
        })
    return pd.DataFrame(rows)

def print_summary(summary, df, label=''):
    print("\n" + "=" * 80)
    if label: print(f"  {label}")
    print(f"{'Year':<8} {'Days':>6} {'DirAcc%':>8} {'AvgErrINR':>11} {'AvgErr%':>9} {'65pct%':>8} {'MaxErr%':>9}")
    print("-" * 80)
    for _, r in summary.iterrows():
        print(f"{int(r['Year']):<8} {int(r['Days']):>6} {r['Dir Acc (%)']:>8.2f} "
              f"{r['Avg Error (INR)']:>11.4f} {r['Avg Error (%)']:>9.4f} "
              f"{r['65th Pct Err (%)']:>8.4f} {r['Max Error (%)']:>9.4f}")
    print("=" * 80)
    
    v  = df.dropna(subset=['predicted_next', 'next_actual'])
    pred_dir   = np.sign(v['predicted_next'].values - v['Close'].values)
    actual_dir = np.sign(v['next_actual'].values    - v['Close'].values)
    dir_acc    = (pred_dir == actual_dir).mean() * 100
    
    ep = (v['predicted_next'] - v['next_actual']).abs() / v['next_actual'] * 100
    print(f"\nOVERALL ({len(v):,} days)")
    print(f"  Directional Acc : {dir_acc:.2f}%")
    print(f"  Avg abs error   : {ep.mean():.4f}%")
    print(f"  Median error    : {ep.median():.4f}%")

def main(input_csv='USD_INR_Exchange.csv', output_folder='usd_inr_poly_plots'):
    print(f'\n{"="*68}')
    print('  USD/INR Two-Stage Predictor (LDA Direction + Ridge Magnitude)')
    print(f'{"="*68}')

    print(f'\nLoading {input_csv} ...')
    raw = pd.read_csv(input_csv)
    raw.columns = raw.columns.str.strip()
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values('Date').reset_index(drop=True).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close'])

    print('\n[1] GARCH(1,1) & Features...')
    full = raw[(raw.index.year >= 2003) & (raw.index.year <= 2023)].copy()
    full = classify_vol_regime(full)
    full = build_features(full)

    df_train = full[full.index.year <= 2020].copy()
    df_val   = full[full.index.year >= 2021].copy()

    print('\n[2] Fitting Initial Two-Stage Model (2003-2020) ...')
    models0 = fit_two_stage(df_train)
    
    print('\n[3] Generating Train Predictions ...')
    df_train = build_train_predictions(df_train, models0)

    print('\n[4] Walk-forward Validation (2021-2023) ...')
    df_val = build_walkforward_predictions(full, df_val, models0, update_every=7, lookback_days=1000)

    print('\n[5] Results -- TRAIN (2003-2020)')
    summary_train = compute_summary(df_train)
    print_summary(summary_train, df_train, label='TRAIN SET')

    print('\n[5b] Results -- WALK-FORWARD (2021-2023)')
    summary_val = compute_summary(df_val)
    print_summary(summary_val, df_val, label='WALK-FORWARD VALIDATION')

    print(f'\nDone.')

if __name__ == '__main__':
    csv_path   = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    out_folder = sys.argv[2] if len(sys.argv) > 2 else 'usd_inr_poly_plots'
    main(csv_path, out_folder)