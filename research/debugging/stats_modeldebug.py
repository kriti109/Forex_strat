"""
Self-contained diagnostic CSV exporter for USD/INR predictor.
No imports from predict_usd_inr.py — everything is inlined.

Run:
    python debug_export.py [path_to_csv] [output_csv]
    python debug_export.py USD_INR_Exchange.csv debug_features.csv
"""

import pandas as pd
import numpy as np
import sys
from scipy.optimize import differential_evolution


# ─────────────────────────────────────────────────────────────────────────────
# GARCH(1,1)
# ─────────────────────────────────────────────────────────────────────────────

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

    omega, alpha, beta_g = best_params
    h = _garch_variance(r, omega, alpha, beta_g)
    print(f"  GARCH params → omega={omega:.2e}  alpha={alpha:.4f}  "
          f"beta={beta_g:.4f}  persistence={alpha+beta_g:.4f}")
    return np.sqrt(np.maximum(h, 1e-12)), omega, alpha, beta_g


# ─────────────────────────────────────────────────────────────────────────────
# Regime
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
# EMA feature
# ─────────────────────────────────────────────────────────────────────────────

def compute_ema_feature(df):
    c = df['Close']
    df['ema_5']       = c.ewm(span=5,  adjust=False).mean()
    df['ema_20']      = c.ewm(span=20, adjust=False).mean()
    df['feat']        = df['ema_5'] - df['ema_20']
    df['close_delta'] = c.diff().shift(-1)     # FUTURE — label only
    df['target']      = np.where(df['close_delta'] > 0, 1, -1)  # FUTURE — label only
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Direction thresholds
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"  Learned thresholds → gamma={gamma:.6f}  beta={beta:.6f}")
    return gamma, beta


# ─────────────────────────────────────────────────────────────────────────────
# Build predictions
# ─────────────────────────────────────────────────────────────────────────────

def build_predictions(df, beta, gamma):
    close  = df['Close'].values
    gvol   = df['garch_vol'].values
    feat   = df['feat'].values
    regime = df['vol_regime'].values
    n      = len(df)

    pred      = np.full(n, np.nan)
    direction = np.full(n, 0)

    for i in range(n - 1):
        c = close[i]
        g = gvol[i]
        f = feat[i]

        if   f > beta:   d =  1
        elif f < gamma:  d = -1
        else:            d =  0

        direction[i] = d

        if   regime[i] == 'low':    pred[i] = c
        elif regime[i] == 'high':   pred[i] = c + d * g * c
        else:                       pred[i] = c + d * 0.5 * g * c

    df = df.copy()
    df['direction_signal'] = direction
    df['predicted_next']   = pred
    df['next_actual']      = df['Close'].shift(-1)   # FUTURE — eval only
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_debug_csv(input_csv='USD_INR_Exchange.csv', output_csv='debug_features.csv'):

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f'\nLoading {input_csv} ...')
    raw = pd.read_csv(input_csv)
    raw.columns = raw.columns.str.strip()
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.sort_values('Date').reset_index(drop=True).set_index('Date')
    for col in ['Open', 'High', 'Low', 'Close']:
        raw[col] = pd.to_numeric(raw[col], errors='coerce')
    raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close'])

    df_train = raw[(raw.index.year >= 2003) & (raw.index.year <= 2021)].copy()
    df_val   = raw[(raw.index.year >= 2022) & (raw.index.year <= 2023)].copy()
    print(f'  Train : {len(df_train):,} rows')
    print(f'  Val   : {len(df_val):,} rows')

    # ── GARCH — train ─────────────────────────────────────────────────────────
    print('\n[1] GARCH on train ...')
    train_ret = df_train['Close'].pct_change()
    df_train['return'] = train_ret
    garch_vol_train, g_omega, g_alpha, g_beta = fit_garch11(train_ret)
    df_train['garch_vol'] = garch_vol_train

    # ── GARCH — full series for val continuity ────────────────────────────────
    print('[1b] GARCH on full series (for val) ...')
    full = raw[(raw.index.year >= 2003) & (raw.index.year <= 2023)].copy()
    full['return'] = full['Close'].pct_change()
    garch_vol_full, _, _, _ = fit_garch11(full['return'])
    full['garch_vol'] = garch_vol_full
    df_val['return']    = full.loc[df_val.index, 'return']
    df_val['garch_vol'] = full.loc[df_val.index, 'garch_vol']

    # ── Regime ────────────────────────────────────────────────────────────────
    print('[2] Regime ...')
    full = classify_vol_regime(full)
    for col in ['std5', 'std14', 'std25', 'vol_regime']:
        df_train[col] = full.loc[df_train.index, col]
        df_val[col]   = full.loc[df_val.index,   col]

    # ── EMA feature ───────────────────────────────────────────────────────────
    print('[3] EMA feature ...')
    full = compute_ema_feature(full)
    for col in ['ema_5', 'ema_20', 'feat', 'close_delta', 'target']:
        df_train[col] = full.loc[df_train.index, col]
        df_val[col]   = full.loc[df_val.index,   col]

    # ── Thresholds (train only) ───────────────────────────────────────────────
    print('[4] Learning thresholds on train only ...')
    gamma, beta = learn_thresholds(df_train)
    print(f'  *** FROZEN: gamma={gamma:.6f}  beta={beta:.6f} ***')

    # ── Predictions ───────────────────────────────────────────────────────────
    print('[5] Building predictions ...')
    df_train = build_predictions(df_train, beta, gamma)
    df_val   = build_predictions(df_val,   beta, gamma)

    # ── Errors ────────────────────────────────────────────────────────────────
    for df in [df_train, df_val]:
        df['error_inr'] = df['predicted_next'] - df['next_actual']
        df['error_pct'] = df['error_inr'] / df['next_actual'] * 100

    # ── Tag dataset ───────────────────────────────────────────────────────────
    df_train['dataset'] = 'train'
    df_val['dataset']   = 'validation'

    # ── Store frozen params as columns (easy to verify) ───────────────────────
    for df in [df_train, df_val]:
        df['garch_omega'] = g_omega
        df['garch_alpha'] = g_alpha
        df['garch_beta']  = g_beta
        df['ema_gamma']   = gamma
        df['ema_beta']    = beta

    # ── Combine & order columns ───────────────────────────────────────────────
    combined = pd.concat([df_train, df_val])

    col_order = [
        # ── metadata
        'dataset',
        # ── raw OHLC
        'Open', 'High', 'Low', 'Close',
        # ── return[t] = close[t]/close[t-1] - 1  (uses only past)
        'return',
        # ── GARCH frozen params (constants)
        'garch_omega', 'garch_alpha', 'garch_beta',
        # ── garch_vol[t] computed from return[0..t-1] only
        'garch_vol',
        # ── rolling std[t] uses close[t-window+1 .. t] only
        'std5', 'std14', 'std25',
        # ── regime assigned from std values at t
        'vol_regime',
        # ── EMA[t] uses close[0..t] only
        'ema_5', 'ema_20',
        # ── feat[t] = ema5[t] - ema20[t]
        'feat',
        # ── frozen direction thresholds (constants)
        'ema_gamma', 'ema_beta',
        # ── direction derived from feat[t] + frozen thresholds  (no future)
        'direction_signal',
        # ── prediction made AT t FOR t+1
        'predicted_next',
        # ══ FUTURE COLUMNS BELOW — for evaluation only ════════════════════
        'close_delta',   # close[t+1] - close[t]  ← FUTURE
        'target',        # sign(close_delta)        ← FUTURE
        'next_actual',   # close[t+1]               ← FUTURE
        'error_inr',     # predicted - actual        ← FUTURE
        'error_pct',     # error %                   ← FUTURE
    ]
    combined = combined[[c for c in col_order if c in combined.columns]]
    combined.index.name = 'Date'

    # ── Header-note row (row 0) to flag future columns ────────────────────────
    future_cols = {'close_delta', 'target', 'next_actual', 'error_inr', 'error_pct'}
    note = {c: ('*** FUTURE / EVAL ONLY ***' if c in future_cols else 'present-or-past only')
            for c in combined.columns}
    note_df = pd.DataFrame([note], index=pd.to_datetime(['1900-01-01']))
    note_df.index.name = 'Date'

    final = pd.concat([note_df, combined])
    final.to_csv(output_csv)
    print(f'\n  Saved → {output_csv}  ({len(combined):,} data rows + 1 note row)')

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print('\n── Sanity Checks ──')

    # 1. GARCH should correlate more with lagged return than current return
    corr_t   = combined['garch_vol'].corr(combined['return'].abs())
    corr_tm1 = combined['garch_vol'].corr(combined['return'].abs().shift(1))
    print(f'  garch_vol vs |return[t]|   = {corr_t:.4f}  '
          f'{"⚠ HIGH — possible leakage!" if corr_t > corr_tm1 + 0.05 else "✓ ok"}')
    print(f'  garch_vol vs |return[t-1]| = {corr_tm1:.4f}  '
          f'(should be >= above)')

    # 2. predicted_next should never equal next_actual except on low-vol days
    exact    = (combined['predicted_next'].round(6) == combined['next_actual'].round(6)).sum()
    low_days = (combined['vol_regime'] == 'low').sum()
    print(f'  predicted == actual  : {exact} days  '
          f'(low-vol days = {low_days}, diff = {exact - low_days})')

    # 3. close_delta should be exactly diff().shift(-1)
    recon   = combined['Close'].diff().shift(-1).round(6)
    mismatch = (combined['close_delta'].round(6) != recon).sum()
    print(f'  close_delta mismatch : {mismatch} rows  '
          f'(expect 0, or 1 for the last NaN row)')

    # 4. feat should be ema5 - ema20
    feat_recon = (combined['ema_5'] - combined['ema_20']).round(6)
    feat_mismatch = (combined['feat'].round(6) != feat_recon).sum()
    print(f'  feat mismatch        : {feat_mismatch} rows  (expect 0)')

    # 5. direction consistency check
    d = combined['direction_signal']
    f = combined['feat']
    g = combined['ema_gamma'].iloc[0]
    b = combined['ema_beta'].iloc[0]
    wrong = (
        ((f > b)  & (d != 1))  |
        ((f < g)  & (d != -1)) |
        ((f >= g) & (f <= b) & (d != 0))
    ).sum()
    print(f'  direction_signal inconsistency: {wrong} rows  (expect 0)')

    return combined


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    out_csv  = sys.argv[2] if len(sys.argv) > 2 else 'debug_features.csv'
    export_debug_csv(csv_path, out_csv)