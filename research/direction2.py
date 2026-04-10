import pandas as pd
import numpy as np
import sys
from scipy.optimize import differential_evolution
from scipy.stats import binom


# ============================================================================
# Technical indicators
# ============================================================================

def compute_indicators(df):
    c = df['Close']

    df['ema_5']  = c.ewm(span=5,  adjust=False).mean()
    df['ema_20'] = c.ewm(span=20, adjust=False).mean()

    # Feature: day-over-day change in EMA(5) - EMA(20) spread
    df['feat']      = df['ema_5'] - df['ema_20']
    df['feat_diff'] = df['feat'].diff()          # feat(t) - feat(t-1)

    # Target: direction of next-day close (close(t+1) - close(t))
    df['close_delta'] = df['Close'].diff().shift(-1)
    df['target']      = np.where(df['close_delta'] > 0, 1, -1)

    return df


# ============================================================================
# Objective: maximise accuracy subject to abstain <= 5%
# ============================================================================

def objective(params, feat_vals, target_vals, max_abstain_frac=0.05):
    gamma, beta = params

    if beta <= gamma:
        return 0.0   # invalid: return worst possible accuracy

    mask_up   = feat_vals >  beta
    mask_down = feat_vals <  gamma
    mask_abs  = (feat_vals >= gamma) & (feat_vals <= beta)

    n_total   = len(feat_vals)
    n_abstain = mask_abs.sum()

    # Soft penalty if abstain fraction exceeds 5%
    abstain_frac = n_abstain / n_total
    if abstain_frac > max_abstain_frac:
        penalty = (abstain_frac - max_abstain_frac) * 200
    else:
        penalty = 0.0

    n_counted = mask_up.sum() + mask_down.sum()
    if n_counted == 0:
        return 0.0

    correct = (
        (mask_up   & (target_vals ==  1)).sum() +
        (mask_down & (target_vals == -1)).sum()
    )

    accuracy = correct / n_counted
    return -(accuracy - penalty)   # minimise negative accuracy


# ============================================================================
# Learn beta & gamma via differential evolution
# ============================================================================

def learn_thresholds(df):
    valid     = df.dropna(subset=['feat_diff', 'close_delta']).copy()
    feat_vals = valid['feat_diff'].values
    tgt_vals  = valid['target'].values

    f_min, f_max = feat_vals.min(), feat_vals.max()
    f_range      = f_max - f_min

    bounds = [
        (f_min, f_min + f_range * 0.6),   # gamma
        (f_min + f_range * 0.4, f_max),   # beta
    ]

    result = differential_evolution(
        objective,
        bounds,
        args=(feat_vals, tgt_vals, 0.05),
        seed=42,
        maxiter=2000,
        popsize=20,
        tol=1e-9,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,
    )

    gamma, beta = result.x
    return gamma, beta


# ============================================================================
# Predict
# ============================================================================

def predict(feat_diff_series, beta, gamma):
    predicted = pd.Series(np.nan, index=feat_diff_series.index)
    predicted[feat_diff_series >  beta]  =  1
    predicted[feat_diff_series <  gamma] = -1
    return predicted


# ============================================================================
# Evaluate
# ============================================================================

def evaluate(df, beta, gamma):
    valid = df.dropna(subset=['feat_diff', 'close_delta']).copy()
    valid['predicted'] = predict(valid['feat_diff'], beta, gamma)

    total_days  = len(valid)
    abstained   = valid['predicted'].isna().sum()
    abstain_pct = abstained / total_days * 100
    counted     = valid.dropna(subset=['predicted'])
    n_counted   = len(counted)

    overall_correct = (counted['predicted'] == counted['target']).sum()
    overall_acc     = overall_correct / n_counted * 100

    # Binomial significance test
    p_value = binom.sf(overall_correct - 1, n_counted, 0.5)

    print("=" * 65)
    print(f"{'USD/INR — Optimised Threshold Model':^65}")
    print(f"{'Feature: Δ(EMA5 − EMA20)  i.e. feat(t) − feat(t−1)':^65}")
    print("=" * 65)
    print(f"  Beta  (upper threshold) : {beta:.6f}")
    print(f"  Gamma (lower threshold) : {gamma:.6f}")
    print(f"  Abstain zone            : [{gamma:.6f}, {beta:.6f}]")
    print(f"  Total days              : {total_days}")
    print(f"  Abstained days          : {abstained}  ({abstain_pct:.3f}%)")
    print(f"  Counted days            : {n_counted}")
    print("=" * 65)
    print(f"{'Year':<10} {'Days':>6} {'Counted':>8} {'Abstained':>10} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 65)

    overall_correct_check = 0
    overall_counted_check = 0

    for year, grp in valid.groupby(valid.index.year):
        grp_counted  = grp.dropna(subset=['predicted'])
        total_yr     = len(grp)
        counted_yr   = len(grp_counted)
        abstained_yr = total_yr - counted_yr
        correct_yr   = (grp_counted['predicted'] == grp_counted['target']).sum()
        acc_yr       = correct_yr / counted_yr * 100 if counted_yr > 0 else float('nan')
        overall_correct_check += correct_yr
        overall_counted_check += counted_yr
        print(f"{year:<10} {total_yr:>6} {counted_yr:>8} {abstained_yr:>10} {correct_yr:>8} {acc_yr:>9.2f}%")

    print("-" * 65)
    print(f"{'OVERALL':<10} {total_days:>6} {n_counted:>8} {abstained:>10} {overall_correct:>8} {overall_acc:>9.2f}%")
    print("=" * 65)

    # Baseline
    baseline_correct = (counted['target'] == 1).sum()
    baseline_acc     = baseline_correct / n_counted * 100
    print(f"\n  Baseline (always predict UP) : {baseline_acc:.2f}%")
    print(f"  Model accuracy               : {overall_acc:.2f}%")
    print(f"  Lift over baseline           : {overall_acc - baseline_acc:+.2f}%")
    print(f"\n  Binomial p-value             : {p_value:.6f}")
    if p_value < 0.01:
        print(f"  Significance                 : *** highly significant (p < 0.01)")
    elif p_value < 0.05:
        print(f"  Significance                 : ** significant (p < 0.05)")
    elif p_value < 0.10:
        print(f"  Significance                 : * marginal (p < 0.10)")
    else:
        print(f"  Significance                 : not significant (p >= 0.10)")
    print("=" * 65)


# ============================================================================
# Main
# ============================================================================

def main(input_csv):
    print(f'\nLoading {input_csv} ...\n')
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.set_index('Date')

    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])

    # Filter to 2003-2019
    df = df[(df.index.year >= 2003) & (df.index.year <= 2019)]

    print('Computing indicators ...')
    df = compute_indicators(df)

    print('Optimising beta & gamma via differential evolution ...')
    print('(This may take a few seconds)\n')
    gamma, beta = learn_thresholds(df)

    evaluate(df, beta, gamma)


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    main(csv_path)