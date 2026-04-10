import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Technical indicators
# ============================================================================

def compute_indicators(df):
    c = df['Close']

    df['sma_5']  = c.ewm(span=6, adjust=False).mean()
    df['ema_40'] = c.ewm(span=15, adjust=False).mean()

    # Feature: SMA(5) - EMA(40) at time t
    df['feat'] = df['sma_5'] - df['ema_40']

    # Target: direction of next-day close (t+1 - t), only for evaluation
    df['close_delta'] = df['Close'].diff().shift(-1)
    df['target']      = np.where(df['close_delta'] > 0, 1, -1)

    return df


# ============================================================================
# Learn beta & gamma via logistic regression
# ============================================================================

def learn_thresholds(df):
    """
    Fit a logistic regression on feat -> target.
    Decision boundary: feat where P(+1|feat) = 0.5, i.e. b0 + b1*feat_scaled = 0.
    Beta and gamma are placed symmetrically around this boundary such that
    the abstain zone [gamma, beta] covers at most 0.5% of trading days.
    """
    valid = df.dropna(subset=['feat', 'close_delta']).copy()
    X = valid[['feat']].values
    y = valid['target'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    # Decision boundary in original feat space
    b0  = model.intercept_[0]
    b1  = model.coef_[0][0]
    mu  = scaler.mean_[0]
    std = scaler.scale_[0]

    feat_boundary = mu + std * (-b0 / b1)

    # Binary search for half-width w around boundary so abstained <= 0.5%
    feat_vals   = valid['feat'].values
    n_total     = len(feat_vals)
    max_abstain = int(np.floor(0.05 * n_total))

    lo, hi = 0.0, (feat_vals.max() - feat_vals.min()) / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        abstained = ((feat_vals >= feat_boundary - mid) &
                     (feat_vals <= feat_boundary + mid)).sum()
        if abstained <= max_abstain:
            lo = mid
        else:
            hi = mid

    w     = lo
    gamma = feat_boundary - w
    beta  = feat_boundary + w

    return model, scaler, beta, gamma, feat_boundary


# ============================================================================
# Predict with abstain zone
# ============================================================================

def predict(feat_series, beta, gamma):
    predicted = pd.Series(np.nan, index=feat_series.index)
    predicted[feat_series >  beta]  = 1
    predicted[feat_series <  gamma] = -1
    # between gamma and beta -> abstain (remains NaN)
    return predicted


# ============================================================================
# Evaluate
# ============================================================================

def evaluate(df, beta, gamma):
    valid = df.dropna(subset=['feat', 'close_delta']).copy()
    valid['predicted'] = predict(valid['feat'], beta, gamma)

    total_days  = len(valid)
    abstained   = valid['predicted'].isna().sum()
    abstain_pct = abstained / total_days * 100
    counted     = valid.dropna(subset=['predicted'])

    print("=" * 65)
    print(f"{'USD/INR — Logistic Regression Threshold Model':^65}")
    print(f"{'Signal: feat = SMA(5) − EMA(40)':^65}")
    print("=" * 65)
    print(f"  Learned boundary : {(beta+gamma)/2:.4f}")
    print(f"  Beta  (upper)    : {beta:.4f}")
    print(f"  Gamma (lower)    : {gamma:.4f}")
    print(f"  Abstain zone     : [{gamma:.4f}, {beta:.4f}]")
    print(f"  Total days       : {total_days}")
    print(f"  Abstained days   : {abstained}  ({abstain_pct:.3f}%)")
    print(f"  Counted days     : {len(counted)}")
    print("=" * 65)
    print(f"{'Year':<10} {'Days':>6} {'Counted':>8} {'Abstained':>10} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 65)

    overall_correct = 0
    overall_counted = 0

    for year, grp in valid.groupby(valid.index.year):
        grp_counted  = grp.dropna(subset=['predicted'])
        total_yr     = len(grp)
        counted_yr   = len(grp_counted)
        abstained_yr = total_yr - counted_yr
        correct_yr   = (grp_counted['predicted'] == grp_counted['target']).sum()
        acc_yr       = correct_yr / counted_yr * 100 if counted_yr > 0 else float('nan')
        overall_correct += correct_yr
        overall_counted += counted_yr
        print(f"{year:<10} {total_yr:>6} {counted_yr:>8} {abstained_yr:>10} {correct_yr:>8} {acc_yr:>9.2f}%")

    print("-" * 65)
    overall_acc = overall_correct / overall_counted * 100
    print(f"{'OVERALL':<10} {total_days:>6} {overall_counted:>8} {abstained:>10} {overall_correct:>8} {overall_acc:>9.2f}%")
    print("=" * 65)

    # Baseline: always predict UP
    baseline_correct = (counted['target'] == 1).sum()
    baseline_acc     = baseline_correct / len(counted) * 100
    print(f"\n  Baseline (always predict UP) : {baseline_acc:.2f}%")
    print(f"  Model accuracy               : {overall_acc:.2f}%")
    print(f"  Lift over baseline           : {overall_acc - baseline_acc:+.2f}%")
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

    print('Learning beta & gamma via logistic regression ...\n')
    model, scaler, beta, gamma, boundary = learn_thresholds(df)

    evaluate(df, beta, gamma)


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    main(csv_path)