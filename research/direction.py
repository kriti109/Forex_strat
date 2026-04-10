import pandas as pd
import numpy as np
import sys


# ============================================================================
# Technical indicators
# ============================================================================

def compute_indicators(df):
    c = df['Close']

    df['sma_5']  = c.rolling(5).mean()
    df['ema_40'] = c.ewm(span=40, adjust=False).mean()

    # Feature: SMA(5) - EMA(40) at time t
    df['feat'] = df['sma_5'] - df['ema_40']

    # Prediction signal: change in feat from t-1 to t (no lookahead)
    df['feat_delta'] = df['feat'].diff()          # feat(t) - feat(t-1)
    df['predicted']  = np.where(df['feat_delta'] > 0, 1, -1)

    # Target: direction of next-day close (uses t+1, only for evaluation)
    df['close_delta'] = df['Close'].diff().shift(-1)   # Close(t+1) - Close(t)
    df['target']      = np.where(df['close_delta'] > 0, 1, -1)

    return df


# ============================================================================
# Evaluate
# ============================================================================

def evaluate(df):
    # Drop rows where either signal is undefined (first row, last row)
    valid = df.dropna(subset=['feat_delta', 'close_delta'])

    print("=" * 55)
    print(f"{'USD/INR — Directional Prediction Accuracy':^55}")
    print(f"{'Signal: Δ(SMA5−EMA40)  →  Next-day Close direction':^55}")
    print("=" * 55)
    print(f"{'Year':<10} {'Days':>6} {'Correct':>8} {'Accuracy':>10}")
    print("-" * 55)

    overall_correct = 0
    overall_total   = 0

    for year, grp in valid.groupby(valid.index.year):
        total   = len(grp)
        correct = (grp['predicted'] == grp['target']).sum()
        acc     = correct / total * 100
        overall_correct += correct
        overall_total   += total
        print(f"{year:<10} {total:>6} {correct:>8} {acc:>9.2f}%")

    print("-" * 55)
    overall_acc = overall_correct / overall_total * 100
    print(f"{'OVERALL':<10} {overall_total:>6} {overall_correct:>8} {overall_acc:>9.2f}%")
    print("=" * 55)

    # Baseline: always predict +1
    baseline_correct = (valid['target'] == 1).sum()
    baseline_acc     = baseline_correct / overall_total * 100
    print(f"\nBaseline (always predict UP): {baseline_acc:.2f}%")
    print(f"Model lift over baseline    : {overall_acc - baseline_acc:+.2f}%")
    print("=" * 55)


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

    df = compute_indicators(df)
    evaluate(df)


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'USD_INR_Exchange.csv'
    main(csv_path)