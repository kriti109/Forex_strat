slides 4 to 9 :
we started from the basics. we had 4 initial features on any particular day - open high low close - we wanted to derive 
and tranform new features from those initial features by studying plots *insert plots*. then we would put those features 
in some regression model to predict returns. We started building constraints using those features on our own which is the 
stats_model we are presenting. halfway through it we realised we werent using any machine learning model. so we went there and explored 
regression and classification models such as regression. knn. decision trees. 

main ideation - our own stat model
Logic:
  - Fit GARCH(1,1) -> conditional volatility (sigma_t, in return units)
  - Rolling std of last 5, 14, 25 days (price std, for regime detection)
  - Direction from optimised EMA5-EMA20 threshold model (beta/gamma)

  Regime classification (using rolling std):
    HIGH vol  : std5 > std14 > std25
    LOW  vol  : std5 < std14 < std25
    MEDIUM    : everything else

  Prediction for day t+1:
    HIGH   -> close_t + direction * garch_vol_t * close_t
    LOW    -> close_t  (same price, no change)
    MEDIUM -> close_t + direction * 0.5 * garch_vol_t * close_t

  Walk-forward gamma/beta update:
    - Initial gamma/beta learned on 2003-2020 (full history)
    - From 2021 onwards, predict day by day
    - Every 7 trading days, re-run optimiser on ALL data seen so far
      (2003-2020 + days elapsed in 2021-2023) -> updated gamma/beta
    - On day T, only data up to T-1 is used. Zero lookahead.


    now we used this as the basis for the classic ml models as well. 
    the only different thing we did was use classification models - decision trees knn and polynomial reg/Lda
    to get direction and then did the same thing with the prediction of closing price. 


decision trees - our final prediction model (best results)
Direction classifier replaces gamma/beta EMA threshold model.
Everything else (GARCH vol, regime, price formula) is identical.

Features used for direction:
  1.  feat            = EMA5 - EMA20
  2.  feat_change     = feat - feat.shift(1)
  3.  price_vs_ema20  = (close - ema20) / ema20
  4.  ret_1d          = close.pct_change(1)
  5.  ret_3d          = close.pct_change(3)
  6.  vol_ratio       = std5 / std25
  7.  intraday_body   = (close - open) / open
  8.  wick_upper      = (high - close) / (high - low + 1e-9)
  9.  close_in_range  = (close - low) / (high - low + 1e-9)
  10. prev_direction  = sign(close.diff().shift(1))

Train : 2003-2020  (tree fitted once, fixed)
Test  : 2021-2023  (zero lookahead, out-of-sample)

knn - simple yet effective 

KNN classifier replaces gamma/beta EMA threshold model for direction.
Everything else (GARCH vol, regime, price formula) is identical.

Features used for direction (StandardScaler applied before KNN):
  1.  feat            = EMA5 - EMA20
  2.  feat_change     = feat - feat.shift(1)
  3.  price_vs_ema20  = (close - ema20) / ema20
  4.  ret_1d          = close.pct_change(1)
  5.  ret_3d          = close.pct_change(3)
  6.  vol_ratio       = std5 / std25
  7.  intraday_body   = (close - open) / open
  8.  wick_upper      = (high - close) / (high - low + 1e-9)
  9.  close_in_range  = (close - low) / (high - low + 1e-9)
  10. prev_direction  = sign(close.diff().shift(1))

Scaler  : StandardScaler fitted on train only, applied to both train & test
k tuning: 5-fold CV over k = 3,5,7,9,11,15,21,31 (odd only, avoids ties)

Train : 2003-2020  (KNN fitted once, fixed)
Test  : 2021-2023  (zero lookahead, out-of-sample)


polynomialreg/lda -
Two-Stage Model (LDA + Ridge CV) replaces the single-target regression model. 
LDA predicts the market direction (+1 / -1), while Ridge CV predicts the absolute magnitude of the return. Final return = Direction * Magnitude.
Everything else (GARCH vol, regime, price formula) is identical.

Features used for Direction (StandardScaler applied before LDA):
  1.  feat            = (EMA5 - EMA20) / EMA20
  2.  price_vs_ema20  = (close - ema20) / ema20
  3.  ret_1d          = close.pct_change(1)
  4.  ret_3d          = close.pct_change(3)
  5.  prev_direction  = sign(close.diff().shift(1))

Features used for Magnitude (StandardScaler applied before Ridge):
  6.  feat_change     = feat - feat.shift(1)
  7.  vol_ratio       = std5 / std25
  8.  intraday_body   = abs(close - open) / open
  9.  wick_upper      = (high - close) / (high - low + 1e-9)
  10. close_in_range  = (close - low) / (high - low + 1e-9)

Scaler  : StandardScaler fitted dynamically on the rolling window, applied to both pipelines.
Tuning  : LDA uses default singular value decomposition. 
          Ridge uses 5-fold CV over alphas = 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0.

Train : 2003-2020  (Initial fit)
Test  : 2021-2023  (Walk-forward: refit every 7 days on 1000-day rolling window, zero lookahead)


now while we are explaining the models we were thinking of giving two slides to our own stat model and 3 plots for decision trees, KNN and Polyregression/LDA 
and we'll put all the graphs in 1 slide so leave that empty 

