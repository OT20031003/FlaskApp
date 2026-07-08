# Direction Validation Report: ^N225

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | ^N225 |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-07-07 09:45:18 JST |
| Last Date | 2026-07-07 |
| Last Close | 69782.2578 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.1668 |
| Probability Down | 0.8332 |
| Decision Threshold | 0.0287 |
| Model Valid | False |
| Validation Method | purged_rolling_walk_forward |
| Target Return Threshold | 0.0050 |
| Buy Probability Threshold | 0.6000 |
| Sell Probability Threshold | 0.4000 |

## Rolling Configuration
| Item | Value |
| --- | --- |
| train_size | 504 |
| test_size | 63 |
| step_size | 63 |
| purge_size | 10 |
| calibration_ratio | 0.2500 |
| threshold_metric | balanced_accuracy |
| ridge_stack_alpha | 1.0000 |
| ridge_stack_train_window | 252 |
| ridge_stack_min_train_samples | 120 |

## Validation Thresholds
| Item | Value |
| --- | --- |
| min_rolling_folds | 3 |
| min_balanced_accuracy_mean | 0.5200 |
| min_roc_auc_mean | 0.5200 |
| min_beats_baseline_fold_ratio | 0.5000 |

## Aggregate Validation Metrics
| Metric | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| fold_count | 28 | N/A | N/A | N/A |
| total_fold_count | 28 | N/A | N/A | N/A |
| beats_baseline_fold_ratio | 0.2500 | N/A | N/A | N/A |
| confusion_matrix_sum | [[514, 320], [512, 418]] | N/A | N/A | N/A |
| accuracy | 0.5283 | 0.1183 | 0.2698 | 0.7460 |
| balanced_accuracy | 0.5520 | 0.0866 | 0.3995 | 0.7485 |
| precision | 0.6035 | 0.2821 | 0.0000 | 1.0000 |
| recall | 0.4678 | 0.3214 | 0.0000 | 1.0000 |
| f1 | 0.4401 | 0.2350 | 0.0000 | 0.8000 |
| roc_auc | 0.6180 | 0.1313 | 0.3768 | 0.8128 |
| log_loss | 1.1131 | 0.3509 | 0.5604 | 1.9984 |
| baseline_accuracy | 0.6043 | 0.0740 | 0.5079 | 0.7460 |
| decision_threshold | 0.5200 | 0.3148 | 0.0137 | 0.9900 |
| calibration_score | 0.5992 | 0.0499 | 0.5076 | 0.7068 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 410 | 978.9634 | 0.1883 |
| 2 | ma_gap_60 | 102 | 538.1659 | 0.1035 |
| 3 | volatility_60 | 232 | 531.7476 | 0.1023 |
| 4 | volatility_20 | 197 | 408.6346 | 0.0786 |
| 5 | ma_gap_120 | 138 | 372.8761 | 0.0717 |
| 6 | close_to_ema_12 | 146 | 271.6407 | 0.0522 |
| 7 | macd_hist_to_close | 151 | 260.0970 | 0.0500 |
| 8 | ema_12_to_ema_26 | 157 | 229.2517 | 0.0441 |
| 9 | ma_gap_5 | 107 | 220.0938 | 0.0423 |
| 10 | return_5 | 132 | 184.8482 | 0.0355 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2019-02-28 | 2019-06-07 | 0.5079 | 0.4857 | 0.4663 | 0.5556 | 0.8473 | ok |
| 2 | ok | 2019-06-10 | 2019-09-06 | 0.4762 | 0.4753 | 0.4254 | 0.5079 | 0.5334 | ok |
| 3 | ok | 2019-09-09 | 2019-12-13 | 0.4444 | 0.5513 | 0.6282 | 0.6190 | 0.4897 | ok |
| 4 | ok | 2019-12-16 | 2020-03-23 | 0.4127 | 0.4645 | 0.5576 | 0.6508 | 0.4698 | ok |
| 5 | ok | 2020-03-24 | 2020-06-24 | 0.6984 | 0.5919 | 0.7477 | 0.6825 | 0.0137 | ok |
| 6 | ok | 2020-06-25 | 2020-09-30 | 0.4444 | 0.5139 | 0.5947 | 0.5714 | 0.9098 | ok |
| 7 | ok | 2020-10-02 | 2021-01-04 | 0.4603 | 0.4722 | 0.4802 | 0.7143 | 0.2677 | ok |
| 8 | ok | 2021-01-05 | 2021-04-06 | 0.4921 | 0.5294 | 0.6805 | 0.5397 | 0.7136 | ok |
| 9 | ok | 2021-04-07 | 2021-07-08 | 0.2698 | 0.5000 | 0.4501 | 0.7302 | 0.0306 | ok |
| 10 | ok | 2021-07-09 | 2021-10-12 | 0.5873 | 0.5568 | 0.5193 | 0.5397 | 0.9265 | ok |
| 11 | ok | 2021-10-13 | 2022-01-14 | 0.5714 | 0.5695 | 0.6021 | 0.6032 | 0.9657 | ok |
| 12 | ok | 2022-01-17 | 2022-04-18 | 0.5079 | 0.6262 | 0.7605 | 0.6825 | 0.4780 | ok |
| 13 | ok | 2022-04-19 | 2022-07-21 | 0.6984 | 0.7181 | 0.8043 | 0.5397 | 0.9384 | ok |
| 14 | ok | 2022-07-22 | 2022-10-24 | 0.5397 | 0.5304 | 0.4868 | 0.5397 | 0.6916 | ok |
| 15 | ok | 2022-10-25 | 2023-01-26 | 0.6667 | 0.6653 | 0.6583 | 0.5079 | 0.2964 | ok |
| 16 | ok | 2023-01-27 | 2023-04-27 | 0.6508 | 0.6790 | 0.7439 | 0.6508 | 0.2169 | ok |
| 17 | ok | 2023-04-28 | 2023-07-31 | 0.6984 | 0.6346 | 0.7786 | 0.5873 | 0.0656 | ok |
| 18 | ok | 2023-08-01 | 2023-10-31 | 0.4444 | 0.4107 | 0.5245 | 0.5556 | 0.6992 | ok |
| 19 | ok | 2023-11-01 | 2024-02-05 | 0.3175 | 0.5000 | 0.8128 | 0.6825 | 0.9900 | ok |
| 20 | ok | 2024-02-06 | 2024-05-10 | 0.5238 | 0.5061 | 0.7162 | 0.5238 | 0.0698 | ok |
| 21 | ok | 2024-05-13 | 2024-08-08 | 0.5397 | 0.5281 | 0.4990 | 0.5873 | 0.4621 | ok |
| 22 | ok | 2024-08-09 | 2024-11-12 | 0.7460 | 0.7485 | 0.7117 | 0.5079 | 0.2846 | ok |
| 23 | ok | 2024-11-13 | 2025-02-17 | 0.6190 | 0.5000 | 0.7607 | 0.6190 | 0.9184 | ok |
| 24 | ok | 2025-02-18 | 2025-05-22 | 0.6349 | 0.6321 | 0.6356 | 0.5079 | 0.6357 | ok |
| 25 | ok | 2025-05-23 | 2025-08-21 | 0.4286 | 0.5814 | 0.7174 | 0.6825 | 0.8306 | ok |
| 26 | ok | 2025-08-22 | 2025-11-25 | 0.5079 | 0.4641 | 0.4176 | 0.7460 | 0.3465 | ok |
| 27 | ok | 2025-11-26 | 2026-03-02 | 0.5556 | 0.6216 | 0.7484 | 0.5873 | 0.1320 | ok |
| 28 | ok | 2026-03-03 | 2026-06-04 | 0.3492 | 0.3995 | 0.3768 | 0.6984 | 0.3374 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-01-04 |
| Train End | 2019-02-12 |
| Test Start | 2019-02-28 |
| Test End | 2019-06-07 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.4857 |
| Precision | 0.4211 |
| Recall | 0.2857 |
| F1 | 0.3404 |
| ROC-AUC | 0.4663 |
| Log Loss | 1.4335 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.8473 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6257 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[24, 11], [20, 8]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0031 |
| p25 | 0.0769 |
| median | 0.5069 |
| p75 | 0.8770 |
| max | 0.9777 |
| mean | 0.5002 |
| std | 0.3815 |
| count_ge_threshold | 19 |
| count_lt_threshold | 44 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 337 | 1085.6319 | 0.2026 |
| 2 | ma_gap_120 | 399 | 869.4019 | 0.1622 |
| 3 | volatility_20 | 295 | 504.4260 | 0.0941 |
| 4 | return_10 | 220 | 306.2513 | 0.0572 |
| 5 | macd_hist_to_close | 250 | 290.3097 | 0.0542 |
| 6 | ema_12_to_ema_26 | 90 | 253.9389 | 0.0474 |
| 7 | return_60 | 92 | 177.3457 | 0.0331 |
| 8 | macd_signal_to_close | 71 | 170.9942 | 0.0319 |
| 9 | return_5 | 83 | 165.5953 | 0.0309 |
| 10 | ridge_pred_future_log_return | 143 | 157.8933 | 0.0295 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-04-05 |
| Train End | 2019-05-23 |
| Test Start | 2019-06-10 |
| Test End | 2019-09-06 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4762 |
| Balanced Accuracy | 0.4753 |
| Precision | 0.4857 |
| Recall | 0.5312 |
| F1 | 0.5075 |
| ROC-AUC | 0.4254 |
| Log Loss | 0.9663 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.5334 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6111 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 18], [15, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1424 |
| p25 | 0.3467 |
| median | 0.5690 |
| p75 | 0.8152 |
| max | 0.9327 |
| mean | 0.5753 |
| std | 0.2535 |
| count_ge_threshold | 35 |
| count_lt_threshold | 28 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 254 | 973.3340 | 0.1875 |
| 2 | ma_gap_120 | 209 | 591.2474 | 0.1139 |
| 3 | macd_hist_to_close | 143 | 388.2179 | 0.0748 |
| 4 | ridge_pred_future_log_return | 151 | 308.6019 | 0.0594 |
| 5 | volatility_20 | 206 | 296.4267 | 0.0571 |
| 6 | ridge_pred_margin_to_threshold | 111 | 282.9437 | 0.0545 |
| 7 | return_60 | 189 | 236.1482 | 0.0455 |
| 8 | macd_signal_to_close | 115 | 200.7953 | 0.0387 |
| 9 | return_20 | 130 | 172.3796 | 0.0332 |
| 10 | close_to_high | 123 | 150.0515 | 0.0289 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-07-12 |
| Train End | 2019-08-23 |
| Test Start | 2019-09-09 |
| Test End | 2019-12-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.5513 |
| Precision | 1.0000 |
| Recall | 0.1026 |
| F1 | 0.1860 |
| ROC-AUC | 0.6282 |
| Log Loss | 1.5679 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.4897 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5446 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[24, 0], [35, 4]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0032 |
| p25 | 0.0219 |
| median | 0.0589 |
| p75 | 0.1987 |
| max | 0.9089 |
| mean | 0.1507 |
| std | 0.2036 |
| count_ge_threshold | 4 |
| count_lt_threshold | 59 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 325 | 1444.3950 | 0.2727 |
| 2 | volatility_20 | 305 | 886.2912 | 0.1673 |
| 3 | ma_gap_120 | 203 | 535.8908 | 0.1012 |
| 4 | return_60 | 201 | 308.1231 | 0.0582 |
| 5 | return_20 | 209 | 287.3939 | 0.0543 |
| 6 | return_5 | 103 | 275.1252 | 0.0519 |
| 7 | macd_signal_to_close | 119 | 205.5402 | 0.0388 |
| 8 | volume_change_5 | 146 | 193.0096 | 0.0364 |
| 9 | return_2 | 103 | 98.3357 | 0.0186 |
| 10 | rsi_14 | 29 | 85.3878 | 0.0161 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-10-13 |
| Train End | 2019-11-29 |
| Test Start | 2019-12-16 |
| Test End | 2020-03-23 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4127 |
| Balanced Accuracy | 0.4645 |
| Precision | 0.3256 |
| Recall | 0.6364 |
| F1 | 0.4308 |
| ROC-AUC | 0.5576 |
| Log Loss | 1.1129 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.4698 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6205 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[12, 29], [8, 14]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0021 |
| p25 | 0.4081 |
| median | 0.6118 |
| p75 | 0.7940 |
| max | 0.9787 |
| mean | 0.5543 |
| std | 0.3246 |
| count_ge_threshold | 43 |
| count_lt_threshold | 20 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 341 | 934.8380 | 0.1839 |
| 2 | volatility_60 | 268 | 614.7867 | 0.1209 |
| 3 | volatility_20 | 285 | 608.5854 | 0.1197 |
| 4 | macd_signal_to_close | 211 | 359.4564 | 0.0707 |
| 5 | ridge_pred_future_log_return | 123 | 285.3196 | 0.0561 |
| 6 | volume_change_5 | 121 | 195.9820 | 0.0385 |
| 7 | volume_change_20 | 167 | 181.1287 | 0.0356 |
| 8 | ema_12_to_ema_26 | 107 | 181.0381 | 0.0356 |
| 9 | ma_gap_60 | 100 | 174.1301 | 0.0342 |
| 10 | return_60 | 95 | 156.3545 | 0.0308 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-01-19 |
| Train End | 2020-03-06 |
| Test Start | 2020-03-24 |
| Test End | 2020-06-24 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6984 |
| Balanced Accuracy | 0.5919 |
| Precision | 0.7308 |
| Recall | 0.8837 |
| F1 | 0.8000 |
| ROC-AUC | 0.7477 |
| Log Loss | 1.1944 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.0137 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5295 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 14], [5, 38]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0045 |
| p25 | 0.0216 |
| median | 0.2886 |
| p75 | 0.9000 |
| max | 0.9875 |
| mean | 0.4378 |
| std | 0.4204 |
| count_ge_threshold | 52 |
| count_lt_threshold | 11 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 324 | 886.3816 | 0.1769 |
| 2 | volatility_60 | 209 | 680.0690 | 0.1357 |
| 3 | ma_gap_60 | 131 | 325.8338 | 0.0650 |
| 4 | volatility_20 | 145 | 311.3503 | 0.0621 |
| 5 | return_60 | 133 | 272.7483 | 0.0544 |
| 6 | ema_12_to_ema_26 | 143 | 272.1855 | 0.0543 |
| 7 | ma_gap_120 | 139 | 222.6769 | 0.0444 |
| 8 | ma_gap_20 | 121 | 205.2688 | 0.0410 |
| 9 | return_5 | 124 | 190.0128 | 0.0379 |
| 10 | volume_change_20 | 175 | 181.1919 | 0.0362 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-04-23 |
| Train End | 2020-06-10 |
| Test Start | 2020-06-25 |
| Test End | 2020-09-30 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.5139 |
| Precision | 1.0000 |
| Recall | 0.0278 |
| F1 | 0.0541 |
| ROC-AUC | 0.5947 |
| Log Loss | 0.7906 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.9098 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6456 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[27, 0], [35, 1]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0396 |
| p25 | 0.3630 |
| median | 0.6697 |
| p75 | 0.7961 |
| max | 0.9109 |
| mean | 0.5703 |
| std | 0.2688 |
| count_ge_threshold | 1 |
| count_lt_threshold | 62 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 347 | 918.2486 | 0.1836 |
| 2 | macd_signal_to_close | 168 | 331.9677 | 0.0664 |
| 3 | ema_12_to_ema_26 | 153 | 285.1751 | 0.0570 |
| 4 | ma_gap_60 | 126 | 283.3170 | 0.0567 |
| 5 | macd_hist_to_close | 153 | 253.5153 | 0.0507 |
| 6 | volatility_20 | 174 | 250.3282 | 0.0501 |
| 7 | ma_gap_120 | 154 | 243.1552 | 0.0486 |
| 8 | return_20 | 117 | 214.7926 | 0.0430 |
| 9 | volume_change_20 | 123 | 208.6316 | 0.0417 |
| 10 | volume_change_5 | 163 | 197.4607 | 0.0395 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-07-25 |
| Train End | 2020-09-14 |
| Test Start | 2020-10-02 |
| Test End | 2021-01-04 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4603 |
| Balanced Accuracy | 0.4722 |
| Precision | 0.6897 |
| Recall | 0.4444 |
| F1 | 0.5405 |
| ROC-AUC | 0.4802 |
| Log Loss | 1.1015 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.2677 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6028 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 9], [25, 20]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0496 |
| p25 | 0.1538 |
| median | 0.2554 |
| p75 | 0.5821 |
| max | 0.9426 |
| mean | 0.3798 |
| std | 0.2921 |
| count_ge_threshold | 29 |
| count_lt_threshold | 34 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 242 | 483.6486 | 0.0956 |
| 2 | volatility_60 | 318 | 477.1199 | 0.0943 |
| 3 | ma_gap_120 | 244 | 467.9140 | 0.0925 |
| 4 | ma_gap_60 | 78 | 318.7020 | 0.0630 |
| 5 | return_60 | 147 | 300.2047 | 0.0593 |
| 6 | macd_signal_to_close | 145 | 279.4875 | 0.0552 |
| 7 | volume_change_20 | 155 | 231.0885 | 0.0457 |
| 8 | return_20 | 117 | 230.8433 | 0.0456 |
| 9 | ma_gap_5 | 128 | 228.2826 | 0.0451 |
| 10 | macd_hist_to_close | 135 | 224.5145 | 0.0444 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-10-26 |
| Train End | 2020-12-17 |
| Test Start | 2021-01-05 |
| Test End | 2021-04-06 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.5294 |
| Precision | 1.0000 |
| Recall | 0.0588 |
| F1 | 0.1111 |
| ROC-AUC | 0.6805 |
| Log Loss | 1.1061 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.7136 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5881 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[29, 0], [32, 2]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0058 |
| p25 | 0.0461 |
| median | 0.1099 |
| p75 | 0.2730 |
| max | 0.7322 |
| mean | 0.1815 |
| std | 0.1828 |
| count_ge_threshold | 2 |
| count_lt_threshold | 61 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 244 | 507.2527 | 0.0980 |
| 2 | ma_gap_120 | 193 | 470.5321 | 0.0909 |
| 3 | ma_gap_60 | 126 | 441.6507 | 0.0854 |
| 4 | macd_signal_to_close | 142 | 377.7088 | 0.0730 |
| 5 | volume_change_20 | 222 | 362.7071 | 0.0701 |
| 6 | volatility_20 | 120 | 346.5666 | 0.0670 |
| 7 | ridge_pred_future_log_return | 169 | 300.7939 | 0.0581 |
| 8 | return_20 | 85 | 287.2237 | 0.0555 |
| 9 | macd_hist_to_close | 107 | 233.4342 | 0.0451 |
| 10 | high_low_range | 217 | 216.4439 | 0.0418 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-02-13 |
| Train End | 2021-03-23 |
| Test Start | 2021-04-07 |
| Test End | 2021-07-08 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2698 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.2698 |
| Recall | 1.0000 |
| F1 | 0.4250 |
| ROC-AUC | 0.4501 |
| Log Loss | 1.2004 |
| Baseline Accuracy | 0.7302 |
| Decision Threshold | 0.0306 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6308 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 46], [0, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0331 |
| p25 | 0.3527 |
| median | 0.6445 |
| p75 | 0.8276 |
| max | 0.9921 |
| mean | 0.5949 |
| std | 0.2832 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 178 | 682.9572 | 0.1294 |
| 2 | ridge_pred_future_log_return | 223 | 582.8521 | 0.1104 |
| 3 | close_to_ema_12 | 205 | 473.8505 | 0.0898 |
| 4 | volume_change_20 | 189 | 332.6693 | 0.0630 |
| 5 | macd_signal_to_close | 162 | 317.1648 | 0.0601 |
| 6 | return_60 | 211 | 284.5855 | 0.0539 |
| 7 | return_10 | 200 | 280.7116 | 0.0532 |
| 8 | volatility_20 | 248 | 256.1006 | 0.0485 |
| 9 | volatility_60 | 168 | 220.2933 | 0.0417 |
| 10 | volume_z_20 | 254 | 182.3186 | 0.0345 |

### Fold 10

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-05-24 |
| Train End | 2021-06-24 |
| Test Start | 2021-07-09 |
| Test End | 2021-10-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5873 |
| Balanced Accuracy | 0.5568 |
| Precision | 0.7143 |
| Recall | 0.1724 |
| F1 | 0.2778 |
| ROC-AUC | 0.5193 |
| Log Loss | 1.0456 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.9265 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6578 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[32, 2], [24, 5]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0097 |
| p25 | 0.1852 |
| median | 0.5374 |
| p75 | 0.7133 |
| max | 0.9870 |
| mean | 0.4805 |
| std | 0.3183 |
| count_ge_threshold | 7 |
| count_lt_threshold | 56 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 215 | 507.7390 | 0.0971 |
| 2 | volatility_60 | 327 | 493.2078 | 0.0943 |
| 3 | macd_hist_to_close | 141 | 477.7946 | 0.0914 |
| 4 | ridge_pred_future_log_return | 125 | 357.2360 | 0.0683 |
| 5 | ma_gap_120 | 137 | 321.4253 | 0.0615 |
| 6 | return_60 | 218 | 301.0919 | 0.0576 |
| 7 | close_to_ema_26 | 83 | 273.2703 | 0.0523 |
| 8 | close_to_ema_12 | 119 | 271.2988 | 0.0519 |
| 9 | volatility_20 | 137 | 247.4947 | 0.0473 |
| 10 | volume_z_20 | 134 | 187.4729 | 0.0358 |

### Fold 11

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-08-26 |
| Train End | 2021-09-28 |
| Test Start | 2021-10-13 |
| Test End | 2022-01-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5714 |
| Balanced Accuracy | 0.5695 |
| Precision | 0.4667 |
| Recall | 0.5600 |
| F1 | 0.5091 |
| ROC-AUC | 0.6021 |
| Log Loss | 1.8971 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.9657 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6337 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[22, 16], [11, 14]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1557 |
| p25 | 0.8970 |
| median | 0.9583 |
| p75 | 0.9888 |
| max | 0.9969 |
| mean | 0.8827 |
| std | 0.1891 |
| count_ge_threshold | 30 |
| count_lt_threshold | 33 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 210 | 702.9595 | 0.1346 |
| 2 | ma_gap_60 | 145 | 595.6900 | 0.1141 |
| 3 | macd_signal_to_close | 242 | 408.1564 | 0.0782 |
| 4 | ma_gap_120 | 154 | 399.8074 | 0.0766 |
| 5 | rsi_14 | 77 | 227.6798 | 0.0436 |
| 6 | ma_gap_20 | 55 | 227.3601 | 0.0435 |
| 7 | ema_12_to_ema_26 | 94 | 223.3592 | 0.0428 |
| 8 | close_to_ema_12 | 149 | 219.7720 | 0.0421 |
| 9 | close_to_high | 161 | 205.4121 | 0.0393 |
| 10 | volume_change_5 | 93 | 190.6058 | 0.0365 |

### Fold 12

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-12-02 |
| Train End | 2021-12-28 |
| Test Start | 2022-01-17 |
| Test End | 2022-04-18 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.6262 |
| Precision | 0.3878 |
| Recall | 0.9500 |
| F1 | 0.5507 |
| ROC-AUC | 0.7605 |
| Log Loss | 1.2653 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.4780 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6227 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 30], [1, 19]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0335 |
| p25 | 0.5515 |
| median | 0.8511 |
| p75 | 0.9734 |
| max | 0.9920 |
| mean | 0.7365 |
| std | 0.2859 |
| count_ge_threshold | 49 |
| count_lt_threshold | 14 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 396 | 1139.5252 | 0.2166 |
| 2 | ma_gap_20 | 65 | 459.2400 | 0.0873 |
| 3 | macd_signal_to_close | 316 | 384.4588 | 0.0731 |
| 4 | ridge_pred_future_log_return | 164 | 316.2410 | 0.0601 |
| 5 | volume_change_5 | 143 | 312.8673 | 0.0595 |
| 6 | ma_gap_60 | 167 | 282.0909 | 0.0536 |
| 7 | macd_hist_to_close | 153 | 251.9508 | 0.0479 |
| 8 | volume_change_20 | 168 | 216.2908 | 0.0411 |
| 9 | close_to_high | 151 | 149.6775 | 0.0285 |
| 10 | close_to_ema_26 | 68 | 140.9913 | 0.0268 |

### Fold 13

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-03-09 |
| Train End | 2022-04-04 |
| Test Start | 2022-04-19 |
| Test End | 2022-07-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6984 |
| Balanced Accuracy | 0.7181 |
| Precision | 0.9412 |
| Recall | 0.4706 |
| F1 | 0.6275 |
| ROC-AUC | 0.8043 |
| Log Loss | 0.6328 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.9384 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5965 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[28, 1], [18, 16]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0408 |
| p25 | 0.3736 |
| median | 0.7356 |
| p75 | 0.9423 |
| max | 0.9948 |
| mean | 0.6507 |
| std | 0.3082 |
| count_ge_threshold | 17 |
| count_lt_threshold | 46 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 367 | 1015.5272 | 0.1920 |
| 2 | volatility_20 | 249 | 427.7795 | 0.0809 |
| 3 | ridge_pred_future_log_return | 205 | 414.4396 | 0.0783 |
| 4 | macd_signal_to_close | 194 | 398.5781 | 0.0753 |
| 5 | volume_change_5 | 195 | 338.3740 | 0.0640 |
| 6 | return_10 | 207 | 328.4323 | 0.0621 |
| 7 | high_low_range | 113 | 316.8053 | 0.0599 |
| 8 | ma_gap_60 | 123 | 231.0485 | 0.0437 |
| 9 | macd_hist_to_close | 120 | 196.5336 | 0.0371 |
| 10 | volume_change_20 | 166 | 167.0093 | 0.0316 |

### Fold 14

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-06-11 |
| Train End | 2022-07-06 |
| Test Start | 2022-07-22 |
| Test End | 2022-10-24 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5304 |
| Precision | 0.5000 |
| Recall | 0.4138 |
| F1 | 0.4528 |
| ROC-AUC | 0.4868 |
| Log Loss | 1.2706 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.6916 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7068 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[22, 12], [17, 12]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0048 |
| p25 | 0.0939 |
| median | 0.3743 |
| p75 | 0.8416 |
| max | 0.9921 |
| mean | 0.4566 |
| std | 0.3650 |
| count_ge_threshold | 24 |
| count_lt_threshold | 39 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 322 | 774.0601 | 0.1485 |
| 2 | ema_12_to_ema_26 | 130 | 392.2696 | 0.0753 |
| 3 | return_20 | 129 | 389.0884 | 0.0747 |
| 4 | macd_signal_to_close | 191 | 333.0037 | 0.0639 |
| 5 | volatility_20 | 174 | 320.2854 | 0.0615 |
| 6 | volume_change_5 | 214 | 317.8080 | 0.0610 |
| 7 | return_5 | 163 | 285.3031 | 0.0548 |
| 8 | macd_hist_to_close | 136 | 265.9353 | 0.0510 |
| 9 | return_60 | 170 | 227.0631 | 0.0436 |
| 10 | high_low_range | 109 | 220.2158 | 0.0423 |

### Fold 15

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-09-15 |
| Train End | 2022-10-07 |
| Test Start | 2022-10-25 |
| Test End | 2023-01-26 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.6653 |
| Precision | 0.6923 |
| Recall | 0.5806 |
| F1 | 0.6316 |
| ROC-AUC | 0.6583 |
| Log Loss | 1.1790 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.2964 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6037 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[24, 8], [13, 18]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0018 |
| p25 | 0.0316 |
| median | 0.1093 |
| p75 | 0.8064 |
| max | 0.9937 |
| mean | 0.3596 |
| std | 0.3895 |
| count_ge_threshold | 26 |
| count_lt_threshold | 37 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_20 | 232 | 877.1170 | 0.1699 |
| 2 | volatility_60 | 284 | 639.2050 | 0.1238 |
| 3 | macd_signal_to_close | 188 | 413.6618 | 0.0801 |
| 4 | close_to_high | 263 | 366.4402 | 0.0710 |
| 5 | return_5 | 162 | 338.6396 | 0.0656 |
| 6 | return_10 | 167 | 267.7357 | 0.0518 |
| 7 | volatility_20 | 194 | 265.5140 | 0.0514 |
| 8 | ema_12_to_ema_26 | 111 | 190.2782 | 0.0368 |
| 9 | ridge_pred_future_log_return | 128 | 186.7341 | 0.0362 |
| 10 | ma_gap_20 | 131 | 178.8444 | 0.0346 |

### Fold 16

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-12-18 |
| Train End | 2023-01-12 |
| Test Start | 2023-01-27 |
| Test End | 2023-04-27 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6508 |
| Balanced Accuracy | 0.6790 |
| Precision | 0.8276 |
| Recall | 0.5854 |
| F1 | 0.6857 |
| ROC-AUC | 0.7439 |
| Log Loss | 1.0064 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.2169 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5352 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 5], [17, 24]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0203 |
| p25 | 0.0866 |
| median | 0.1667 |
| p75 | 0.4310 |
| max | 0.9300 |
| mean | 0.2797 |
| std | 0.2366 |
| count_ge_threshold | 29 |
| count_lt_threshold | 34 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 405 | 919.3092 | 0.1772 |
| 2 | macd_signal_to_close | 283 | 790.6991 | 0.1524 |
| 3 | return_20 | 235 | 705.0776 | 0.1359 |
| 4 | ema_12_to_ema_26 | 141 | 305.9939 | 0.0590 |
| 5 | close_to_high | 171 | 263.3312 | 0.0508 |
| 6 | ma_gap_20 | 37 | 240.7467 | 0.0464 |
| 7 | ma_gap_120 | 99 | 164.8662 | 0.0318 |
| 8 | return_5 | 142 | 164.2612 | 0.0317 |
| 9 | close_to_low | 189 | 149.5227 | 0.0288 |
| 10 | ma_gap_60 | 53 | 147.7254 | 0.0285 |

### Fold 17

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-03-24 |
| Train End | 2023-04-13 |
| Test Start | 2023-04-28 |
| Test End | 2023-07-31 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6984 |
| Balanced Accuracy | 0.6346 |
| Precision | 0.6607 |
| Recall | 1.0000 |
| F1 | 0.7957 |
| ROC-AUC | 0.7786 |
| Log Loss | 0.9269 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.0656 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5962 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 19], [0, 37]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0109 |
| p25 | 0.0942 |
| median | 0.1618 |
| p75 | 0.2589 |
| max | 0.8890 |
| mean | 0.2290 |
| std | 0.2041 |
| count_ge_threshold | 56 |
| count_lt_threshold | 7 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_20 | 157 | 567.2726 | 0.1146 |
| 2 | macd_signal_to_close | 258 | 544.9349 | 0.1101 |
| 3 | volatility_60 | 263 | 515.6432 | 0.1041 |
| 4 | ema_12_to_ema_26 | 230 | 323.0007 | 0.0652 |
| 5 | close_to_high | 142 | 309.8155 | 0.0626 |
| 6 | volatility_20 | 175 | 266.6914 | 0.0539 |
| 7 | return_60 | 154 | 235.9464 | 0.0476 |
| 8 | macd_hist_to_close | 131 | 223.1437 | 0.0451 |
| 9 | ma_gap_120 | 114 | 184.6928 | 0.0373 |
| 10 | ma_gap_60 | 96 | 137.2610 | 0.0277 |

### Fold 18

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-06-25 |
| Train End | 2023-07-14 |
| Test Start | 2023-08-01 |
| Test End | 2023-10-31 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.4107 |
| Precision | 0.2308 |
| Recall | 0.1071 |
| F1 | 0.1463 |
| ROC-AUC | 0.5245 |
| Log Loss | 0.8935 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.6992 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5150 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[25, 10], [25, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0129 |
| p25 | 0.1435 |
| median | 0.3347 |
| p75 | 0.5971 |
| max | 0.9142 |
| mean | 0.3861 |
| std | 0.2689 |
| count_ge_threshold | 13 |
| count_lt_threshold | 50 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 243 | 706.9385 | 0.1430 |
| 2 | macd_signal_to_close | 177 | 551.4594 | 0.1115 |
| 3 | ma_gap_60 | 166 | 392.1545 | 0.0793 |
| 4 | macd_to_close | 126 | 374.2205 | 0.0757 |
| 5 | ema_12_to_ema_26 | 164 | 301.8139 | 0.0610 |
| 6 | ma_gap_20 | 102 | 217.5923 | 0.0440 |
| 7 | volume_z_20 | 120 | 213.7398 | 0.0432 |
| 8 | ma_gap_120 | 128 | 200.5709 | 0.0406 |
| 9 | return_20 | 109 | 177.3627 | 0.0359 |
| 10 | volume_change_5 | 161 | 166.5831 | 0.0337 |

### Fold 19

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-09-29 |
| Train End | 2023-10-17 |
| Test Start | 2023-11-01 |
| Test End | 2024-02-05 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3175 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.8128 |
| Log Loss | 0.9440 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.9900 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5076 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 0], [43, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0066 |
| p25 | 0.0835 |
| median | 0.1890 |
| p75 | 0.5338 |
| max | 0.9167 |
| mean | 0.3115 |
| std | 0.2736 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 186 | 687.3445 | 0.1327 |
| 2 | ridge_pred_future_log_return | 216 | 588.8584 | 0.1137 |
| 3 | volatility_60 | 200 | 332.1050 | 0.0641 |
| 4 | volatility_20 | 156 | 284.9193 | 0.0550 |
| 5 | close_to_ema_12 | 68 | 273.8385 | 0.0529 |
| 6 | ma_gap_60 | 117 | 270.9484 | 0.0523 |
| 7 | volume_change_20 | 213 | 246.5653 | 0.0476 |
| 8 | return_2 | 136 | 239.5062 | 0.0462 |
| 9 | macd_hist_to_close | 121 | 205.2744 | 0.0396 |
| 10 | ma_gap_120 | 123 | 180.3712 | 0.0348 |

### Fold 20

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-12-29 |
| Train End | 2024-01-22 |
| Test Start | 2024-02-06 |
| Test End | 2024-05-10 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.5061 |
| Precision | 0.5273 |
| Recall | 0.8788 |
| F1 | 0.6591 |
| ROC-AUC | 0.7162 |
| Log Loss | 0.8035 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.0698 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5577 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[4, 26], [4, 29]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0193 |
| p25 | 0.1309 |
| median | 0.2334 |
| p75 | 0.3668 |
| max | 0.9036 |
| mean | 0.2858 |
| std | 0.2060 |
| count_ge_threshold | 55 |
| count_lt_threshold | 8 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 378 | 773.6405 | 0.1477 |
| 2 | return_60 | 190 | 557.2905 | 0.1064 |
| 3 | ridge_pred_future_log_return | 202 | 546.5731 | 0.1044 |
| 4 | volatility_20 | 184 | 407.5049 | 0.0778 |
| 5 | macd_hist_to_close | 172 | 332.7534 | 0.0635 |
| 6 | close_to_high | 210 | 252.1578 | 0.0481 |
| 7 | ma_gap_60 | 91 | 245.9876 | 0.0470 |
| 8 | ma_gap_5 | 127 | 208.3009 | 0.0398 |
| 9 | ridge_pred_margin_to_threshold | 76 | 173.3113 | 0.0331 |
| 10 | ma_gap_120 | 58 | 168.3446 | 0.0321 |

### Fold 21

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-04-05 |
| Train End | 2024-04-23 |
| Test Start | 2024-05-13 |
| Test End | 2024-08-08 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5281 |
| Precision | 0.4444 |
| Recall | 0.4615 |
| F1 | 0.4528 |
| ROC-AUC | 0.4990 |
| Log Loss | 0.9525 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.4621 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6095 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[22, 15], [14, 12]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0410 |
| p25 | 0.2021 |
| median | 0.3386 |
| p75 | 0.6129 |
| max | 0.9588 |
| mean | 0.4205 |
| std | 0.2667 |
| count_ge_threshold | 27 |
| count_lt_threshold | 36 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 369 | 1104.0247 | 0.2077 |
| 2 | return_60 | 236 | 500.7057 | 0.0942 |
| 3 | volatility_60 | 238 | 481.9842 | 0.0907 |
| 4 | ridge_pred_future_log_return | 138 | 430.0365 | 0.0809 |
| 5 | macd_hist_to_close | 307 | 402.0641 | 0.0756 |
| 6 | volatility_20 | 241 | 327.3349 | 0.0616 |
| 7 | ridge_pred_margin_to_threshold | 109 | 221.6270 | 0.0417 |
| 8 | ma_gap_120 | 160 | 217.4522 | 0.0409 |
| 9 | ma_gap_60 | 123 | 194.2721 | 0.0366 |
| 10 | return_20 | 94 | 152.8669 | 0.0288 |

### Fold 22

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-07-07 |
| Train End | 2024-07-25 |
| Test Start | 2024-08-09 |
| Test End | 2024-11-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7460 |
| Balanced Accuracy | 0.7485 |
| Precision | 0.6829 |
| Recall | 0.9032 |
| F1 | 0.7778 |
| ROC-AUC | 0.7117 |
| Log Loss | 0.7822 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.2846 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5849 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 13], [3, 28]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0027 |
| p25 | 0.1264 |
| median | 0.7893 |
| p75 | 0.9381 |
| max | 0.9870 |
| mean | 0.5718 |
| std | 0.3974 |
| count_ge_threshold | 41 |
| count_lt_threshold | 22 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ridge_pred_future_log_return | 248 | 931.8714 | 0.1761 |
| 2 | volatility_60 | 248 | 541.0031 | 0.1022 |
| 3 | return_60 | 253 | 395.0642 | 0.0747 |
| 4 | rsi_14 | 131 | 384.0610 | 0.0726 |
| 5 | ma_gap_120 | 152 | 325.0506 | 0.0614 |
| 6 | macd_signal_to_close | 131 | 265.8861 | 0.0502 |
| 7 | volatility_20 | 152 | 264.5347 | 0.0500 |
| 8 | ridge_pred_future_return | 42 | 222.6897 | 0.0421 |
| 9 | ridge_pred_margin_to_threshold | 88 | 211.8549 | 0.0400 |
| 10 | return_10 | 175 | 171.6931 | 0.0324 |

### Fold 23

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-10-11 |
| Train End | 2024-10-28 |
| Test Start | 2024-11-13 |
| Test End | 2025-02-17 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.7607 |
| Log Loss | 0.5604 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.9184 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6858 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[39, 0], [24, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0130 |
| p25 | 0.1993 |
| median | 0.3246 |
| p75 | 0.5384 |
| max | 0.8065 |
| mean | 0.3584 |
| std | 0.2253 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 337 | 729.0256 | 0.1366 |
| 2 | volatility_20 | 298 | 679.2286 | 0.1272 |
| 3 | ridge_pred_margin_to_threshold | 141 | 579.0496 | 0.1085 |
| 4 | volatility_60 | 278 | 557.9504 | 0.1045 |
| 5 | return_60 | 173 | 419.4179 | 0.0786 |
| 6 | ema_12_to_ema_26 | 131 | 311.4105 | 0.0583 |
| 7 | ridge_pred_future_log_return | 135 | 233.0590 | 0.0437 |
| 8 | ma_gap_120 | 91 | 205.7708 | 0.0385 |
| 9 | rsi_14 | 63 | 197.2858 | 0.0370 |
| 10 | ma_gap_5 | 176 | 191.6035 | 0.0359 |

### Fold 24

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-01-13 |
| Train End | 2025-01-31 |
| Test Start | 2025-02-18 |
| Test End | 2025-05-22 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6349 |
| Balanced Accuracy | 0.6321 |
| Precision | 0.6047 |
| Recall | 0.8125 |
| F1 | 0.6933 |
| ROC-AUC | 0.6356 |
| Log Loss | 0.8926 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.6357 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6093 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 17], [6, 26]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0772 |
| p25 | 0.5440 |
| median | 0.8077 |
| p75 | 0.9180 |
| max | 0.9906 |
| mean | 0.7099 |
| std | 0.2480 |
| count_ge_threshold | 43 |
| count_lt_threshold | 20 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 260 | 642.0089 | 0.1254 |
| 2 | macd_signal_to_close | 178 | 616.7181 | 0.1205 |
| 3 | ma_gap_120 | 203 | 493.9778 | 0.0965 |
| 4 | ma_gap_60 | 135 | 489.0059 | 0.0955 |
| 5 | volatility_20 | 247 | 366.5360 | 0.0716 |
| 6 | volume_change_5 | 203 | 336.9857 | 0.0658 |
| 7 | ema_12_to_ema_26 | 144 | 263.7805 | 0.0515 |
| 8 | return_60 | 171 | 211.6274 | 0.0413 |
| 9 | ridge_pred_future_log_return | 168 | 207.7147 | 0.0406 |
| 10 | return_5 | 131 | 165.4844 | 0.0323 |

### Fold 25

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-04-14 |
| Train End | 2025-05-08 |
| Test Start | 2025-05-23 |
| Test End | 2025-08-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4286 |
| Balanced Accuracy | 0.5814 |
| Precision | 1.0000 |
| Recall | 0.1628 |
| F1 | 0.2800 |
| ROC-AUC | 0.7174 |
| Log Loss | 0.7045 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.8306 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6232 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 0], [36, 7]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0368 |
| p25 | 0.2594 |
| median | 0.4316 |
| p75 | 0.6818 |
| max | 0.9602 |
| mean | 0.4565 |
| std | 0.2583 |
| count_ge_threshold | 7 |
| count_lt_threshold | 56 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 278 | 679.5889 | 0.1306 |
| 2 | volatility_60 | 250 | 598.6465 | 0.1151 |
| 3 | volatility_20 | 165 | 508.5365 | 0.0978 |
| 4 | return_20 | 145 | 362.8938 | 0.0698 |
| 5 | ma_gap_60 | 136 | 350.7208 | 0.0674 |
| 6 | ema_12_to_ema_26 | 186 | 317.9085 | 0.0611 |
| 7 | macd_signal_to_close | 152 | 290.2275 | 0.0558 |
| 8 | volume_z_20 | 138 | 264.7546 | 0.0509 |
| 9 | ma_gap_20 | 82 | 208.4133 | 0.0401 |
| 10 | macd_hist_to_close | 143 | 170.3429 | 0.0327 |

### Fold 26

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-07-18 |
| Train End | 2025-08-06 |
| Test Start | 2025-08-22 |
| Test End | 2025-11-25 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.4641 |
| Precision | 0.7222 |
| Recall | 0.5532 |
| F1 | 0.6265 |
| ROC-AUC | 0.4176 |
| Log Loss | 1.1640 |
| Baseline Accuracy | 0.7460 |
| Decision Threshold | 0.3465 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5525 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 10], [21, 26]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0139 |
| p25 | 0.1835 |
| median | 0.5041 |
| p75 | 0.7720 |
| max | 0.9873 |
| mean | 0.4860 |
| std | 0.3182 |
| count_ge_threshold | 36 |
| count_lt_threshold | 27 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 276 | 747.2461 | 0.1413 |
| 2 | macd_signal_to_close | 164 | 482.4813 | 0.0912 |
| 3 | volatility_60 | 251 | 467.7200 | 0.0884 |
| 4 | ridge_pred_future_log_return | 202 | 457.7023 | 0.0866 |
| 5 | volatility_20 | 229 | 319.6111 | 0.0604 |
| 6 | ma_gap_60 | 74 | 299.5849 | 0.0567 |
| 7 | rsi_14 | 76 | 263.5801 | 0.0498 |
| 8 | ma_gap_20 | 69 | 217.8406 | 0.0412 |
| 9 | return_5 | 196 | 186.5965 | 0.0353 |
| 10 | high_low_range | 155 | 159.5052 | 0.0302 |

### Fold 27

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-10-18 |
| Train End | 2025-11-10 |
| Test Start | 2025-11-26 |
| Test End | 2026-03-02 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.6216 |
| Precision | 1.0000 |
| Recall | 0.2432 |
| F1 | 0.3913 |
| ROC-AUC | 0.7484 |
| Log Loss | 1.7728 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.1320 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5258 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[26, 0], [28, 9]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0029 |
| p25 | 0.0120 |
| median | 0.0228 |
| p75 | 0.0555 |
| max | 0.9820 |
| mean | 0.0884 |
| std | 0.1847 |
| count_ge_threshold | 9 |
| count_lt_threshold | 54 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 295 | 1073.1240 | 0.2026 |
| 2 | volatility_60 | 385 | 817.7969 | 0.1544 |
| 3 | ma_gap_60 | 192 | 362.4708 | 0.0684 |
| 4 | volatility_20 | 227 | 274.7881 | 0.0519 |
| 5 | ma_gap_20 | 94 | 238.0240 | 0.0449 |
| 6 | macd_signal_to_close | 165 | 228.6575 | 0.0432 |
| 7 | high_low_range | 149 | 223.7090 | 0.0422 |
| 8 | ridge_pred_future_log_return | 163 | 222.3293 | 0.0420 |
| 9 | ma_gap_120 | 130 | 208.0963 | 0.0393 |
| 10 | ridge_pred_margin_to_threshold | 101 | 182.9002 | 0.0345 |

### Fold 28

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2024-01-23 |
| Train End | 2026-02-13 |
| Test Start | 2026-03-03 |
| Test End | 2026-06-04 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3492 |
| Balanced Accuracy | 0.3995 |
| Precision | 0.5714 |
| Recall | 0.2727 |
| F1 | 0.3692 |
| ROC-AUC | 0.3768 |
| Log Loss | 1.9984 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.3374 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6544 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[10, 9], [32, 12]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0060 |
| p25 | 0.0234 |
| median | 0.0565 |
| p75 | 0.4578 |
| max | 0.8553 |
| mean | 0.2492 |
| std | 0.2854 |
| count_ge_threshold | 21 |
| count_lt_threshold | 42 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 365 | 770.0402 | 0.1457 |
| 2 | macd_signal_to_close | 199 | 545.2213 | 0.1032 |
| 3 | ma_gap_120 | 203 | 534.0128 | 0.1011 |
| 4 | macd_hist_to_close | 193 | 529.1003 | 0.1001 |
| 5 | return_60 | 146 | 390.2494 | 0.0738 |
| 6 | return_20 | 153 | 341.0241 | 0.0645 |
| 7 | ema_12_to_ema_26 | 126 | 252.5285 | 0.0478 |
| 8 | volatility_20 | 139 | 228.9134 | 0.0433 |
| 9 | macd_to_close | 101 | 208.8721 | 0.0395 |
| 10 | ma_gap_5 | 143 | 158.3080 | 0.0300 |
