# Direction Validation Report: ^N225

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | ^N225 |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-06-28 23:22:11 JST |
| Last Date | 2026-06-26 |
| Last Close | 69360.8828 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.9408 |
| Probability Down | 0.0592 |
| Decision Threshold | 0.1677 |
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
| fold_count | 9 | N/A | N/A | N/A |
| total_fold_count | 9 | N/A | N/A | N/A |
| beats_baseline_fold_ratio | 0.2222 | N/A | N/A | N/A |
| confusion_matrix_sum | [[112, 136], [150, 169]] | N/A | N/A | N/A |
| accuracy | 0.4956 | 0.1381 | 0.2540 | 0.6667 |
| balanced_accuracy | 0.5027 | 0.0556 | 0.4455 | 0.6321 |
| precision | 0.4485 | 0.1947 | 0.0000 | 0.6667 |
| recall | 0.5407 | 0.4183 | 0.0000 | 1.0000 |
| f1 | 0.4399 | 0.3015 | 0.0000 | 0.8000 |
| roc_auc | 0.5870 | 0.1312 | 0.4293 | 0.8035 |
| log_loss | 1.1109 | 0.3940 | 0.5479 | 1.7617 |
| baseline_accuracy | 0.6208 | 0.0619 | 0.5397 | 0.7460 |
| decision_threshold | 0.4605 | 0.3199 | 0.0156 | 0.9525 |
| calibration_score | 0.5939 | 0.0670 | 0.5000 | 0.7021 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 314 | 953.1039 | 0.0946 |
| 2 | ema_12 | 222 | 939.5844 | 0.0932 |
| 3 | ema_26 | 257 | 935.3243 | 0.0928 |
| 4 | ma_gap_120 | 206 | 787.7559 | 0.0781 |
| 5 | macd_signal | 292 | 742.9798 | 0.0737 |
| 6 | volatility_10 | 211 | 720.5364 | 0.0715 |
| 7 | macd_hist | 176 | 691.6139 | 0.0686 |
| 8 | macd | 141 | 538.7030 | 0.0534 |
| 9 | return_60 | 180 | 469.1390 | 0.0465 |
| 10 | volatility_5 | 222 | 408.8291 | 0.0406 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2024-01-29 | 2024-04-30 | 0.4286 | 0.4503 | 0.4949 | 0.5397 | 0.7788 | ok |
| 2 | ok | 2024-05-01 | 2024-07-31 | 0.5079 | 0.4610 | 0.4293 | 0.5873 | 0.5166 | ok |
| 3 | ok | 2024-08-01 | 2024-11-01 | 0.6190 | 0.5556 | 0.8035 | 0.5714 | 0.2674 | ok |
| 4 | ok | 2024-11-05 | 2025-02-06 | 0.4127 | 0.4455 | 0.4530 | 0.6190 | 0.2124 | ok |
| 5 | ok | 2025-02-07 | 2025-05-14 | 0.6032 | 0.6321 | 0.7020 | 0.5556 | 0.6323 | ok |
| 6 | ok | 2025-05-15 | 2025-08-13 | 0.2540 | 0.4794 | 0.5120 | 0.7460 | 0.9525 | ok |
| 7 | ok | 2025-08-14 | 2025-11-14 | 0.3333 | 0.5000 | 0.6236 | 0.6667 | 0.7284 | ok |
| 8 | ok | 2025-11-17 | 2026-02-19 | 0.6667 | 0.5000 | 0.7596 | 0.6667 | 0.0156 | ok |
| 9 | ok | 2026-02-20 | 2026-05-27 | 0.6349 | 0.5000 | 0.5054 | 0.6349 | 0.0406 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-12-21 |
| Train End | 2024-01-12 |
| Test Start | 2024-01-29 |
| Test End | 2024-04-30 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4286 |
| Balanced Accuracy | 0.4503 |
| Precision | 0.4286 |
| Recall | 0.1765 |
| F1 | 0.2500 |
| ROC-AUC | 0.4949 |
| Log Loss | 0.9421 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.7788 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5000 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 8], [28, 6]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0913 |
| p25 | 0.2923 |
| median | 0.3917 |
| p75 | 0.5929 |
| max | 0.9818 |
| mean | 0.4657 |
| std | 0.2654 |
| count_ge_threshold | 14 |
| count_lt_threshold | 49 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 357 | 2022.2878 | 0.2023 |
| 2 | ema_26 | 362 | 1400.5285 | 0.1401 |
| 3 | volatility_60 | 373 | 1376.7545 | 0.1377 |
| 4 | return_60 | 189 | 775.9092 | 0.0776 |
| 5 | macd_signal | 134 | 443.8449 | 0.0444 |
| 6 | ma_gap_20 | 150 | 323.9225 | 0.0324 |
| 7 | volatility_10 | 154 | 308.5492 | 0.0309 |
| 8 | macd | 152 | 303.4359 | 0.0304 |
| 9 | macd_hist | 131 | 238.3488 | 0.0238 |
| 10 | volatility_20 | 128 | 222.6351 | 0.0223 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-03-28 |
| Train End | 2024-04-15 |
| Test Start | 2024-05-01 |
| Test End | 2024-07-31 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5079 |
| Balanced Accuracy | 0.4610 |
| Precision | 0.3333 |
| Recall | 0.1923 |
| F1 | 0.2439 |
| ROC-AUC | 0.4293 |
| Log Loss | 0.9190 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.5166 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5556 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[27, 10], [21, 5]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0446 |
| p25 | 0.2254 |
| median | 0.3359 |
| p75 | 0.4953 |
| max | 0.9175 |
| mean | 0.4012 |
| std | 0.2451 |
| count_ge_threshold | 15 |
| count_lt_threshold | 48 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_26 | 403 | 1532.3533 | 0.1542 |
| 2 | macd_signal | 211 | 1127.6339 | 0.1135 |
| 3 | return_60 | 227 | 1091.8343 | 0.1099 |
| 4 | volatility_60 | 358 | 1022.0839 | 0.1028 |
| 5 | ema_12 | 202 | 719.7058 | 0.0724 |
| 6 | macd_hist | 190 | 401.4094 | 0.0404 |
| 7 | volatility_20 | 163 | 391.5700 | 0.0394 |
| 8 | volatility_10 | 172 | 375.4861 | 0.0378 |
| 9 | ma_gap_120 | 125 | 313.6924 | 0.0316 |
| 10 | ma_gap_5 | 130 | 288.5160 | 0.0290 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-06-29 |
| Train End | 2024-07-17 |
| Test Start | 2024-08-01 |
| Test End | 2024-11-01 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.5556 |
| Precision | 0.6000 |
| Recall | 1.0000 |
| F1 | 0.7500 |
| ROC-AUC | 0.8035 |
| Log Loss | 0.6798 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.2674 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5833 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[3, 24], [0, 36]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1004 |
| p25 | 0.6404 |
| median | 0.8389 |
| p75 | 0.9549 |
| max | 0.9927 |
| mean | 0.7761 |
| std | 0.2197 |
| count_ge_threshold | 60 |
| count_lt_threshold | 3 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 294 | 1611.0355 | 0.1632 |
| 2 | ema_26 | 418 | 1605.5758 | 0.1627 |
| 3 | return_60 | 303 | 1042.1366 | 0.1056 |
| 4 | volatility_60 | 261 | 778.2005 | 0.0788 |
| 5 | macd_signal | 174 | 568.4539 | 0.0576 |
| 6 | ma_gap_5 | 179 | 394.6995 | 0.0400 |
| 7 | volatility_20 | 182 | 372.5960 | 0.0378 |
| 8 | ma_gap_60 | 82 | 355.6872 | 0.0360 |
| 9 | macd_hist | 139 | 321.6926 | 0.0326 |
| 10 | volume_change_5 | 194 | 318.9797 | 0.0323 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-09-30 |
| Train End | 2024-10-18 |
| Test Start | 2024-11-05 |
| Test End | 2025-02-06 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4127 |
| Balanced Accuracy | 0.4455 |
| Precision | 0.3415 |
| Recall | 0.5833 |
| F1 | 0.4308 |
| ROC-AUC | 0.4530 |
| Log Loss | 0.9887 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.2124 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5388 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[12, 27], [10, 14]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0484 |
| p25 | 0.1399 |
| median | 0.4178 |
| p75 | 0.7292 |
| max | 0.9089 |
| mean | 0.4508 |
| std | 0.2966 |
| count_ge_threshold | 41 |
| count_lt_threshold | 22 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 335 | 1743.6771 | 0.1755 |
| 2 | ema_26 | 471 | 1591.7969 | 0.1602 |
| 3 | return_60 | 384 | 1449.4688 | 0.1459 |
| 4 | ma_gap_60 | 129 | 543.9813 | 0.0548 |
| 5 | volatility_60 | 182 | 433.1569 | 0.0436 |
| 6 | volatility_20 | 168 | 428.5397 | 0.0431 |
| 7 | volatility_5 | 139 | 378.1233 | 0.0381 |
| 8 | macd_hist | 119 | 342.8683 | 0.0345 |
| 9 | macd_signal | 109 | 342.6297 | 0.0345 |
| 10 | ma_gap_120 | 96 | 284.0937 | 0.0286 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-01-04 |
| Train End | 2025-01-23 |
| Test Start | 2025-02-07 |
| Test End | 2025-05-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.6321 |
| Precision | 0.5319 |
| Recall | 0.8929 |
| F1 | 0.6667 |
| ROC-AUC | 0.7020 |
| Log Loss | 1.3418 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.6323 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7021 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 22], [3, 25]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0415 |
| p25 | 0.6232 |
| median | 0.9573 |
| p75 | 0.9881 |
| max | 0.9973 |
| mean | 0.7628 |
| std | 0.3314 |
| count_ge_threshold | 47 |
| count_lt_threshold | 16 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_26 | 443 | 2046.9377 | 0.1988 |
| 2 | ema_12 | 381 | 1680.5323 | 0.1632 |
| 3 | return_60 | 302 | 813.1018 | 0.0790 |
| 4 | ma_gap_20 | 144 | 663.3679 | 0.0644 |
| 5 | ma_gap_60 | 157 | 648.0854 | 0.0630 |
| 6 | volatility_60 | 243 | 499.0326 | 0.0485 |
| 7 | volatility_20 | 181 | 383.1915 | 0.0372 |
| 8 | macd_signal | 189 | 368.2444 | 0.0358 |
| 9 | volatility_10 | 202 | 365.5632 | 0.0355 |
| 10 | return_5 | 139 | 295.9327 | 0.0287 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-04-06 |
| Train End | 2025-04-25 |
| Test Start | 2025-05-15 |
| Test End | 2025-08-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2540 |
| Balanced Accuracy | 0.4794 |
| Precision | 0.5000 |
| Recall | 0.0213 |
| F1 | 0.0408 |
| ROC-AUC | 0.5120 |
| Log Loss | 1.7617 |
| Baseline Accuracy | 0.7460 |
| Decision Threshold | 0.9525 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6344 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[15, 1], [46, 1]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0006 |
| p25 | 0.0227 |
| median | 0.2256 |
| p75 | 0.7522 |
| max | 0.9564 |
| mean | 0.3670 |
| std | 0.3585 |
| count_ge_threshold | 2 |
| count_lt_threshold | 61 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_26 | 643 | 2835.5591 | 0.2723 |
| 2 | ema_12 | 269 | 1480.0896 | 0.1422 |
| 3 | ma_gap_20 | 214 | 789.9153 | 0.0759 |
| 4 | volatility_60 | 252 | 698.7879 | 0.0671 |
| 5 | return_60 | 209 | 669.6923 | 0.0643 |
| 6 | volatility_10 | 219 | 479.8194 | 0.0461 |
| 7 | volatility_20 | 191 | 439.9794 | 0.0423 |
| 8 | macd_hist | 167 | 398.1878 | 0.0382 |
| 9 | volatility_5 | 213 | 383.0203 | 0.0368 |
| 10 | macd_signal | 116 | 225.6919 | 0.0217 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-07-07 |
| Train End | 2025-07-29 |
| Test Start | 2025-08-14 |
| Test End | 2025-11-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3333 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.6236 |
| Log Loss | 1.6913 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.7284 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6045 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[21, 0], [42, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0072 |
| p25 | 0.0284 |
| median | 0.0776 |
| p75 | 0.1681 |
| max | 0.6996 |
| mean | 0.1249 |
| std | 0.1379 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_26 | 386 | 1367.1818 | 0.1341 |
| 2 | ema_12 | 304 | 1234.5646 | 0.1211 |
| 3 | macd_hist | 293 | 997.3562 | 0.0979 |
| 4 | volatility_60 | 281 | 889.8893 | 0.0873 |
| 5 | return_60 | 219 | 679.5469 | 0.0667 |
| 6 | macd_signal | 190 | 588.9174 | 0.0578 |
| 7 | volatility_10 | 204 | 531.6420 | 0.0522 |
| 8 | volatility_5 | 238 | 496.8860 | 0.0488 |
| 9 | volatility_20 | 212 | 490.8202 | 0.0482 |
| 10 | ma_gap_60 | 153 | 394.9143 | 0.0387 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-10-10 |
| Train End | 2025-10-30 |
| Test Start | 2025-11-17 |
| Test End | 2026-02-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.6667 |
| Recall | 1.0000 |
| F1 | 0.8000 |
| ROC-AUC | 0.7596 |
| Log Loss | 0.5479 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.0156 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5332 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 21], [0, 42]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0716 |
| p25 | 0.6848 |
| median | 0.8261 |
| p75 | 0.9026 |
| max | 0.9858 |
| mean | 0.7514 |
| std | 0.2198 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_26 | 370 | 1404.5911 | 0.1389 |
| 2 | ema_12 | 250 | 1076.4491 | 0.1065 |
| 3 | volatility_60 | 339 | 969.0375 | 0.0958 |
| 4 | macd_hist | 273 | 915.6238 | 0.0906 |
| 5 | volatility_5 | 352 | 709.7245 | 0.0702 |
| 6 | volatility_10 | 219 | 609.9427 | 0.0603 |
| 7 | macd_signal | 171 | 540.5129 | 0.0535 |
| 8 | volatility_20 | 190 | 429.5709 | 0.0425 |
| 9 | return_60 | 201 | 428.2533 | 0.0424 |
| 10 | ma_gap_60 | 140 | 407.0885 | 0.0403 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2024-01-15 |
| Train End | 2026-02-04 |
| Test Start | 2026-02-20 |
| Test End | 2026-05-27 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6349 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.6349 |
| Recall | 1.0000 |
| F1 | 0.7767 |
| ROC-AUC | 0.5054 |
| Log Loss | 1.1261 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.0406 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6933 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 23], [0, 40]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0585 |
| p25 | 0.4063 |
| median | 0.8492 |
| p75 | 0.9817 |
| max | 0.9946 |
| mean | 0.6856 |
| std | 0.3179 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 359 | 1298.6550 | 0.1267 |
| 2 | ema_26 | 351 | 1166.6913 | 0.1138 |
| 3 | volatility_60 | 404 | 1084.0430 | 0.1058 |
| 4 | macd_hist | 222 | 728.4362 | 0.0711 |
| 5 | volatility_10 | 211 | 656.8997 | 0.0641 |
| 6 | macd_signal | 250 | 614.5570 | 0.0600 |
| 7 | volatility_5 | 278 | 574.1185 | 0.0560 |
| 8 | ma_gap_120 | 214 | 565.9301 | 0.0552 |
| 9 | return_20 | 140 | 500.7438 | 0.0489 |
| 10 | return_60 | 208 | 482.4810 | 0.0471 |
