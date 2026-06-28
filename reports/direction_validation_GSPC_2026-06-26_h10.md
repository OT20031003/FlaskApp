# Direction Validation Report: ^GSPC

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | ^GSPC |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-06-28 23:25:17 JST |
| Last Date | 2026-06-26 |
| Last Close | 7354.0200 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.3137 |
| Probability Down | 0.6863 |
| Decision Threshold | 0.4875 |
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
| confusion_matrix_sum | [[118, 122], [199, 128]] | N/A | N/A | N/A |
| accuracy | 0.4339 | 0.1474 | 0.2222 | 0.7143 |
| balanced_accuracy | 0.5512 | 0.0666 | 0.5000 | 0.7122 |
| precision | 0.3872 | 0.3190 | 0.0000 | 0.8846 |
| recall | 0.4912 | 0.3979 | 0.0000 | 1.0000 |
| f1 | 0.3884 | 0.2796 | 0.0000 | 0.6667 |
| roc_auc | 0.5192 | 0.1637 | 0.3364 | 0.8206 |
| log_loss | 1.5336 | 0.8131 | 0.5369 | 3.0124 |
| baseline_accuracy | 0.6561 | 0.1135 | 0.5079 | 0.8413 |
| decision_threshold | 0.6691 | 0.2976 | 0.0157 | 0.9891 |
| calibration_score | 0.5793 | 0.0528 | 0.5052 | 0.6728 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 663 | 2492.6112 | 0.2430 |
| 2 | volatility_60 | 319 | 1106.5462 | 0.1079 |
| 3 | ema_26 | 289 | 737.8512 | 0.0719 |
| 4 | return_60 | 265 | 500.1391 | 0.0487 |
| 5 | volatility_10 | 188 | 476.5554 | 0.0464 |
| 6 | return_20 | 179 | 452.7617 | 0.0441 |
| 7 | macd_signal | 166 | 425.3564 | 0.0415 |
| 8 | volume_z_20 | 162 | 405.8567 | 0.0396 |
| 9 | volatility_20 | 132 | 386.2311 | 0.0376 |
| 10 | ma_gap_20 | 151 | 348.1001 | 0.0339 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2024-01-05 | 2024-04-05 | 0.2222 | 0.5000 | 0.3571 | 0.7778 | 0.7652 | ok |
| 2 | ok | 2024-04-08 | 2024-07-08 | 0.2540 | 0.5000 | 0.3364 | 0.7460 | 0.9775 | ok |
| 3 | ok | 2024-07-09 | 2024-10-04 | 0.5397 | 0.5431 | 0.6430 | 0.5397 | 0.7880 | ok |
| 4 | ok | 2024-10-07 | 2025-01-06 | 0.7143 | 0.7122 | 0.8206 | 0.5079 | 0.7768 | ok |
| 5 | ok | 2025-01-07 | 2025-04-08 | 0.4603 | 0.6047 | 0.7151 | 0.6825 | 0.8381 | ok |
| 6 | ok | 2025-04-09 | 2025-07-10 | 0.4762 | 0.5670 | 0.4981 | 0.8413 | 0.4510 | ok |
| 7 | ok | 2025-07-11 | 2025-10-08 | 0.3651 | 0.5000 | 0.5304 | 0.6349 | 0.9891 | ok |
| 8 | ok | 2025-10-09 | 2026-01-08 | 0.5397 | 0.5338 | 0.4083 | 0.5079 | 0.4209 | ok |
| 9 | ok | 2026-01-09 | 2026-04-10 | 0.3333 | 0.5000 | 0.3639 | 0.6667 | 0.0157 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-12-15 |
| Train End | 2023-12-19 |
| Test Start | 2024-01-05 |
| Test End | 2024-04-05 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2222 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.3571 |
| Log Loss | 3.0124 |
| Baseline Accuracy | 0.7778 |
| Decision Threshold | 0.7652 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6161 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 0], [49, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0036 |
| p25 | 0.0132 |
| median | 0.0206 |
| p75 | 0.0367 |
| max | 0.3278 |
| mean | 0.0400 |
| std | 0.0594 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_26 | 362 | 1555.9168 | 0.1511 |
| 2 | volatility_60 | 336 | 1277.8352 | 0.1241 |
| 3 | return_60 | 300 | 889.2054 | 0.0863 |
| 4 | volatility_20 | 316 | 816.2003 | 0.0793 |
| 5 | ema_12 | 193 | 765.3627 | 0.0743 |
| 6 | macd_signal | 191 | 708.3609 | 0.0688 |
| 7 | return_20 | 203 | 449.7810 | 0.0437 |
| 8 | macd_hist | 172 | 441.7709 | 0.0429 |
| 9 | volume_change_20 | 211 | 358.8535 | 0.0348 |
| 10 | ma_gap_60 | 145 | 333.6207 | 0.0324 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-03-17 |
| Train End | 2024-03-21 |
| Test Start | 2024-04-08 |
| Test End | 2024-07-08 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2540 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.3364 |
| Log Loss | 0.7233 |
| Baseline Accuracy | 0.7460 |
| Decision Threshold | 0.9775 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5052 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[16, 0], [47, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.5733 |
| p25 | 0.8046 |
| median | 0.8760 |
| p75 | 0.9234 |
| max | 0.9650 |
| mean | 0.8540 |
| std | 0.0850 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_26 | 328 | 1664.9826 | 0.1651 |
| 2 | volatility_60 | 297 | 1499.3787 | 0.1486 |
| 3 | ema_12 | 178 | 866.7884 | 0.0859 |
| 4 | volatility_20 | 234 | 818.0624 | 0.0811 |
| 5 | return_60 | 244 | 662.0355 | 0.0656 |
| 6 | macd_signal | 180 | 555.2250 | 0.0550 |
| 7 | macd_hist | 158 | 379.2355 | 0.0376 |
| 8 | ma_gap_60 | 199 | 362.4967 | 0.0359 |
| 9 | return_20 | 138 | 351.6762 | 0.0349 |
| 10 | volatility_10 | 134 | 312.7348 | 0.0310 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-06-16 |
| Train End | 2024-06-21 |
| Test Start | 2024-07-09 |
| Test End | 2024-10-04 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5431 |
| Precision | 0.5862 |
| Recall | 0.5000 |
| F1 | 0.5397 |
| ROC-AUC | 0.6430 |
| Log Loss | 0.8393 |
| Baseline Accuracy | 0.5397 |
| Decision Threshold | 0.7880 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6099 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[17, 12], [17, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.3534 |
| p25 | 0.5920 |
| median | 0.7713 |
| p75 | 0.9285 |
| max | 0.9731 |
| mean | 0.7368 |
| std | 0.1926 |
| count_ge_threshold | 29 |
| count_lt_threshold | 34 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 369 | 1491.0268 | 0.1491 |
| 2 | ema_26 | 307 | 1357.0732 | 0.1357 |
| 3 | ema_12 | 207 | 1070.1314 | 0.1070 |
| 4 | macd_signal | 206 | 810.9739 | 0.0811 |
| 5 | return_60 | 170 | 598.7557 | 0.0599 |
| 6 | return_20 | 209 | 594.7765 | 0.0595 |
| 7 | macd_hist | 176 | 545.0385 | 0.0545 |
| 8 | volatility_20 | 177 | 481.3773 | 0.0481 |
| 9 | macd | 132 | 410.0409 | 0.0410 |
| 10 | ma_gap_60 | 144 | 289.7227 | 0.0290 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-09-16 |
| Train End | 2024-09-20 |
| Test Start | 2024-10-07 |
| Test End | 2025-01-06 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7143 |
| Balanced Accuracy | 0.7122 |
| Precision | 0.7826 |
| Recall | 0.5806 |
| F1 | 0.6667 |
| ROC-AUC | 0.8206 |
| Log Loss | 0.5369 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.7768 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5499 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[27, 5], [13, 18]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0802 |
| p25 | 0.2981 |
| median | 0.7048 |
| p75 | 0.8314 |
| max | 0.9752 |
| mean | 0.6002 |
| std | 0.2853 |
| count_ge_threshold | 23 |
| count_lt_threshold | 40 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 307 | 1408.0938 | 0.1424 |
| 2 | ema_12 | 260 | 1025.8160 | 0.1037 |
| 3 | ema_26 | 289 | 1022.4225 | 0.1034 |
| 4 | macd_signal | 254 | 939.5456 | 0.0950 |
| 5 | return_60 | 193 | 781.1915 | 0.0790 |
| 6 | return_20 | 193 | 596.0535 | 0.0603 |
| 7 | macd_hist | 212 | 488.3402 | 0.0494 |
| 8 | volatility_20 | 150 | 467.6631 | 0.0473 |
| 9 | ma_gap_60 | 124 | 455.3316 | 0.0460 |
| 10 | macd | 124 | 301.8472 | 0.0305 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-12-15 |
| Train End | 2024-12-19 |
| Test Start | 2025-01-07 |
| Test End | 2025-04-08 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4603 |
| Balanced Accuracy | 0.6047 |
| Precision | 0.3704 |
| Recall | 1.0000 |
| F1 | 0.5405 |
| ROC-AUC | 0.7151 |
| Log Loss | 2.0816 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.8381 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6728 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 34], [0, 20]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.2466 |
| p25 | 0.9259 |
| median | 0.9681 |
| p75 | 0.9834 |
| max | 0.9984 |
| mean | 0.9100 |
| std | 0.1563 |
| count_ge_threshold | 54 |
| count_lt_threshold | 9 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 319 | 1541.6052 | 0.1484 |
| 2 | volatility_60 | 361 | 1538.8129 | 0.1481 |
| 3 | ema_26 | 236 | 1018.8678 | 0.0981 |
| 4 | return_20 | 307 | 877.6673 | 0.0845 |
| 5 | ma_gap_120 | 202 | 660.8497 | 0.0636 |
| 6 | macd_signal | 224 | 646.2746 | 0.0622 |
| 7 | return_60 | 232 | 608.3445 | 0.0585 |
| 8 | macd_hist | 159 | 530.2953 | 0.0510 |
| 9 | volatility_20 | 185 | 453.5887 | 0.0437 |
| 10 | volatility_10 | 224 | 382.6819 | 0.0368 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-03-20 |
| Train End | 2025-03-25 |
| Test Start | 2025-04-09 |
| Test End | 2025-07-10 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4762 |
| Balanced Accuracy | 0.5670 |
| Precision | 0.8846 |
| Recall | 0.4340 |
| F1 | 0.5823 |
| ROC-AUC | 0.4981 |
| Log Loss | 1.6237 |
| Baseline Accuracy | 0.8413 |
| Decision Threshold | 0.4510 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6161 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 3], [30, 23]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0018 |
| p25 | 0.0573 |
| median | 0.2983 |
| p75 | 0.7301 |
| max | 0.9442 |
| mean | 0.3948 |
| std | 0.3438 |
| count_ge_threshold | 26 |
| count_lt_threshold | 37 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 373 | 1675.5033 | 0.1609 |
| 2 | volatility_60 | 324 | 1097.1390 | 0.1054 |
| 3 | volatility_20 | 289 | 1021.9255 | 0.0982 |
| 4 | ema_26 | 313 | 952.4419 | 0.0915 |
| 5 | macd_signal | 255 | 730.9342 | 0.0702 |
| 6 | return_60 | 304 | 683.4144 | 0.0656 |
| 7 | return_20 | 224 | 549.2779 | 0.0528 |
| 8 | ma_gap_120 | 156 | 439.4320 | 0.0422 |
| 9 | return_10 | 178 | 400.8357 | 0.0385 |
| 10 | ma_gap_60 | 98 | 377.1294 | 0.0362 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-06-21 |
| Train End | 2025-06-25 |
| Test Start | 2025-07-11 |
| Test End | 2025-10-08 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3651 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5304 |
| Log Loss | 2.5732 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.9891 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5130 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[23, 0], [40, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0028 |
| p25 | 0.0097 |
| median | 0.0160 |
| p75 | 0.0263 |
| max | 0.0787 |
| mean | 0.0224 |
| std | 0.0185 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 415 | 1752.6616 | 0.1659 |
| 2 | ema_26 | 434 | 1460.8272 | 0.1382 |
| 3 | volatility_60 | 287 | 1096.0367 | 0.1037 |
| 4 | return_60 | 342 | 992.5646 | 0.0939 |
| 5 | volatility_20 | 286 | 808.7859 | 0.0765 |
| 6 | return_20 | 173 | 485.5727 | 0.0459 |
| 7 | macd_hist | 146 | 439.5243 | 0.0416 |
| 8 | macd_signal | 189 | 426.2912 | 0.0403 |
| 9 | volume_z_20 | 131 | 355.0351 | 0.0336 |
| 10 | ma_gap_120 | 149 | 327.7615 | 0.0310 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-09-21 |
| Train End | 2025-09-24 |
| Test Start | 2025-10-09 |
| Test End | 2026-01-08 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5397 |
| Balanced Accuracy | 0.5338 |
| Precision | 0.5273 |
| Recall | 0.9062 |
| F1 | 0.6667 |
| ROC-AUC | 0.4083 |
| Log Loss | 1.0885 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.4209 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5934 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[5, 26], [3, 29]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0834 |
| p25 | 0.6498 |
| median | 0.8010 |
| p75 | 0.9069 |
| max | 0.9849 |
| mean | 0.7320 |
| std | 0.2159 |
| count_ge_threshold | 55 |
| count_lt_threshold | 8 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 496 | 1985.0131 | 0.1930 |
| 2 | volatility_60 | 338 | 1065.8022 | 0.1036 |
| 3 | return_60 | 272 | 813.7823 | 0.0791 |
| 4 | ema_26 | 214 | 726.2904 | 0.0706 |
| 5 | macd_signal | 212 | 685.9523 | 0.0667 |
| 6 | volatility_10 | 177 | 478.3754 | 0.0465 |
| 7 | ma_gap_60 | 181 | 413.5421 | 0.0402 |
| 8 | volume_z_20 | 119 | 406.4100 | 0.0395 |
| 9 | return_20 | 149 | 395.4904 | 0.0385 |
| 10 | high_low_range | 162 | 355.8609 | 0.0346 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-12-20 |
| Train End | 2025-12-23 |
| Test Start | 2026-01-09 |
| Test End | 2026-04-10 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3333 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.3333 |
| Recall | 1.0000 |
| F1 | 0.5000 |
| ROC-AUC | 0.3639 |
| Log Loss | 1.3237 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.0157 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5371 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 42], [0, 21]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0201 |
| p25 | 0.4071 |
| median | 0.6552 |
| p75 | 0.8610 |
| max | 0.9685 |
| mean | 0.6095 |
| std | 0.2789 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ema_12 | 468 | 1776.7957 | 0.1764 |
| 2 | ema_26 | 356 | 1119.4575 | 0.1111 |
| 3 | volatility_60 | 304 | 993.6189 | 0.0986 |
| 4 | return_60 | 216 | 851.8092 | 0.0846 |
| 5 | volatility_20 | 224 | 678.8402 | 0.0674 |
| 6 | macd_hist | 185 | 519.0885 | 0.0515 |
| 7 | macd_signal | 175 | 424.0789 | 0.0421 |
| 8 | return_10 | 168 | 375.9771 | 0.0373 |
| 9 | volume_change_20 | 126 | 330.4758 | 0.0328 |
| 10 | macd | 124 | 329.1488 | 0.0327 |
