# Direction Validation Report: ^GSPC

## Prediction Summary
| Item | Value |
| --- | --- |
| Ticker | ^GSPC |
| Model | LightGBM Direction Classifier |
| Generated At | 2026-07-08 15:21:27 NDT |
| Last Date | 2026-07-08 |
| Last Close | 7473.6802 |
| Horizon Days | 10 |
| Predicted Direction | MODEL_INVALID |
| Signal | HOLD |
| Probability Up | 0.9314 |
| Probability Down | 0.0686 |
| Decision Threshold | 0.9649 |
| Model Valid | False |
| Validation Method | purged_rolling_walk_forward |
| Target Return Threshold | 0.0000 |
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
| fold_count | 29 | N/A | N/A | N/A |
| total_fold_count | 29 | N/A | N/A | N/A |
| beats_baseline_fold_ratio | 0.1724 | N/A | N/A | N/A |
| confusion_matrix_sum | [[268, 374], [464, 721]] | N/A | N/A | N/A |
| accuracy | 0.5413 | 0.1808 | 0.0952 | 0.9048 |
| balanced_accuracy | 0.5486 | 0.0713 | 0.4537 | 0.7076 |
| precision | 0.6066 | 0.3025 | 0.0000 | 1.0000 |
| recall | 0.6380 | 0.3884 | 0.0000 | 1.0000 |
| f1 | 0.5492 | 0.2957 | 0.0000 | 0.9500 |
| roc_auc | 0.6065 | 0.1263 | 0.3759 | 0.8655 |
| log_loss | 1.3193 | 0.6593 | 0.3137 | 2.8756 |
| baseline_accuracy | 0.6962 | 0.1248 | 0.5079 | 0.9206 |
| decision_threshold | 0.5283 | 0.3660 | 0.0103 | 0.9892 |
| calibration_score | 0.5785 | 0.0815 | 0.4821 | 0.7781 |

## Final Model Top Features
| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 313 | 1100.6270 | 0.2101 |
| 2 | return_60 | 247 | 730.8780 | 0.1395 |
| 3 | return_5 | 258 | 402.8572 | 0.0769 |
| 4 | volatility_20 | 290 | 359.9612 | 0.0687 |
| 5 | log_return_1 | 155 | 313.7311 | 0.0599 |
| 6 | ma_gap_120 | 169 | 277.1810 | 0.0529 |
| 7 | macd_signal_to_close | 130 | 233.9503 | 0.0447 |
| 8 | macd_hist_to_close | 133 | 228.4793 | 0.0436 |
| 9 | return_20 | 147 | 203.1960 | 0.0388 |
| 10 | ema_12_to_ema_26 | 140 | 162.2836 | 0.0310 |

## Fold Overview
| Fold | Status | Test Start | Test End | Acc | BalAcc | AUC | Baseline | Threshold | Threshold Search |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ok | 2019-01-14 | 2019-04-12 | 0.0952 | 0.5000 | 0.8655 | 0.9048 | 0.8931 | ok |
| 2 | ok | 2019-04-15 | 2019-07-15 | 0.6190 | 0.5000 | 0.6154 | 0.6190 | 0.0232 | ok |
| 3 | ok | 2019-07-16 | 2019-10-11 | 0.4921 | 0.4844 | 0.5716 | 0.5079 | 0.9844 | ok |
| 4 | ok | 2019-10-14 | 2020-01-13 | 0.9048 | 0.5000 | 0.5702 | 0.9048 | 0.0129 | ok |
| 5 | ok | 2020-01-14 | 2020-04-14 | 0.5238 | 0.5076 | 0.5641 | 0.5238 | 0.2632 | ok |
| 6 | ok | 2020-04-15 | 2020-07-14 | 0.7937 | 0.5000 | 0.7523 | 0.7937 | 0.0103 | ok |
| 7 | ok | 2020-07-15 | 2020-10-12 | 0.4127 | 0.5795 | 0.8098 | 0.6984 | 0.9679 | ok |
| 8 | ok | 2020-10-13 | 2021-01-12 | 0.6349 | 0.4630 | 0.4794 | 0.8571 | 0.1019 | ok |
| 9 | ok | 2021-01-13 | 2021-04-14 | 0.6667 | 0.7016 | 0.7476 | 0.6984 | 0.2484 | ok |
| 10 | ok | 2021-04-15 | 2021-07-14 | 0.7619 | 0.5917 | 0.8181 | 0.7619 | 0.4405 | ok |
| 11 | ok | 2021-07-15 | 2021-10-12 | 0.6032 | 0.5192 | 0.3909 | 0.5873 | 0.4281 | ok |
| 12 | ok | 2021-10-13 | 2022-01-11 | 0.6190 | 0.5385 | 0.6133 | 0.5873 | 0.5000 | ok |
| 13 | ok | 2022-01-12 | 2022-04-12 | 0.5873 | 0.6303 | 0.6729 | 0.6508 | 0.9892 | ok |
| 14 | ok | 2022-04-13 | 2022-07-14 | 0.4444 | 0.5000 | 0.5194 | 0.5556 | 0.0639 | ok |
| 15 | ok | 2022-07-15 | 2022-10-12 | 0.5238 | 0.6250 | 0.6870 | 0.6349 | 0.6652 | ok |
| 16 | ok | 2022-10-13 | 2023-01-12 | 0.6032 | 0.5357 | 0.6440 | 0.6667 | 0.5144 | ok |
| 17 | ok | 2023-01-13 | 2023-04-14 | 0.3968 | 0.5000 | 0.4705 | 0.6032 | 0.9638 | ok |
| 18 | ok | 2023-04-17 | 2023-07-19 | 0.7778 | 0.4537 | 0.4815 | 0.8571 | 0.0224 | ok |
| 19 | ok | 2023-07-20 | 2023-10-17 | 0.5238 | 0.6500 | 0.7519 | 0.7143 | 0.0132 | ok |
| 20 | ok | 2023-10-18 | 2024-01-18 | 0.4444 | 0.6250 | 0.5969 | 0.8889 | 0.2198 | ok |
| 21 | ok | 2024-01-19 | 2024-04-18 | 0.6825 | 0.5000 | 0.6221 | 0.6825 | 0.5000 | ok |
| 22 | ok | 2024-04-19 | 2024-07-19 | 0.2857 | 0.5312 | 0.6097 | 0.7619 | 0.9628 | ok |
| 23 | ok | 2024-07-22 | 2024-10-17 | 0.4127 | 0.4645 | 0.4723 | 0.6508 | 0.9232 | ok |
| 24 | ok | 2024-10-18 | 2025-01-21 | 0.7460 | 0.7076 | 0.7098 | 0.6349 | 0.8955 | ok |
| 25 | ok | 2025-01-22 | 2025-04-22 | 0.3968 | 0.5000 | 0.5789 | 0.6032 | 0.6758 | ok |
| 26 | ok | 2025-04-23 | 2025-07-23 | 0.3651 | 0.6552 | 0.6586 | 0.9206 | 0.9824 | ok |
| 27 | ok | 2025-07-24 | 2025-10-21 | 0.1746 | 0.5000 | 0.3759 | 0.8254 | 0.9825 | ok |
| 28 | ok | 2025-10-22 | 2026-01-22 | 0.6508 | 0.6111 | 0.5031 | 0.5714 | 0.3867 | ok |
| 29 | ok | 2026-01-23 | 2026-04-23 | 0.5556 | 0.5348 | 0.4354 | 0.5238 | 0.6863 | ok |

## Fold Details

### Fold 1

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2016-12-27 |
| Train End | 2018-12-27 |
| Test Start | 2019-01-14 |
| Test End | 2019-04-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.0952 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.8655 |
| Log Loss | 2.1159 |
| Baseline Accuracy | 0.9048 |
| Decision Threshold | 0.8931 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5467 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 0], [57, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0059 |
| p25 | 0.0304 |
| median | 0.0874 |
| p75 | 0.2471 |
| max | 0.7105 |
| mean | 0.1566 |
| std | 0.1620 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 444 | 1529.1543 | 0.2878 |
| 2 | ma_gap_120 | 228 | 636.8487 | 0.1198 |
| 3 | ma_gap_60 | 183 | 483.2630 | 0.0909 |
| 4 | ridge_pred_future_log_return | 184 | 343.7634 | 0.0647 |
| 5 | volume_change_20 | 154 | 259.4256 | 0.0488 |
| 6 | return_60 | 83 | 235.7208 | 0.0444 |
| 7 | volume_change_5 | 254 | 229.8867 | 0.0433 |
| 8 | rsi_14 | 65 | 178.9588 | 0.0337 |
| 9 | return_20 | 165 | 175.6396 | 0.0331 |
| 10 | macd_hist_to_close | 110 | 149.2205 | 0.0281 |

### Fold 2

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-03-29 |
| Train End | 2019-03-29 |
| Test Start | 2019-04-15 |
| Test End | 2019-07-15 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.6190 |
| Recall | 1.0000 |
| F1 | 0.7647 |
| ROC-AUC | 0.6154 |
| Log Loss | 0.9365 |
| Baseline Accuracy | 0.6190 |
| Decision Threshold | 0.0232 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5176 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 24], [0, 39]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1426 |
| p25 | 0.4813 |
| median | 0.8510 |
| p75 | 0.9812 |
| max | 0.9967 |
| mean | 0.7293 |
| std | 0.2640 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 452 | 1460.5373 | 0.2722 |
| 2 | ma_gap_60 | 167 | 598.3589 | 0.1115 |
| 3 | ma_gap_120 | 230 | 589.2199 | 0.1098 |
| 4 | return_60 | 137 | 410.1569 | 0.0764 |
| 5 | volume_change_20 | 205 | 325.9634 | 0.0608 |
| 6 | volatility_20 | 81 | 305.0975 | 0.0569 |
| 7 | volume_change_5 | 258 | 203.5139 | 0.0379 |
| 8 | volume_z_20 | 158 | 168.0727 | 0.0313 |
| 9 | macd_signal_to_close | 92 | 154.7399 | 0.0288 |
| 10 | macd_hist_to_close | 140 | 150.4018 | 0.0280 |

### Fold 3

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-06-28 |
| Train End | 2019-06-28 |
| Test Start | 2019-07-16 |
| Test End | 2019-10-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4921 |
| Balanced Accuracy | 0.4844 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.5716 |
| Log Loss | 1.0973 |
| Baseline Accuracy | 0.5079 |
| Decision Threshold | 0.9844 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.4821 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[31, 1], [31, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0154 |
| p25 | 0.3228 |
| median | 0.7982 |
| p75 | 0.9425 |
| max | 0.9851 |
| mean | 0.6446 |
| std | 0.3416 |
| count_ge_threshold | 1 |
| count_lt_threshold | 62 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 463 | 1069.5379 | 0.2123 |
| 2 | volatility_20 | 182 | 512.1616 | 0.1017 |
| 3 | ridge_pred_future_log_return | 241 | 416.0010 | 0.0826 |
| 4 | volume_change_20 | 220 | 386.5004 | 0.0767 |
| 5 | macd_hist_to_close | 194 | 364.5624 | 0.0724 |
| 6 | macd_signal_to_close | 157 | 323.9964 | 0.0643 |
| 7 | ma_gap_60 | 177 | 287.4153 | 0.0571 |
| 8 | ma_gap_120 | 122 | 226.0742 | 0.0449 |
| 9 | return_5 | 69 | 132.1602 | 0.0262 |
| 10 | volume_change_5 | 75 | 103.7178 | 0.0206 |

### Fold 4

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-09-27 |
| Train End | 2019-09-27 |
| Test Start | 2019-10-14 |
| Test End | 2020-01-13 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.9048 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.9048 |
| Recall | 1.0000 |
| F1 | 0.9500 |
| ROC-AUC | 0.5702 |
| Log Loss | 0.6708 |
| Baseline Accuracy | 0.9048 |
| Decision Threshold | 0.0129 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5000 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 6], [0, 57]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0293 |
| p25 | 0.4027 |
| median | 0.6724 |
| p75 | 0.9132 |
| max | 0.9920 |
| mean | 0.6451 |
| std | 0.2906 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 427 | 1185.1532 | 0.2326 |
| 2 | macd_signal_to_close | 175 | 515.5855 | 0.1012 |
| 3 | ma_gap_120 | 188 | 386.5089 | 0.0758 |
| 4 | volume_change_20 | 169 | 300.4388 | 0.0590 |
| 5 | volatility_20 | 96 | 290.8063 | 0.0571 |
| 6 | macd_hist_to_close | 149 | 245.3353 | 0.0481 |
| 7 | ridge_pred_future_log_return | 110 | 206.4340 | 0.0405 |
| 8 | close_to_ema_26 | 54 | 202.5171 | 0.0397 |
| 9 | ema_12_to_ema_26 | 132 | 197.2539 | 0.0387 |
| 10 | return_10 | 101 | 185.2633 | 0.0364 |

### Fold 5

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2017-12-27 |
| Train End | 2019-12-27 |
| Test Start | 2020-01-14 |
| Test End | 2020-04-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.5076 |
| Precision | 0.5283 |
| Recall | 0.8485 |
| F1 | 0.6512 |
| ROC-AUC | 0.5641 |
| Log Loss | 1.2453 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.2632 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5843 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[5, 25], [5, 28]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0113 |
| p25 | 0.6318 |
| median | 0.8459 |
| p75 | 0.9668 |
| max | 0.9941 |
| mean | 0.7148 |
| std | 0.3261 |
| count_ge_threshold | 53 |
| count_lt_threshold | 10 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 380 | 1059.7798 | 0.2019 |
| 2 | volatility_20 | 227 | 716.2313 | 0.1364 |
| 3 | rsi_14 | 84 | 396.3248 | 0.0755 |
| 4 | ma_gap_60 | 91 | 377.0795 | 0.0718 |
| 5 | volume_change_20 | 142 | 280.4380 | 0.0534 |
| 6 | ridge_pred_future_log_return | 143 | 265.8011 | 0.0506 |
| 7 | ema_12_to_ema_26 | 181 | 235.5647 | 0.0449 |
| 8 | return_60 | 187 | 226.5301 | 0.0432 |
| 9 | ma_gap_120 | 141 | 205.3180 | 0.0391 |
| 10 | macd_signal_to_close | 97 | 178.2228 | 0.0339 |

### Fold 6

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-03-29 |
| Train End | 2020-03-30 |
| Test Start | 2020-04-15 |
| Test End | 2020-07-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7937 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.7937 |
| Recall | 1.0000 |
| F1 | 0.8850 |
| ROC-AUC | 0.7523 |
| Log Loss | 0.5295 |
| Baseline Accuracy | 0.7937 |
| Decision Threshold | 0.0103 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.4831 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 13], [0, 50]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.7349 |
| p25 | 0.8793 |
| median | 0.9528 |
| p75 | 0.9747 |
| max | 0.9943 |
| mean | 0.9259 |
| std | 0.0662 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 433 | 1194.5522 | 0.2275 |
| 2 | return_60 | 254 | 557.6398 | 0.1062 |
| 3 | volatility_20 | 234 | 490.9096 | 0.0935 |
| 4 | ridge_pred_future_log_return | 183 | 479.3935 | 0.0913 |
| 5 | volume_change_20 | 277 | 403.7059 | 0.0769 |
| 6 | ma_gap_20 | 129 | 225.3538 | 0.0429 |
| 7 | return_5 | 156 | 216.3016 | 0.0412 |
| 8 | ma_gap_60 | 61 | 189.6893 | 0.0361 |
| 9 | high_low_range | 133 | 139.5790 | 0.0266 |
| 10 | close_to_ema_26 | 66 | 130.9651 | 0.0249 |

### Fold 7

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-06-28 |
| Train End | 2020-06-29 |
| Test Start | 2020-07-15 |
| Test End | 2020-10-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4127 |
| Balanced Accuracy | 0.5795 |
| Precision | 1.0000 |
| Recall | 0.1591 |
| F1 | 0.2745 |
| ROC-AUC | 0.8098 |
| Log Loss | 0.4959 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.9679 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5963 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[19, 0], [37, 7]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1442 |
| p25 | 0.4650 |
| median | 0.7650 |
| p75 | 0.9142 |
| max | 0.9903 |
| mean | 0.6845 |
| std | 0.2578 |
| count_ge_threshold | 7 |
| count_lt_threshold | 56 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 344 | 856.7636 | 0.1581 |
| 2 | ma_gap_120 | 295 | 665.2396 | 0.1228 |
| 3 | volatility_60 | 295 | 607.1284 | 0.1121 |
| 4 | volume_change_20 | 223 | 483.3956 | 0.0892 |
| 5 | ridge_pred_future_log_return | 213 | 387.5692 | 0.0715 |
| 6 | macd_hist_to_close | 140 | 292.4410 | 0.0540 |
| 7 | rsi_14 | 169 | 263.7443 | 0.0487 |
| 8 | macd_signal_to_close | 93 | 248.5981 | 0.0459 |
| 9 | ma_gap_60 | 191 | 225.6759 | 0.0417 |
| 10 | return_20 | 166 | 197.5046 | 0.0365 |

### Fold 8

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-09-27 |
| Train End | 2020-09-28 |
| Test Start | 2020-10-13 |
| Test End | 2021-01-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6349 |
| Balanced Accuracy | 0.4630 |
| Precision | 0.8444 |
| Recall | 0.7037 |
| F1 | 0.7677 |
| ROC-AUC | 0.4794 |
| Log Loss | 1.6469 |
| Baseline Accuracy | 0.8571 |
| Decision Threshold | 0.1019 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6458 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[2, 7], [16, 38]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0173 |
| p25 | 0.0964 |
| median | 0.1509 |
| p75 | 0.3070 |
| max | 0.9558 |
| mean | 0.2303 |
| std | 0.2011 |
| count_ge_threshold | 45 |
| count_lt_threshold | 18 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 245 | 679.2663 | 0.1326 |
| 2 | macd_hist_to_close | 208 | 637.9278 | 0.1246 |
| 3 | volume_change_20 | 216 | 400.6639 | 0.0782 |
| 4 | return_60 | 260 | 397.0467 | 0.0775 |
| 5 | volatility_20 | 174 | 332.1707 | 0.0649 |
| 6 | volatility_60 | 185 | 318.1749 | 0.0621 |
| 7 | ma_gap_60 | 152 | 303.1985 | 0.0592 |
| 8 | macd_signal_to_close | 171 | 241.7839 | 0.0472 |
| 9 | close_to_high | 146 | 198.6417 | 0.0388 |
| 10 | volume_change_5 | 159 | 178.5073 | 0.0349 |

### Fold 9

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2018-12-28 |
| Train End | 2020-12-28 |
| Test Start | 2021-01-13 |
| Test End | 2021-04-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6667 |
| Balanced Accuracy | 0.7016 |
| Precision | 0.8710 |
| Recall | 0.6136 |
| F1 | 0.7200 |
| ROC-AUC | 0.7476 |
| Log Loss | 1.0170 |
| Baseline Accuracy | 0.6984 |
| Decision Threshold | 0.2484 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7492 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[15, 4], [17, 27]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0166 |
| p25 | 0.0852 |
| median | 0.2435 |
| p75 | 0.4515 |
| max | 0.9190 |
| mean | 0.3038 |
| std | 0.2380 |
| count_ge_threshold | 31 |
| count_lt_threshold | 32 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 273 | 734.7934 | 0.1419 |
| 2 | ma_gap_120 | 262 | 656.9121 | 0.1268 |
| 3 | volatility_60 | 279 | 653.5385 | 0.1262 |
| 4 | macd_hist_to_close | 206 | 461.9159 | 0.0892 |
| 5 | volume_change_20 | 159 | 406.1190 | 0.0784 |
| 6 | volume_z_20 | 200 | 330.5010 | 0.0638 |
| 7 | close_to_low | 166 | 212.9443 | 0.0411 |
| 8 | volatility_20 | 82 | 165.0646 | 0.0319 |
| 9 | return_10 | 80 | 151.7845 | 0.0293 |
| 10 | ema_12_to_ema_26 | 72 | 145.8757 | 0.0282 |

### Fold 10

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-04-01 |
| Train End | 2021-03-30 |
| Test Start | 2021-04-15 |
| Test End | 2021-07-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7619 |
| Balanced Accuracy | 0.5917 |
| Precision | 0.8000 |
| Recall | 0.9167 |
| F1 | 0.8544 |
| ROC-AUC | 0.8181 |
| Log Loss | 0.4773 |
| Baseline Accuracy | 0.7619 |
| Decision Threshold | 0.4405 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7781 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[4, 11], [4, 44]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0386 |
| p25 | 0.7170 |
| median | 0.9035 |
| p75 | 0.9734 |
| max | 0.9960 |
| mean | 0.7978 |
| std | 0.2532 |
| count_ge_threshold | 55 |
| count_lt_threshold | 8 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 311 | 649.7693 | 0.1208 |
| 2 | volatility_60 | 227 | 594.1110 | 0.1104 |
| 3 | ema_12_to_ema_26 | 141 | 441.3601 | 0.0820 |
| 4 | volume_z_20 | 139 | 409.5340 | 0.0761 |
| 5 | return_60 | 193 | 403.3374 | 0.0750 |
| 6 | macd_signal_to_close | 218 | 394.6849 | 0.0734 |
| 7 | volume_change_20 | 222 | 380.5346 | 0.0707 |
| 8 | rsi_14 | 66 | 340.9886 | 0.0634 |
| 9 | return_10 | 76 | 268.3864 | 0.0499 |
| 10 | volatility_20 | 82 | 175.0681 | 0.0325 |

### Fold 11

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-07-01 |
| Train End | 2021-06-29 |
| Test Start | 2021-07-15 |
| Test End | 2021-10-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5192 |
| Precision | 0.5968 |
| Recall | 1.0000 |
| F1 | 0.7475 |
| ROC-AUC | 0.3909 |
| Log Loss | 1.1297 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.4281 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7185 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[1, 25], [0, 37]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.1536 |
| p25 | 0.6488 |
| median | 0.8415 |
| p75 | 0.9112 |
| max | 0.9954 |
| mean | 0.7834 |
| std | 0.1763 |
| count_ge_threshold | 62 |
| count_lt_threshold | 1 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 338 | 893.2203 | 0.1661 |
| 2 | ema_12_to_ema_26 | 217 | 677.9468 | 0.1261 |
| 3 | ma_gap_120 | 283 | 550.5384 | 0.1024 |
| 4 | macd_signal_to_close | 183 | 461.0259 | 0.0857 |
| 5 | macd_hist_to_close | 180 | 390.0442 | 0.0725 |
| 6 | return_60 | 218 | 318.0667 | 0.0591 |
| 7 | volume_z_20 | 122 | 299.3337 | 0.0557 |
| 8 | rsi_14 | 98 | 204.7383 | 0.0381 |
| 9 | close_to_low | 157 | 187.8955 | 0.0349 |
| 10 | return_10 | 57 | 185.0864 | 0.0344 |

### Fold 12

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-09-30 |
| Train End | 2021-09-28 |
| Test Start | 2021-10-13 |
| Test End | 2022-01-11 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6190 |
| Balanced Accuracy | 0.5385 |
| Precision | 0.6066 |
| Recall | 1.0000 |
| F1 | 0.7551 |
| ROC-AUC | 0.6133 |
| Log Loss | 1.0324 |
| Baseline Accuracy | 0.5873 |
| Decision Threshold | 0.5000 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5808 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[2, 24], [0, 37]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.3533 |
| p25 | 0.8131 |
| median | 0.9292 |
| p75 | 0.9704 |
| max | 0.9945 |
| mean | 0.8642 |
| std | 0.1468 |
| count_ge_threshold | 61 |
| count_lt_threshold | 2 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 327 | 804.2573 | 0.1482 |
| 2 | ma_gap_120 | 248 | 650.7775 | 0.1199 |
| 3 | ema_12_to_ema_26 | 209 | 626.2201 | 0.1154 |
| 4 | return_60 | 79 | 442.4916 | 0.0815 |
| 5 | volatility_20 | 255 | 426.0075 | 0.0785 |
| 6 | close_to_ema_26 | 128 | 313.6175 | 0.0578 |
| 7 | volume_z_20 | 184 | 311.0414 | 0.0573 |
| 8 | ridge_pred_future_log_return | 141 | 179.7110 | 0.0331 |
| 9 | macd_signal_to_close | 80 | 161.2092 | 0.0297 |
| 10 | return_10 | 152 | 142.7080 | 0.0263 |

### Fold 13

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2019-12-30 |
| Train End | 2021-12-28 |
| Test Start | 2022-01-12 |
| Test End | 2022-04-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5873 |
| Balanced Accuracy | 0.6303 |
| Precision | 0.4474 |
| Recall | 0.7727 |
| F1 | 0.5667 |
| ROC-AUC | 0.6729 |
| Log Loss | 2.8756 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.9892 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5302 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[20, 21], [5, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.9085 |
| p25 | 0.9807 |
| median | 0.9920 |
| p75 | 0.9949 |
| max | 0.9984 |
| mean | 0.9832 |
| std | 0.0199 |
| count_ge_threshold | 38 |
| count_lt_threshold | 25 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | ma_gap_120 | 350 | 1243.5686 | 0.2284 |
| 2 | volatility_60 | 345 | 929.2777 | 0.1706 |
| 3 | volatility_20 | 316 | 721.1160 | 0.1324 |
| 4 | volume_z_20 | 169 | 329.0063 | 0.0604 |
| 5 | ema_12_to_ema_26 | 195 | 274.6813 | 0.0504 |
| 6 | return_60 | 71 | 223.3982 | 0.0410 |
| 7 | return_10 | 146 | 200.0347 | 0.0367 |
| 8 | return_20 | 73 | 175.4971 | 0.0322 |
| 9 | ridge_pred_future_log_return | 104 | 153.9383 | 0.0283 |
| 10 | return_5 | 156 | 134.2216 | 0.0246 |

### Fold 14

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-03-31 |
| Train End | 2022-03-29 |
| Test Start | 2022-04-13 |
| Test End | 2022-07-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.4444 |
| Recall | 1.0000 |
| F1 | 0.6154 |
| ROC-AUC | 0.5194 |
| Log Loss | 1.7350 |
| Baseline Accuracy | 0.5556 |
| Decision Threshold | 0.0639 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5000 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 35], [0, 28]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.6066 |
| p25 | 0.9371 |
| median | 0.9636 |
| p75 | 0.9745 |
| max | 0.9955 |
| mean | 0.9351 |
| std | 0.0754 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 386 | 1164.2799 | 0.2204 |
| 2 | ma_gap_120 | 270 | 465.4066 | 0.0881 |
| 3 | volatility_20 | 232 | 412.2350 | 0.0780 |
| 4 | ridge_pred_future_log_return | 132 | 349.0183 | 0.0661 |
| 5 | return_10 | 93 | 312.9701 | 0.0592 |
| 6 | ema_12_to_ema_26 | 157 | 264.9723 | 0.0502 |
| 7 | volume_change_20 | 225 | 256.2829 | 0.0485 |
| 8 | return_20 | 172 | 247.3098 | 0.0468 |
| 9 | return_60 | 124 | 235.7199 | 0.0446 |
| 10 | macd_signal_to_close | 154 | 198.7972 | 0.0376 |

### Fold 15

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-06-30 |
| Train End | 2022-06-29 |
| Test Start | 2022-07-15 |
| Test End | 2022-10-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.6250 |
| Precision | 0.4340 |
| Recall | 1.0000 |
| F1 | 0.6053 |
| ROC-AUC | 0.6870 |
| Log Loss | 1.5876 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.6652 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5398 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[10, 30], [0, 23]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.3522 |
| p25 | 0.8549 |
| median | 0.9473 |
| p75 | 0.9778 |
| max | 0.9980 |
| mean | 0.8710 |
| std | 0.1613 |
| count_ge_threshold | 53 |
| count_lt_threshold | 10 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 438 | 1105.9425 | 0.2118 |
| 2 | volatility_20 | 232 | 418.6286 | 0.0802 |
| 3 | ma_gap_120 | 116 | 405.9107 | 0.0777 |
| 4 | volume_change_20 | 214 | 346.2345 | 0.0663 |
| 5 | ma_gap_60 | 78 | 276.1543 | 0.0529 |
| 6 | macd_signal_to_close | 172 | 245.1661 | 0.0470 |
| 7 | ema_12_to_ema_26 | 135 | 223.1902 | 0.0427 |
| 8 | return_20 | 148 | 222.1752 | 0.0426 |
| 9 | ma_gap_5 | 118 | 218.4759 | 0.0418 |
| 10 | return_2 | 181 | 215.2350 | 0.0412 |

### Fold 16

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-09-29 |
| Train End | 2022-09-28 |
| Test Start | 2022-10-13 |
| Test End | 2023-01-12 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6032 |
| Balanced Accuracy | 0.5357 |
| Precision | 0.6889 |
| Recall | 0.7381 |
| F1 | 0.7126 |
| ROC-AUC | 0.6440 |
| Log Loss | 0.6637 |
| Baseline Accuracy | 0.6667 |
| Decision Threshold | 0.5144 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5959 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[7, 14], [11, 31]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.2007 |
| p25 | 0.4920 |
| median | 0.6409 |
| p75 | 0.7847 |
| max | 0.9261 |
| mean | 0.6154 |
| std | 0.1962 |
| count_ge_threshold | 45 |
| count_lt_threshold | 18 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 427 | 1129.6094 | 0.2234 |
| 2 | volatility_20 | 158 | 665.1682 | 0.1315 |
| 3 | macd_signal_to_close | 226 | 504.9929 | 0.0999 |
| 4 | volume_change_20 | 195 | 326.0696 | 0.0645 |
| 5 | ridge_pred_future_log_return | 139 | 289.4766 | 0.0572 |
| 6 | ma_gap_60 | 126 | 222.5032 | 0.0440 |
| 7 | volume_change_5 | 148 | 217.4075 | 0.0430 |
| 8 | macd_hist_to_close | 112 | 170.8577 | 0.0338 |
| 9 | log_return_1 | 103 | 148.6393 | 0.0294 |
| 10 | return_10 | 101 | 148.2915 | 0.0293 |

### Fold 17

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2020-12-29 |
| Train End | 2022-12-28 |
| Test Start | 2023-01-13 |
| Test End | 2023-04-14 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3968 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.4705 |
| Log Loss | 1.3968 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.9638 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5154 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[25, 0], [38, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0018 |
| p25 | 0.0617 |
| median | 0.2063 |
| p75 | 0.4644 |
| max | 0.8032 |
| mean | 0.2798 |
| std | 0.2368 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 559 | 1287.5877 | 0.2487 |
| 2 | volume_change_20 | 220 | 572.1100 | 0.1105 |
| 3 | volatility_20 | 245 | 532.5670 | 0.1029 |
| 4 | ma_gap_60 | 124 | 469.5905 | 0.0907 |
| 5 | return_60 | 168 | 297.9105 | 0.0575 |
| 6 | ridge_pred_future_log_return | 175 | 253.3715 | 0.0489 |
| 7 | macd_hist_to_close | 152 | 197.4961 | 0.0381 |
| 8 | ema_12_to_ema_26 | 78 | 175.3785 | 0.0339 |
| 9 | macd_signal_to_close | 154 | 145.3187 | 0.0281 |
| 10 | log_return_1 | 149 | 138.5705 | 0.0268 |

### Fold 18

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-03-31 |
| Train End | 2023-03-30 |
| Test Start | 2023-04-17 |
| Test End | 2023-07-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7778 |
| Balanced Accuracy | 0.4537 |
| Precision | 0.8448 |
| Recall | 0.9074 |
| F1 | 0.8750 |
| ROC-AUC | 0.4815 |
| Log Loss | 1.5894 |
| Baseline Accuracy | 0.8571 |
| Decision Threshold | 0.0224 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5497 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 9], [5, 49]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0144 |
| p25 | 0.0562 |
| median | 0.1927 |
| p75 | 0.6631 |
| max | 0.9814 |
| mean | 0.3452 |
| std | 0.3196 |
| count_ge_threshold | 58 |
| count_lt_threshold | 5 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 371 | 1203.6363 | 0.2278 |
| 2 | return_60 | 376 | 920.1546 | 0.1742 |
| 3 | ridge_pred_future_log_return | 232 | 558.7335 | 0.1058 |
| 4 | volume_change_20 | 285 | 475.5916 | 0.0900 |
| 5 | volatility_20 | 180 | 310.1630 | 0.0587 |
| 6 | ema_12_to_ema_26 | 205 | 271.6785 | 0.0514 |
| 7 | ma_gap_60 | 174 | 228.8810 | 0.0433 |
| 8 | macd_signal_to_close | 180 | 206.3493 | 0.0391 |
| 9 | macd_hist_to_close | 101 | 137.3335 | 0.0260 |
| 10 | log_return_1 | 156 | 124.9146 | 0.0236 |

### Fold 19

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-06-30 |
| Train End | 2023-07-05 |
| Test Start | 2023-07-20 |
| Test End | 2023-10-17 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5238 |
| Balanced Accuracy | 0.6500 |
| Precision | 0.3696 |
| Recall | 0.9444 |
| F1 | 0.5312 |
| ROC-AUC | 0.7519 |
| Log Loss | 0.8329 |
| Baseline Accuracy | 0.7143 |
| Decision Threshold | 0.0132 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.7326 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[16, 29], [1, 17]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0029 |
| p25 | 0.0130 |
| median | 0.0213 |
| p75 | 0.0706 |
| max | 0.3550 |
| mean | 0.0590 |
| std | 0.0757 |
| count_ge_threshold | 46 |
| count_lt_threshold | 17 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 279 | 1196.1008 | 0.2292 |
| 2 | volatility_60 | 276 | 747.0047 | 0.1432 |
| 3 | ma_gap_120 | 209 | 375.5767 | 0.0720 |
| 4 | macd_hist_to_close | 168 | 331.0914 | 0.0635 |
| 5 | volatility_20 | 214 | 303.1310 | 0.0581 |
| 6 | ma_gap_60 | 183 | 299.6747 | 0.0574 |
| 7 | volume_change_20 | 278 | 247.2292 | 0.0474 |
| 8 | macd_signal_to_close | 103 | 233.8228 | 0.0448 |
| 9 | volume_z_20 | 145 | 231.1009 | 0.0443 |
| 10 | return_2 | 110 | 146.8700 | 0.0281 |

### Fold 20

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-09-29 |
| Train End | 2023-10-03 |
| Test Start | 2023-10-18 |
| Test End | 2024-01-18 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4444 |
| Balanced Accuracy | 0.6250 |
| Precision | 0.9565 |
| Recall | 0.3929 |
| F1 | 0.5570 |
| ROC-AUC | 0.5969 |
| Log Loss | 2.3453 |
| Baseline Accuracy | 0.8889 |
| Decision Threshold | 0.2198 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5603 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[6, 1], [34, 22]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0022 |
| p25 | 0.0123 |
| median | 0.0397 |
| p75 | 0.8790 |
| max | 0.9964 |
| mean | 0.3297 |
| std | 0.4196 |
| count_ge_threshold | 23 |
| count_lt_threshold | 40 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 409 | 1076.6934 | 0.2041 |
| 2 | volatility_60 | 286 | 607.5183 | 0.1152 |
| 3 | macd_hist_to_close | 178 | 580.1283 | 0.1100 |
| 4 | volatility_20 | 201 | 344.9461 | 0.0654 |
| 5 | return_60 | 128 | 292.0122 | 0.0554 |
| 6 | ridge_pred_future_log_return | 215 | 288.8928 | 0.0548 |
| 7 | return_10 | 166 | 278.2123 | 0.0527 |
| 8 | ma_gap_120 | 173 | 263.3743 | 0.0499 |
| 9 | volume_change_20 | 194 | 180.3953 | 0.0342 |
| 10 | ma_gap_60 | 161 | 179.6418 | 0.0341 |

### Fold 21

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2021-12-29 |
| Train End | 2024-01-03 |
| Test Start | 2024-01-19 |
| Test End | 2024-04-18 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6825 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.6825 |
| Recall | 1.0000 |
| F1 | 0.8113 |
| ROC-AUC | 0.6221 |
| Log Loss | 1.0398 |
| Baseline Accuracy | 0.6825 |
| Decision Threshold | 0.5000 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5000 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 20], [0, 43]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.6322 |
| p25 | 0.9379 |
| median | 0.9677 |
| p75 | 0.9848 |
| max | 0.9979 |
| mean | 0.9472 |
| std | 0.0595 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | macd_signal_to_close | 249 | 1007.7482 | 0.1907 |
| 2 | volatility_60 | 329 | 836.7609 | 0.1583 |
| 3 | macd_hist_to_close | 305 | 804.7849 | 0.1523 |
| 4 | volatility_20 | 228 | 491.5595 | 0.0930 |
| 5 | ma_gap_120 | 207 | 285.7198 | 0.0541 |
| 6 | return_60 | 168 | 215.2270 | 0.0407 |
| 7 | volume_z_20 | 120 | 158.7909 | 0.0300 |
| 8 | return_2 | 138 | 143.4667 | 0.0271 |
| 9 | ma_gap_20 | 79 | 132.1463 | 0.0250 |
| 10 | volume_change_20 | 153 | 123.1308 | 0.0233 |

### Fold 22

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-03-30 |
| Train End | 2024-04-04 |
| Test Start | 2024-04-19 |
| Test End | 2024-07-19 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.2857 |
| Balanced Accuracy | 0.5312 |
| Precision | 1.0000 |
| Recall | 0.0625 |
| F1 | 0.1176 |
| ROC-AUC | 0.6097 |
| Log Loss | 2.0221 |
| Baseline Accuracy | 0.7619 |
| Decision Threshold | 0.9628 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5051 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[15, 0], [45, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0023 |
| p25 | 0.0147 |
| median | 0.0302 |
| p75 | 0.7177 |
| max | 0.9841 |
| mean | 0.2986 |
| std | 0.3866 |
| count_ge_threshold | 3 |
| count_lt_threshold | 60 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 217 | 964.6457 | 0.1798 |
| 2 | return_60 | 359 | 637.1951 | 0.1188 |
| 3 | macd_signal_to_close | 205 | 574.8867 | 0.1072 |
| 4 | ridge_pred_future_log_return | 349 | 492.7601 | 0.0919 |
| 5 | return_20 | 129 | 366.5855 | 0.0683 |
| 6 | ema_12_to_ema_26 | 149 | 350.0769 | 0.0653 |
| 7 | ma_gap_120 | 245 | 326.8983 | 0.0609 |
| 8 | macd_hist_to_close | 182 | 315.5057 | 0.0588 |
| 9 | volatility_20 | 107 | 185.7127 | 0.0346 |
| 10 | volume_change_20 | 148 | 170.0249 | 0.0317 |

### Fold 23

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-06-30 |
| Train End | 2024-07-05 |
| Test Start | 2024-07-22 |
| Test End | 2024-10-17 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.4127 |
| Balanced Accuracy | 0.4645 |
| Precision | 0.6000 |
| Recall | 0.2927 |
| F1 | 0.3934 |
| ROC-AUC | 0.4723 |
| Log Loss | 1.5355 |
| Baseline Accuracy | 0.6508 |
| Decision Threshold | 0.9232 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5442 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[14, 8], [29, 12]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0096 |
| p25 | 0.1005 |
| median | 0.3298 |
| p75 | 0.9733 |
| max | 0.9979 |
| mean | 0.4717 |
| std | 0.3914 |
| count_ge_threshold | 20 |
| count_lt_threshold | 43 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 473 | 1516.0777 | 0.2798 |
| 2 | return_20 | 200 | 550.9625 | 0.1017 |
| 3 | macd_signal_to_close | 165 | 474.4051 | 0.0876 |
| 4 | return_60 | 289 | 380.8732 | 0.0703 |
| 5 | ridge_pred_future_log_return | 318 | 333.1895 | 0.0615 |
| 6 | ma_gap_120 | 243 | 314.6857 | 0.0581 |
| 7 | ma_gap_60 | 79 | 199.2238 | 0.0368 |
| 8 | close_to_ema_12 | 106 | 181.3417 | 0.0335 |
| 9 | macd_hist_to_close | 94 | 177.6378 | 0.0328 |
| 10 | rsi_14 | 53 | 135.4224 | 0.0250 |

### Fold 24

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-09-29 |
| Train End | 2024-10-03 |
| Test Start | 2024-10-18 |
| Test End | 2025-01-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.7460 |
| Balanced Accuracy | 0.7076 |
| Precision | 0.7727 |
| Recall | 0.8500 |
| F1 | 0.8095 |
| ROC-AUC | 0.7098 |
| Log Loss | 1.0270 |
| Baseline Accuracy | 0.6349 |
| Decision Threshold | 0.8955 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5659 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[13, 10], [6, 34]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0027 |
| p25 | 0.5891 |
| median | 0.9722 |
| p75 | 0.9937 |
| max | 0.9997 |
| mean | 0.7527 |
| std | 0.3798 |
| count_ge_threshold | 44 |
| count_lt_threshold | 19 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 547 | 1219.8351 | 0.2250 |
| 2 | volatility_20 | 262 | 754.4698 | 0.1391 |
| 3 | ma_gap_60 | 149 | 635.9444 | 0.1173 |
| 4 | return_60 | 207 | 393.4667 | 0.0726 |
| 5 | ridge_pred_future_log_return | 240 | 369.2950 | 0.0681 |
| 6 | return_20 | 148 | 294.0739 | 0.0542 |
| 7 | macd_hist_to_close | 159 | 252.1002 | 0.0465 |
| 8 | close_to_ema_12 | 125 | 185.6706 | 0.0342 |
| 9 | ma_gap_20 | 125 | 173.7584 | 0.0320 |
| 10 | macd_signal_to_close | 120 | 154.9423 | 0.0286 |

### Fold 25

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2022-12-29 |
| Train End | 2025-01-03 |
| Test Start | 2025-01-22 |
| Test End | 2025-04-22 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3968 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.3968 |
| Recall | 1.0000 |
| F1 | 0.5682 |
| ROC-AUC | 0.5789 |
| Log Loss | 2.7853 |
| Baseline Accuracy | 0.6032 |
| Decision Threshold | 0.6758 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6622 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[0, 38], [0, 25]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.9488 |
| p25 | 0.9817 |
| median | 0.9912 |
| p75 | 0.9962 |
| max | 0.9988 |
| mean | 0.9863 |
| std | 0.0130 |
| count_ge_threshold | 63 |
| count_lt_threshold | 0 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 303 | 966.1880 | 0.1785 |
| 2 | return_60 | 361 | 809.8193 | 0.1496 |
| 3 | volatility_20 | 375 | 636.4544 | 0.1176 |
| 4 | ridge_pred_future_log_return | 202 | 506.4673 | 0.0936 |
| 5 | macd_signal_to_close | 111 | 361.5614 | 0.0668 |
| 6 | return_20 | 202 | 357.8717 | 0.0661 |
| 7 | rsi_14 | 88 | 275.1713 | 0.0508 |
| 8 | ma_gap_120 | 173 | 198.2195 | 0.0366 |
| 9 | ma_gap_60 | 111 | 178.0196 | 0.0329 |
| 10 | volume_change_20 | 163 | 148.1840 | 0.0274 |

### Fold 26

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-03-31 |
| Train End | 2025-04-07 |
| Test Start | 2025-04-23 |
| Test End | 2025-07-23 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.3651 |
| Balanced Accuracy | 0.6552 |
| Precision | 1.0000 |
| Recall | 0.3103 |
| F1 | 0.4737 |
| ROC-AUC | 0.6586 |
| Log Loss | 0.3137 |
| Baseline Accuracy | 0.9206 |
| Decision Threshold | 0.9824 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5218 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[5, 0], [40, 18]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.4032 |
| p25 | 0.8273 |
| median | 0.9265 |
| p75 | 0.9879 |
| max | 0.9996 |
| mean | 0.8691 |
| std | 0.1574 |
| count_ge_threshold | 18 |
| count_lt_threshold | 45 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 418 | 1228.1910 | 0.2258 |
| 2 | volatility_20 | 244 | 744.6405 | 0.1369 |
| 3 | ma_gap_60 | 179 | 553.5143 | 0.1018 |
| 4 | volatility_60 | 273 | 472.6142 | 0.0869 |
| 5 | ridge_pred_future_log_return | 225 | 305.9069 | 0.0562 |
| 6 | macd_signal_to_close | 153 | 302.3671 | 0.0556 |
| 7 | macd_hist_to_close | 205 | 198.9033 | 0.0366 |
| 8 | volume_z_20 | 110 | 180.9040 | 0.0333 |
| 9 | ma_gap_120 | 113 | 154.6754 | 0.0284 |
| 10 | high_low_range | 140 | 116.4403 | 0.0214 |

### Fold 27

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-07-06 |
| Train End | 2025-07-09 |
| Test Start | 2025-07-24 |
| Test End | 2025-10-21 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.1746 |
| Balanced Accuracy | 0.5000 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| ROC-AUC | 0.3759 |
| Log Loss | 2.0096 |
| Baseline Accuracy | 0.8254 |
| Decision Threshold | 0.9825 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5774 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[11, 0], [52, 0]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0144 |
| p25 | 0.0402 |
| median | 0.1033 |
| p75 | 0.2904 |
| max | 0.9398 |
| mean | 0.1998 |
| std | 0.2363 |
| count_ge_threshold | 0 |
| count_lt_threshold | 63 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 260 | 1312.1246 | 0.2461 |
| 2 | volatility_60 | 326 | 548.1169 | 0.1028 |
| 3 | macd_signal_to_close | 239 | 538.7667 | 0.1011 |
| 4 | volatility_20 | 164 | 408.4495 | 0.0766 |
| 5 | ridge_pred_future_log_return | 238 | 317.3028 | 0.0595 |
| 6 | ma_gap_120 | 177 | 228.4413 | 0.0429 |
| 7 | macd_hist_to_close | 144 | 215.5302 | 0.0404 |
| 8 | close_to_ema_26 | 119 | 206.7800 | 0.0388 |
| 9 | volume_z_20 | 137 | 200.6027 | 0.0376 |
| 10 | volume_change_20 | 79 | 189.9193 | 0.0356 |

### Fold 28

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2023-10-04 |
| Train End | 2025-10-07 |
| Test Start | 2025-10-22 |
| Test End | 2026-01-22 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.6508 |
| Balanced Accuracy | 0.6111 |
| Precision | 0.6400 |
| Recall | 0.8889 |
| F1 | 0.7442 |
| ROC-AUC | 0.5031 |
| Log Loss | 0.8637 |
| Baseline Accuracy | 0.5714 |
| Decision Threshold | 0.3867 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.6559 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[9, 18], [4, 32]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0098 |
| p25 | 0.4898 |
| median | 0.7705 |
| p75 | 0.8879 |
| max | 0.9546 |
| mean | 0.6417 |
| std | 0.2973 |
| count_ge_threshold | 50 |
| count_lt_threshold | 13 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | return_60 | 323 | 1059.1608 | 0.2056 |
| 2 | macd_signal_to_close | 308 | 818.6444 | 0.1589 |
| 3 | ridge_pred_future_log_return | 273 | 418.7516 | 0.0813 |
| 4 | volatility_60 | 291 | 401.6641 | 0.0780 |
| 5 | ma_gap_120 | 147 | 283.0468 | 0.0550 |
| 6 | close_to_ema_26 | 105 | 219.7517 | 0.0427 |
| 7 | volatility_20 | 136 | 186.8028 | 0.0363 |
| 8 | volume_z_20 | 80 | 173.4124 | 0.0337 |
| 9 | volume_change_20 | 119 | 170.8727 | 0.0332 |
| 10 | return_20 | 85 | 146.4817 | 0.0284 |

### Fold 29

| Item | Value |
| --- | --- |
| Status | ok |
| Train Start | 2024-01-04 |
| Train End | 2026-01-07 |
| Test Start | 2026-01-23 |
| Test End | 2026-04-23 |
| Train Size | 504 |
| Test Size | 63 |
| Purge Size | 10 |
| Accuracy | 0.5556 |
| Balanced Accuracy | 0.5348 |
| Precision | 0.7500 |
| Recall | 0.1000 |
| F1 | 0.1765 |
| ROC-AUC | 0.4354 |
| Log Loss | 1.2410 |
| Baseline Accuracy | 0.5238 |
| Decision Threshold | 0.6863 |
| Threshold Search Metric | balanced_accuracy |
| Threshold Search Status | ok |
| Calibration Score | 0.5370 |
| Model Backend | LightGBM Direction Classifier |
| Model Warning | N/A |
| Confusion Matrix | [[32, 1], [27, 3]] |

Probability Summary

| Item | Value |
| --- | --- |
| min | 0.0098 |
| p25 | 0.0511 |
| median | 0.1522 |
| p75 | 0.2288 |
| max | 0.9375 |
| mean | 0.2016 |
| std | 0.2226 |
| count_ge_threshold | 4 |
| count_lt_threshold | 59 |
| sample_size | 63 |

Top Features

| Rank | Feature | Split Importance | Gain Importance | Gain Share |
| --- | --- | --- | --- | --- |
| 1 | volatility_60 | 450 | 1492.6632 | 0.2816 |
| 2 | macd_signal_to_close | 288 | 604.3125 | 0.1140 |
| 3 | volatility_20 | 255 | 549.1681 | 0.1036 |
| 4 | return_60 | 257 | 543.4879 | 0.1025 |
| 5 | ridge_pred_future_log_return | 170 | 250.7152 | 0.0473 |
| 6 | ma_gap_120 | 98 | 200.3408 | 0.0378 |
| 7 | close_to_ema_12 | 58 | 190.4034 | 0.0359 |
| 8 | ma_gap_60 | 98 | 188.2628 | 0.0355 |
| 9 | volume_z_20 | 104 | 168.5451 | 0.0318 |
| 10 | return_5 | 91 | 105.5669 | 0.0199 |
