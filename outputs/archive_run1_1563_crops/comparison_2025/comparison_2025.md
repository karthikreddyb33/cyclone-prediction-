# 2025 test set (193 fixes, 15 systems)

| Model                                      |   RMSE |    MAE |     R2 |   Bias |   N |
|:-------------------------------------------|-------:|-------:|-------:|-------:|----:|
| Tabular MLP [physical]                     |  3.360 |  2.613 |  0.855 |  1.018 | 193 |
| Multimodal (efficientnet_b0) [physical]    |  3.880 |  3.253 |  0.806 |  1.482 | 193 |
| Multimodal (efficientnet_b0) [no_pressure] |  7.206 |  5.303 |  0.332 |  2.480 | 193 |
| Tabular MLP [no_pressure]                  | 21.777 | 13.835 | -5.101 | 11.902 | 193 |
