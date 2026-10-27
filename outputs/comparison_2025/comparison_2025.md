# 2025 test set (193 fixes, 15 systems)

| Model                                      |   RMSE |    MAE |     R2 |   Bias |   N |
|:-------------------------------------------|-------:|-------:|-------:|-------:|----:|
| Tabular MLP [physical]                     |  3.360 |  2.642 |  0.855 |  1.540 | 193 |
| Multimodal (efficientnet_b0) [physical]    |  4.105 |  3.366 |  0.783 |  1.085 | 193 |
| Multimodal (efficientnet_b0) [no_pressure] |  8.493 |  6.586 |  0.072 |  4.336 | 193 |
| Tabular MLP [no_pressure]                  | 18.740 | 12.150 | -3.518 | 10.501 | 193 |
