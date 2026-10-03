# Preserved result breakdown / 保存结果分解

Generated from frozen CSVs; mean individual-run metrics unless explicitly marked ensemble. / 从冻结 CSV 生成；除案例与混淆外均为单次种子均值。

## Dataset means / 分库均值

| Model | Dataset | Epochs | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| A0 | ISRUC-III | 1678 | 64.44% | 0.5755 |
| A0 | MIT-BIH | 1331 | 70.35% | 0.5063 |
| A1 | ISRUC-III | 1678 | 65.67% | 0.6055 |
| A1 | MIT-BIH | 1331 | 69.17% | 0.5019 |
| A6 | ISRUC-III | 1678 | 66.41% | 0.6069 |
| A6 | MIT-BIH | 1331 | 60.88% | 0.4796 |

## Subject means / 分人均值

| Model | Subject | Epochs | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| A0 | ISRUC-III-04 | 764 | 54.14% | 0.5130 |
| A0 | ISRUC-III-05 | 914 | 73.05% | 0.6121 |
| A0 | MIT-slp02 | 621 | 68.92% | 0.3796 |
| A0 | MIT-slp60 | 710 | 71.60% | 0.4623 |
| A1 | ISRUC-III-04 | 764 | 54.54% | 0.5339 |
| A1 | ISRUC-III-05 | 914 | 74.98% | 0.6552 |
| A1 | MIT-slp02 | 621 | 65.43% | 0.3524 |
| A1 | MIT-slp60 | 710 | 72.44% | 0.5125 |
| A6 | ISRUC-III-04 | 764 | 53.40% | 0.5184 |
| A6 | ISRUC-III-05 | 914 | 77.28% | 0.6699 |
| A6 | MIT-slp02 | 621 | 69.08% | 0.4264 |
| A6 | MIT-slp60 | 710 | 53.71% | 0.4084 |

## Largest off-diagonal errors per true stage / 每个真实阶段最大的错误去向

Ensembles; percentage denominator is all epochs of the true class. / 概率平均集成；分母为该真实阶段的全部窗口。

| Model | True | Most frequent wrong prediction | Count / support | Fraction |
|---|---|---|---:|---:|
| A1 | Wake | N1 | 94 / 882 | 10.66% |
| A1 | N1 | Wake | 73 / 583 | 12.52% |
| A1 | N2 | N1 | 92 / 875 | 10.51% |
| A1 | N3 | N2 | 100 / 361 | 27.70% |
| A1 | REM | N1 | 127 / 308 | 41.23% |
| A6 | Wake | N1 | 78 / 882 | 8.84% |
| A6 | N1 | N2 | 214 / 583 | 36.71% |
| A6 | N2 | N1 | 84 / 875 | 9.60% |
| A6 | N3 | N2 | 87 / 361 | 24.10% |
| A6 | REM | N1 | 138 / 308 | 44.81% |

## Fixed cases / 固定案例

| Case | Record | Epoch (0-based) | Expert | A1 | A6 | A6 gate |
|---|---|---:|---|---|---|---:|
| both_correct | ISRUC-III-04 | 14 | Wake | Wake | Wake | 0.3008 |
| A1_only_correct | ISRUC-III-04 | 29 | N1 | N1 | Wake | 0.1985 |
| A6_only_correct | ISRUC-III-04 | 27 | Wake | N1 | Wake | 0.2184 |
| both_wrong | ISRUC-III-04 | 26 | Wake | N1 | N1 | 0.3329 |
