# 图像分类与时序预测实验

本仓库包含两个已完成的实验：比较冻结视觉骨干与 SE 门控的物种分类，以及比较多种编码器的电力负荷预测。各项目的源码、配置、实测结果和技术报告分别保存在独立目录中。

## 项目一：物种图像分类与 SE 消融

使用 [inat24_tiny](https://huggingface.co/datasets/evendrow/inat24_tiny) 的 100 类、1000 张图像，按 600/200/200 划分训练、验证和测试集。比较冻结的 ImageNet 预训练 ResNet-50 与 ViT-B/16，并在分类头前加入 SE 门控进行配对消融。四组使用相同的 200 张测试图片。

| 模型 | Top-1 (%) | Top-5 (%) | Macro-F1 (%) |
|---|---:|---:|---:|
| ResNet-50 | 45.00 | 72.00 | 41.25 |
| ResNet-50 + SE | 51.50 | 71.50 | 46.76 |
| ViT-B/16 | 35.00 | 64.00 | 31.61 |
| ViT-B/16 + SE | 34.50 | 60.50 | 30.13 |

本次加入 SE 后，ResNet-50 的 Top-1 提升 6.50 个百分点，ViT-B/16 下降 0.50 个百分点。结果来自[配对结果表](project_1_iNat-Track/results/task2/comparison.csv)，按百分数保留两位小数；更多[图表](project_1_iNat-Track/results/figures/)和[技术报告](project_1_iNat-Track/technical_report/main.pdf)见项目目录。

在 Windows PowerShell 中，从仓库根目录进入项目并准备独立运行目录：

```powershell
cd project_1_iNat-Track
.\scripts\prepare_run.ps1 -Mode Prepare
# 需要完整训练时，使用：.\scripts\prepare_run.ps1 -Mode Train
```

原训练入口为 `scripts/run_task1.py` 和 `scripts/run_task2.py`。具体依赖、数据和权重准备及核验方法见[项目一 README](project_1_iNat-Track/README.md)。

## 项目二：多用户电力负荷预测

使用 [UCI ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) 的 2014 年数据，用 370 个用户过去 96 个 15 分钟点预测未来 96 点，比较四种编码器的效果。以下 MAE 在反标准化后的 kW 空间计算，对 61 个测试起点和 370 个用户取平均；1、6、24 小时分别取第 4、24、96 个预测点，全天指标对全部 96 点平均。

| 模型 | 1 小时 MAE | 6 小时 MAE | 24 小时 MAE | 全天 MAE |
|---|---:|---:|---:|---:|
| CNN | 47.174 | 38.964 | 44.064 | 56.705 |
| GRU | 47.940 | 39.214 | 44.600 | 57.510 |
| Transformer | 47.960 | 38.727 | 44.495 | 56.859 |
| Mamba | 47.819 | 39.590 | 45.475 | 57.763 |

数值单位为 kW，越低越好，来自[完整指标表](project_2_UCL/results/metrics.csv)，保留三位小数；另见[预测图表](project_2_UCL/results/figures/)和[技术报告](project_2_UCL/technical_report/main.pdf)。

在 Windows PowerShell 中，从仓库根目录进入项目并准备独立运行目录：

```powershell
cd project_2_UCL
.\scripts\prepare_run.ps1 -Mode Prepare
# 需要完整训练时，使用：.\scripts\prepare_run.ps1 -Mode Train
```

原训练与评测入口为 `src/run.py`，绘图与旧版报告入口为 `src/report.py`。具体依赖、数据准备和评测方法见[项目二 README](project_2_UCL/README.md)。

## 数据、权重与结果解读

原始数据和视觉预训练权重体积较大，未纳入仓库；运行前须按各项目 README 准备。准备脚本在 `results/runs/` 创建隔离运行目录，避免覆盖已保存的实验结果。上述结果均为对应项目的单种子实测，不代表模型在其他数据或设置下的普遍表现。
