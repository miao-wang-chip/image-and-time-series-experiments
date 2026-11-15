# 冻结 ImageNet-1K 视觉骨干对比

## 实验目的与权重

比较冻结预训练表征对本地物种分类的迁移效果，仅训练物种分类头。

| 模型 | torchvision 权重枚举 | 官方文件 | 特征维度 | 分类头可训练参数 |
|---|---|---|---:|---:|
| ResNet-50 | ResNet50_Weights.IMAGENET1K_V2 | resnet50-11ad3fa6.pth | 2048 | 204,900 |
| ViT-B/16 | ViT_B_16_Weights.IMAGENET1K_V1 | vit_b_16-c867db91.pth | 768 | 76,900 |

两者均为 ImageNet-1K 监督预训练；未使用 SWAG 权重。不同模型的版本后缀不代表相同训练配方。本实验比较指定官方检查点的迁移效果，不能将差异完全归因于架构。

官方依据：[ResNet-50](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.resnet50.html)、[ViT-B/16](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.vit_b_16.html)。实际权重完整 SHA256 记录于各模型的 `backbone.json`。

## 数据与处理

本地单个 Parquet 共 1000 条、100 类，每类 10 条。种子 42，按类别各取 6 张训练、2 张验证、2 张测试，实际规模为 600/200/200。测试为本地训练文件中保留的评测子集，并非数据源官方测试集。每条样本的索引和 ID 在 split.json 中保存。

只使用图片及 label。图片转 RGB，保持纵横比缩放后中心裁剪 224×224，并使用 ImageNet 均值和标准差。ResNet V2 短边缩放为 232，ViT V1 为 256；完整配置保存于 backbone.json。采用各检查点官方预处理，以避免人为更改权重使用条件，不使用随机增强。

骨干全部 requires_grad=False、eval()，提取时 inference_mode()；ResNet 移除 fc，ViT 移除 heads。只用训练特征计算逐维标准化统计。分类头为 Linear(D,100)，参数量 D×100+100。

当前统一特征定义为 ResNet layer4 特征图的空间均值，以及 ViT 最终 LayerNorm 后的图像块 token 均值（排除 CLS）。Task 2 直接读取这里保存的特征、划分、分类头初始参数和基线结果，仅训练 +SE。旧 CLS 实验保存在 results/task1_previous。

## 训练与评测

固定种子 42，CPU、8 线程、提取 batch size 16；AdamW，学习率 0.01，weight decay 0.01，交叉熵，全批次训练 300 轮。两组采用相同规则，各自按验证集交叉熵最小保存分类头，随后评测测试集一次。实际软件版本、配置和数据哈希见 config.json。

Top-1 为最高分预测正确率；Top-5 为真值进入最高分五类的比例；Macro-F1 为所有 100 类各自 F1 的算术平均，无真阳性时该类 F1 为 0。所有展示指标乘 100，单位为百分数。

## 实际结果

本次已完成全部 1000 张图片的两组实验，在相同 200 张图片上得到如下结果。训练部分均为物种分类头，骨干可训练参数均为 0。

| 特征提取器 | 训练部分 | 可训练参数量 | Top-1 (%) | Top-5 (%) | Macro-F1 (%) |
|---|---|---:|---:|---:|---:|
| ResNet-50 | 物种分类头 | 204,900 | 45.00 | 72.00 | 41.25 |
| ViT-B/16（图像块均值） | 物种分类头 | 76,900 | 35.00 | 64.00 | 31.61 |

ResNet 最佳轮次为第 2 轮，ViT 为第 3 轮，均从完整 300 轮的验证损失记录选出。本次复用已归档旧 Task 2 中实际提取的同方式冻结特征，并实际重训两个分类头。缓存与当前权重、数据、预处理和汇总方式匹配，来源记录见 feature_migration.json。feature_seconds 为原特征提取耗时。

当前统一汇总方式下，ResNet 的 Top-1 为 45.00%，ViT 为 35.00%。本次结果以当前图像块均值表征为准。Task 2 对应 +SE 分别为 51.50% 和 34.50%，可以直接与此处基线配对比较。

程序输出的 [对比表](../results/task1/comparison.md) 和 [完整结果](../results/task1/summary.json) 直接来自保存的测试预测。已运行 `python verify_results.py`，确认划分互斥、测试 ID 完全一致、每类测试 2 张，且逐条预测复算指标与混淆矩阵一致，验证结论见 verification.txt。

每类测试仅 2 张、总计 200 张，结果对划分较敏感；单种子实验仅代表本次划分。划分按样本进行，未依据拍摄者或重复图像分组，不能据此声称独立来源泛化。更可靠结论需要后续多种子实验或独立测试集。

## 训练曲线与同样本预测

下图上排为 Task 1 的两个基线，下排为 Task 2 对应 +SE。全部曲线直接由保存的 history.json 绘制，覆盖完整 300 轮。蓝线为训练交叉熵，橙线为验证交叉熵，红星标出验证损失最低的模型。训练损失在本轮更新前计算，验证损失在更新后计算。

![四组完整训练曲线](figures/loss_curves_full.png)

四组训练损失很快趋近零，验证损失在早期最低后明显上升，表现出过拟合。最终评测采用验证集选出的早期模型。曲线由原始记录直接连线生成。

选取样本 ID **3082102**、原始行号 **395**，真实类别为 **10，Batrachoseps attenuatus**。该样本特意选自两个基线 Top-1 预测不同的记录，用于展示差异。原图和四组预测可逐条核对。

![样本3082102](figures/sample_3082102.png)

| 模型 | 样本 ID | 真实类别 | Top-1 预测类别 | Top-1 正确 | Top-5 命中 |
|---|---:|---|---|---|---|
| ResNet-50 | 3082102 | 10，Batrachoseps attenuatus | 10，Batrachoseps attenuatus | 是 | 是 |
| ViT-B/16 | 3082102 | 10，Batrachoseps attenuatus | 50，Lysurus mokusin | 否 | 是 |

ResNet 在该样本上将真实物种排在第一，ViT 将真实类别排在第二，Top-1 选择了类别 50。该例展示两种表征的预测差异，整体优劣依据全部 200 张测试图片的指标判断。四组完整同样本记录保存在 [预测对照 CSV](figures/sample_3082102_predictions.csv)，SE 对该样本及整体测试集的影响见 Task 2 报告。
