# Task 2：分类头前 SE 门控的配对消融

## 目的与实验条件

分别比较 ResNet-50 与 ResNet-50+SE、ViT-B/16 与 ViT-B/16+SE。每对共用相同冻结骨干汇总特征、数据划分、分类头初始权重、特征标准化、优化器和模型选择规则。主要观察量为加入 SE 后的 Top-1、Top-5、Macro-F1 变化及可训练参数增量。

使用全部 1000 张图片、100 类，每类 10 张。直接读取 Task 1 的 split.json、features.pt、initial_state.pt 和基线 metrics.json，并核对数据、标签和配置。训练/验证/测试为 600/200/200，每类分别为 6/2/2。Task 1 统一使用空间/图像块均值特征，Task 2 仅训练两个 +SE 模型。输出配置记录基线配置哈希，每个模型检查点记录特征与基线指标文件哈希。

预训练权重保持为 ResNet50_Weights.IMAGENET1K_V2 与 ViT_B_16_Weights.IMAGENET1K_V1，均为 ImageNet-1K 检查点。官方文件为 resnet50-11ad3fa6.pth 与 vit_b_16-c867db91.pth，完整 SHA256、预处理和冻结记录见 results/task1/<骨干>/backbone.json。使用各自权重的确定性 RGB 预处理，最终裁剪为 224×224。每个骨干的两个分支使用完全相同的图片预处理。

## 接入位置与每一步用途

1. **冻结与推理。** 加载官方骨干，移除原 ImageNet 分类层，设置 requires_grad=False、eval() 和 inference_mode()。这样训练只作用于新接入模块。提取期间还检查骨干状态张量版本计数保持一致。
2. **ResNet 汇总。** 从 layer4 输出的最后一层特征图 [B,2048,7,7] 沿 H、W 平均，得到 [B,2048]。这保留每个通道的全局平均响应，代码为 extract_summary 的 ResNet 分支。
3. **ViT 汇总。** 将图像块序列和 CLS token 一起送入冻结 encoder，在最终 LayerNorm 输出 [B,197,768] 上排除 CLS，再对 196 个图像块 token 取平均，得到 [B,768]。CLS 参与 encoder 的注意力交互，但不参加最终均值。
4. **读取冻结特征。** 上述提取由 Task 1 完成，两个实验分支读取 Task 1 同一个 features.pt。当前实际复用了归档旧 Task 2 提取的同方式特征并重训 Task 1 基线，迁移记录在 results/task1/feature_migration.json。旧实验及日志保存在带 previous 后缀的位置。
5. **标准化。** 每个骨干仅用训练集特征计算逐维均值 μ 和总体标准差 σ，令 x=(z−μ)/max(σ,1e−6)。基线与 SE 分支使用完全相同的 μ、σ，避免测试信息参与拟合，并改善优化尺度。
6. **加入门控。** 在标准化后的汇总向量 x 与物种分类头之间接入 SE，输出 x⊙g(x)。当前实现属于在汇总特征上加入的 SE 门控，两层全连接层均带偏置，降维比例 r=16，无额外残差或 Dropout。
7. **配对训练。** Task 1 训练基线并保存初始分类头参数。Task 2 直接加载这份 initial_state.pt，SE 参数独立初始化，仅训练门控和分类头。Task 2 汇总表中的基线行直接引用 Task 1 的指标并保存 result_source。
8. **验证选择。** 两组均用 AdamW，lr=0.01、weight_decay=0.01，交叉熵，全批次 600 条特征，每轮一次更新，训练完整 300 轮。每轮记录更新前的训练损失和更新后的验证损失，选择验证交叉熵最低的模型。
9. **最终评测。** 恢复最佳模型后，在同一 200 条保留测试样本上评测一次，保存预测、混淆矩阵、分类头和完整记录。Top-1、Top-5、Macro-F1 均以百分数展示，Macro-F1 对全部 100 类等权平均。
10. **结果核验。** verify_task2.py 重新加载共享特征和保存的模型执行推理，检查预测、指标、混淆矩阵、最佳轮次、参数量、训练集标准化统计，以及配对的分类头初始化一致。它依赖保存的冻结特征进行核验。

两条路线可以概括为下列计算。

```text
ResNet layer4 特征图 ──空间平均── z [B,2048]
ViT 最终图像块 token ──token 平均── z [B,768]

每对共享 z ──训练集统计标准化── x
                               ├── Linear(D,100) ── 基线分数
                               └── x * g(x) ── Linear(D,100) ── SE 组分数

g(x) = Sigmoid(Linear(D/r,D)(ReLU(Linear(D,D/r)(x))))
```

这是一次在标准化汇总特征上添加门控的明确实现选择。门控对同一图片的各维度生成不同权重，权重会随图片而变化。Squeeze 汇总已经由冻结特征提取完成，训练阶段学习 Excitation 部分。

SE 的全局汇总和两层门控依据 [Squeeze-and-Excitation Networks, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.html)。本实验只在分类头前接入一个门控，接入位置及冻结训练方式由当前任务规定。

## 参数量

令 h=max(1,D//16)，两层带偏置线性层的 SE 参数为 D×h+h+h×D+D，即 2Dh+h+D。分类头参数为 100D+100。标准化均值和标准差保存为 buffer，参与推理但无需优化。

| 骨干 | D | h | 分类头参数 | 新增 SE 参数 | SE 组总可训练参数 |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 2048 | 128 | 204,900 | 526,464 | 731,364 |
| ViT-B/16 | 768 | 48 | 76,900 | 74,544 | 151,444 |

## 实际结果与记录

本次已实际运行 Task 1 两个基线和 Task 2 两个 +SE，每组各 300 轮。冻结特征共用已提取的 1000 张图片汇总，四组共用 600/200/200 划分，测试集共 200 张。Task 2 本次只执行两个 +SE 训练。

| 模型 | 训练部分 | 新增 SE 参数 | 总可训练参数 | Top-1 (%) | Top-5 (%) | Macro-F1 (%) | 最佳轮次 |
|---|---|---:|---:|---:|---:|---:|---:|
| ResNet-50 | 分类头 | 0 | 204,900 | 45.00 | 72.00 | 41.25 | 2 |
| ResNet-50+SE | SE 与分类头 | 526,464 | 731,364 | 51.50 | 71.50 | 46.76 | 3 |
| ViT-B/16（图像块均值） | 分类头 | 0 | 76,900 | 35.00 | 64.00 | 31.61 | 3 |
| ViT-B/16+SE（图像块均值） | SE 与分类头 | 74,544 | 151,444 | 34.50 | 60.50 | 30.13 | 3 |

| 配对 | Top-1 变化（百分点） | Top-5 变化（百分点） | Macro-F1 变化（百分点） |
|---|---:|---:|---:|
| ResNet 加入 SE | +6.50 | −0.50 | +5.51 |
| ViT 加入 SE | −0.50 | −3.50 | −1.48 |

本次 ResNet 加入 SE 后，Top-1 与 Macro-F1 提升，Top-5 略降。ViT 的三项测试指标均下降。SE 未在两条路线产生一致收益。ViT+SE 的最佳验证交叉熵略低（2.7439 对 2.7892），但测试指标降低，验证损失改善并不保证测试准确率同步提高。

当前 Task 1 ViT 基线已统一为图像块均值，Top-1 为 35.00%。此处直接引用该结果，Task 2 ViT+SE 为 34.50%，配对差值为 −0.50 个百分点。

运行产物保存在 [对比表](../results/task2/comparison.md)、summary.json、paired_deltas.json 和项目根目录 run_task2.log。最佳模型包含标准化 buffer、门控参数（如有）、分类头参数和特征元数据。

已实际执行 `python verify_task2.py`，四组保存的模型均能从共享冻结特征重现全部 Top-5 预测和指标。核验还确认每对分类头初始参数相同、标准化只使用训练集、最佳轮次符合验证损失选择规则，以及参数计数正确。核验结论保存在 results/task2/verification.txt。

## 训练曲线与同样本预测分析

四组完整 300 轮损失曲线由 history.json 生成，蓝线表示训练交叉熵，橙线表示验证交叉熵，红星表示所选最佳轮次。基线记录来自 Task 1，SE 记录来自 Task 2。

![四组完整训练曲线](figures/loss_curves_full.png)

四组训练损失快速趋近零，验证损失在第 2～3 轮最低后上升，存在过拟合。最终结果使用验证损失最低的检查点。训练损失与验证损失分别对应本轮更新前和更新后的模型，数值差异也包含数据集合与计算时点的影响。

同样本 ID 为 **3082102**，原始行号为 **395**，真实物种为 **10，Batrachoseps attenuatus**。选择该样本是为了展示两个基线给出不同 Top-1 的情况。

![同样本原图](figures/sample_3082102.png)

| 模型 | 样本 ID | 真实类别 | Top-1 预测类别 | Top-5 类别编号（降序） |
|---|---:|---|---|---|
| ResNet-50 | 3082102 | 10，Batrachoseps attenuatus | 10，Batrachoseps attenuatus | [10,82,46,31,50] |
| ResNet-50+SE | 3082102 | 10，Batrachoseps attenuatus | 10，Batrachoseps attenuatus | [10,82,46,31,50] |
| ViT-B/16 | 3082102 | 10，Batrachoseps attenuatus | 50，Lysurus mokusin | [50,10,22,82,2] |
| ViT-B/16+SE | 3082102 | 10，Batrachoseps attenuatus | 50，Lysurus mokusin | [50,10,82,22,66] |

ResNet 的两个分支均正确预测该样本，ViT 的两个分支均将类别 50 排在第一、真实类别 10 排在第二。SE 调整了 ViT 后续候选的排序，该样本的 Top-1 及 Top-5 命中情况均保持一致。原始四组记录见 [同样本预测 CSV](figures/sample_3082102_predictions.csv)。

全部测试样本上，SE 对 ResNet 的 Top-1、Macro-F1 分别带来 +6.50、+5.51 个百分点，Top-5 为 −0.50 个百分点。SE 对 ViT 三项指标的影响分别为 −0.50、−3.50、−1.48 个百分点。单样本展示与总体结果共同记录，本次 SE 的效果随骨干而异。

## 环境与解释范围

实际运行使用现有 D:\python\pythonsetup\python.exe、Python 3.9.9、PyTorch 2.7.1+cpu、torchvision 0.22.1+cpu、NumPy 2.0.2，CPU 8 线程，特征提取 batch size 16。本轮未安装新依赖，也未新建 Conda 环境，完整配置见 config.json。README 提供隔离环境的复现命令。

Task 1 与 Task 2 现已统一特征定义，SE 的效应通过 Task 1 基线与 Task 2 对应 +SE 的差值衡量。每类测试仅 2 张，当前为单种子、单划分实验。新增门控同时增加了模型容量，当前实验评估加入整个模块的效果。
