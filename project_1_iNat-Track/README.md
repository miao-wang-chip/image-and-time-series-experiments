# 物种分类：冻结视觉骨干与 SE 消融

**仓库文件说明：** 原始数据和预训练权重未上传。数据集来源：[evendrow/inat24_tiny（Hugging Face）](https://huggingface.co/datasets/evendrow/inat24_tiny)。运行前须自行准备 `data/train-00000-of-00001.parquet` 以及 `checkpoints/resnet50-11ad3fa6.pth`、`checkpoints/vit_b_16-c867db91.pth`。使用与已记录数据和权重哈希一致的文件，才能对应本次结果；数据哈希见 `results/task1/config.json`。

实验使用 100 类、1000 张图像，训练／验证／测试分别为 600／200／200 张；比较 ResNet-50 和 ViT-B/16 的冻结特征及分类头前的 SE 门控。

## 目录与入口

```text
project_1_iNat-Track/
├── README.md、requirements.txt
├── scripts/             训练、核验、绘图和隔离运行入口
├── configs/             已完成运行的配置副本
├── data/                原始 Parquet 数据
├── checkpoints/         两种骨干的预训练权重
├── results/             原始运行日志、指标、预测、模型及图表
│   ├── task1/、task2/   基线和 SE 消融结果及原始运行配置
│   ├── figures/         原始曲线、样本图及预测对照
│   └── previous_reports/ 旧版说明
└── technical_report/    main.tex、main.pdf、figures/ 和 review.md
```

训练入口是 `scripts/run_task1.py`、`scripts/run_task2.py`；核验入口为 `scripts/verify_results.py`、`scripts/verify_task2.py`。这些原始脚本按当前工作目录查找 `data/`、`weights/` 和 `results/`，应通过隔离运行目录使用。

## 运行与核验

在项目根目录运行：

```powershell
.\scripts\prepare_run.ps1 -Mode Prepare
# 如需训练，在隔离目录中依次训练两项任务
.\scripts\prepare_run.ps1 -Mode Train
```

准备脚本复制代码、数据、预训练权重和已有结果至新建的 `results/runs/<时间戳>/`，将项目的 `checkpoints/` 复制为原脚本所需的 `weights/`。它打印隔离目录；在该目录内运行 `python verify_results.py` 和 `python verify_task2.py` 可核验已有结果。训练、核验写入隔离目录，不覆盖主 `results/task1/` 和 `results/task2/`。默认使用 `D:\python\pythonsetup\python.exe`，可通过 `-Python` 指定解释器，依赖见 `requirements.txt`。准备一次约需 0.6 GB 额外空间。

配置副本见 `configs/task1.json`、`configs/task2.json`，实测配置原件分别位于 `results/task1/config.json`、`results/task2/config.json`；原训练脚本不直接读取副本，运行参数以脚本参数为准。

## 结果与文档

[基线对比](results/task1/comparison.md) · [SE 配对对比](results/task2/comparison.md) · [配对差值](results/task2/paired_deltas.json)。同一 200 张测试图片上，ResNet-50 基线与 SE 的 Top-1 分别为 45.00% 和 51.50%；ViT-B/16 分别为 35.00% 和 34.50%，属于单种子、小样本实验。旧特征迁移来源及旧日志未找到，现有说明见 `results/task1/feature_migration.json`。

[技术报告 PDF](technical_report/main.pdf) · [LaTeX 源文件](technical_report/main.tex) · [审查记录](technical_report/review.md)。在 `technical_report/` 以 XeLaTeX 编译 `main.tex` 三遍，需要 acmart、xeCJK 和 Windows 中文字体；`plot_loss_curves.py` 从已保存的历史重新绘制报告中的紧凑曲线。
