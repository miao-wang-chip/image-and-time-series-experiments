# UCI 多用户电力负荷预测

**数据集来源及仓库文件说明：** 原始数据和处理缓存未上传。运行前从 [ElectricityLoadDiagrams20112014（UCI）](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) 获取数据，将原文件放在 `data/electricityloaddiagrams20112014/LD2011_2014.txt`；若无 `data/processed.npz`，程序会在隔离运行目录生成缓存。

以 UCI ElectricityLoadDiagrams20112014 的 2014 年数据，使用 370 个用户过去 96 个 15 分钟点预测未来 96 点，比较 CNN、GRU、Transformer、Mamba 和前一天复制基线。

## 目录与入口

```text
project_2_UCL/
├── README.md、requirements.txt
├── src/                 数据处理、模型、训练、评测和报告代码
├── scripts/             隔离运行入口
├── configs/config.json  实验时间范围及模型、训练参数
├── data/                原始数据和处理后的缓存
├── tests/               模型形状、梯度与因果性测试
├── results/             原始运行记录、预测、指标、模型和图表
│   ├── metrics.csv、run.log、protocol.json、raw_audit.json
│   ├── {cnn,gru,transformer,mamba}/ 各模型结果和检查点
│   ├── figures/         预测、分时距误差和验证曲线
│   └── previous_reports/ 旧版说明
└── technical_report/    main.tex、main.pdf、figures/ 和 review.md
```

原始训练与评测入口是 `src/run.py`，图表与旧版报告入口是 `src/report.py`。它们将脚本父目录视为运行根目录，并依赖该目录下的 `data/`、`config.json` 和 `results/`；请使用隔离运行目录。

## 运行与核验

在项目根目录运行：

```powershell
.\scripts\prepare_run.ps1 -Mode Prepare
# 如需重新训练四组模型
.\scripts\prepare_run.ps1 -Mode Train
# 模型实现测试
& 'D:\python\pythonsetup\python.exe' -m unittest discover -s tests -v
```

准备脚本复制源码、数据、已有结果和 `configs/config.json` 至新建的 `results/runs/<时间戳>/`；隔离目录中的配置名为 `config.json`，保持原入口的路径约定。在脚本打印的隔离目录中可执行 `python src/run.py --evaluate-only` 和 `python src/report.py`。这些命令会覆盖隔离目录中的结果，不覆盖主 `results/`。默认使用 `D:\python\pythonsetup\python.exe`，可通过 `-Python` 指定解释器，依赖见 `requirements.txt`；准备一次约需 0.8 GB 额外空间。修改配置时间范围后需重新生成隔离目录的 `data/processed.npz`。

## 结果与文档

[主指标表](results/metrics.csv) · [运行协议](results/protocol.json) · [运行日志](results/run.log) · [预测图表](results/figures/)。1、6、24 小时 MAE 对应未来第 4、24、96 个点，单位 kW，在 61 个测试起点和 370 个用户上平均。CNN 全天 MAE 为 56.705 kW，前一天复制基线为 59.349 kW；单种子结果不能代表模型家族的普遍排名。

[技术报告 PDF](technical_report/main.pdf) · [LaTeX 源文件](technical_report/main.tex) · [审查记录](technical_report/review.md)。在 `technical_report/` 以 XeLaTeX 编译 `main.tex` 三遍，需要 acmart、xeCJK 和 Windows 中文字体。
