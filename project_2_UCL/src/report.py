# -*- coding: utf-8 -*-
"""Generate reproducible figures, example CSV and Chinese report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
NAMES={'cnn':'1D CNN','gru':'GRU','transformer':'Transformer','mamba':'Mamba'}


def main():
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    audit=json.loads((OUT/'raw_audit.json').read_text(encoding='utf-8'))
    target=np.load(OUT/'test_targets.npz')
    truth=target['truth']; origins=pd.DatetimeIndex(target['origins'])
    uid=protocol['example_user_index']; user=protocol['example_user']
    preds={k:np.load(OUT/k/'predictions.npz')['predictions'] for k in NAMES}
    table=pd.read_csv(OUT/'metrics.csv')
    assert truth.shape==(61,96,370)
    for k,p in preds.items():
        assert p.shape==truth.shape and np.isfinite(p).all()
        for step,label in [(3,'1h'),(23,'6h'),(95,'24h')]:
            calc=abs(p[:,step].astype('float64')-truth[:,step]).mean()
            assert np.isclose(calc,table.set_index('model').loc[k,f'mae_{label}_kw'])
    figdir=OUT/'figures'; figdir.mkdir(exist_ok=True)
    dates=pd.DatetimeIndex([o+pd.Timedelta(minutes=15*i) for o in origins for i in range(1,97)])
    example=pd.DataFrame({'timestamp':dates,'origin':np.repeat(origins.values,96),
                          'actual_kw':truth[:,:,uid].ravel()})
    for k,p in preds.items(): example[k+'_kw']=p[:,:,uid].ravel()
    example.to_csv(OUT/'example_predictions.csv',index=False)
    for days,filename in [(1,'forecast_day.png'),(7,'forecast_week.png')]:
        fig,ax=plt.subplots(figsize=(13,4.8))
        ax.plot(dates[:days*96],example.actual_kw[:days*96],color='black',lw=2,label='Observed')
        for k in NAMES: ax.plot(dates[:days*96],example[k+'_kw'][:days*96],lw=1.3,alpha=.85,label=NAMES[k])
        if days>1:
            for origin in origins[1:days]: ax.axvline(origin,color='gray',alpha=.2,lw=1)
        ax.set(title=f'{user} | first {days} test day(s) | daily 24-hour forecasts',ylabel='Power (kW)',xlabel='Portuguese dataset local time')
        ax.set_xlim(origins[0],origins[0]+pd.Timedelta(days=days))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(range(0,24,3))) if days==1 else mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
        ax.set_title(f'{user} | first {days} test day(s) | daily 24-hour forecasts',pad=38)
        ax.legend(ncol=5,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=9)
        ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figdir/filename,dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for k,p in preds.items():
        ax.plot(np.arange(1,97)/4,abs(p.astype('float64')-truth).mean((0,2)),label=NAMES[k])
    ax.set(xlabel='Forecast horizon (hours)',ylabel='MAE (kW)',title='370 users / 61 daily forecast origins')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figdir/'mae_horizon.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for k in NAMES:
        log=pd.read_csv(OUT/k/'training_log.csv')
        ax.plot(log.epoch,log.val_mae_kw,label=NAMES[k])
    ax.set(xlabel='Epoch',ylabel='Validation MAE (kW)',title='Checkpoint selection uses validation only')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'validation_curves.png',dpi=170);plt.close(fig)
    summaries={k:json.loads((OUT/k/'training_summary.json').read_text()) for k in NAMES}
    def mdtable(frame):
        lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
        for row in frame.itertuples(index=False,name=None):
            lines.append('| '+' | '.join(f'{x:.3f}' if isinstance(x,float) else str(x) for x in row)+' |')
        return '\n'.join(lines)
    mt=table.rename(columns={'model':'模型','mae_1h_kw':'1小时 MAE','mae_6h_kw':'6小时 MAE','mae_24h_kw':'24小时 MAE','mae_all_kw':'全天96步 MAE'})
    neural=table[table.model!='seasonal_naive'].set_index('model')
    best={c:neural[c].idxmin() for c in neural.columns}
    base=table.set_index('model').loc['seasonal_naive']
    descriptions=[]
    for k in NAMES:
        r=neural.loc[k]
        improvements=[100*(base[f'mae_{h}h_kw']-r[f'mae_{h}h_kw'])/base[f'mae_{h}h_kw'] for h in [1,6,24]]
        descriptions.append(f'- **{NAMES[k]}**：1/6/24 小时相对前一天复制基线的误差降低分别为 {improvements[0]:.2f}% / {improvements[1]:.2f}% / {improvements[2]:.2f}%（负值表示更差）；全天 MAE 为 {r.mae_all_kw:.3f} kW。')
    report=f'''# 多用户电力负荷预测实验技术报告

## 1. 实验目标与数据范围

用 370 个用户过去 24 小时的负荷，同时预测全部 370 个用户未来 24 小时负荷。单样本输入 `[96,370]`，标签/输出 `[96,370]`；批量形状 `[B,96,370]`。每步为 15 分钟，单位 kW。没有把功率除以 4，也没有按小时聚合。

例如测试首个预测起点为 2014-11-01 00:00，输入为 2014-10-31 00:15 至 2014-11-01 00:00，输出对应 2014-11-01 00:15 至 2014-11-02 00:00。这 96×370 个未来真实数值只用于评分，不输入模型。

## 2. 数据审计与实际范围

- 原始记录：{audit['rows']:,} 个时间点、{audit['users']} 个用户；{audit['first']} 至 {audit['last']}。
- 实际实验：{audit['selected_first']} 至 {audit['selected_last']}，共 {audit['selected_rows']:,} 个时间点；全部 MT_001–MT_370，无用户筛选。
- 为 CPU 计算选择 2014 年，2011–2013 年未参与训练；这是完整原数据的时间子集实验。
- 原始缺失值 {audit['missing']}；已检查跨分块时间连续、15 分钟间隔、有限数值。实验子集零值比例 {audit['zero_fraction']:.4%}。没有填补、删除零值或裁剪异常峰值。
- 训练期常数用户：{protocol['constant_train_users']}；这些用户仍保留。
- UCI 原始时间是葡萄牙本地标签；每个记录日固定96点，夏令时有特殊零值/合并记录，本实验保留原标签，不做时区重采样。
- 原文件 SHA256：`{audit['sha256']}`。

## 3. 数据集划分、历史窗口与预测窗口

### 3.1 按目标时间先后划分

将2014年数据按时间顺序切分，不随机打乱时间。以下日期区间是用于划分的**目标数据区间**，两端均包含；第一个训练日还要先提供历史，所以首个实际训练目标从1月2日00:15开始。测试时不会用目标值更新参数或选择模型。

| 数据集 | 划分用目标区间 | 样本预测起点 | 起点间隔 | 样本数 | 用途 |
|---|---|---|---|---:|---|
| 训练 | 2014-01-01 00:15 至 2014-09-01 00:00 | 2014-01-02 00:00 至 2014-08-31 00:00 | 6小时 | {protocol['split_samples']['train']} | 计算损失并更新参数 |
| 验证 | 2014-09-01 00:15 至 2014-11-01 00:00 | 2014-09-01 00:00 至 2014-10-31 00:00 | 24小时 | {protocol['split_samples']['val']} | 每轮评估，选最佳权重 |
| 测试 | 2014-11-01 00:15 至 2015-01-01 00:00 | 2014-11-01 00:00 至 2014-12-31 00:00 | 24小时 | {protocol['split_samples']['test']} | 最终比较，不更新参数 |

每个样本的全部96个目标必须落在对应区间内；时间分界及实际窗口索引可在 `results/train_windows.csv`、`val_windows.csv` 和 `test_windows.csv` 查证。训练最后一个目标是9月1日00:00，验证第一个目标是00:15；验证最后一个目标是11月1日00:00，测试第一个目标是00:15，因此不同集合的目标不重叠。

### 3.2 一个窗口具体包含什么

预测起点记为 `t`，也就是最后一个已知记录。每15分钟一个点：

```text
历史输入（96点、24小时）：t - 23小时45分，……，t
                                             ↑ 预测起点
未来标签（96点、24小时）：t + 15分，……，t + 24小时
```

单个输入和标签都是 `[96, 370]`，对应全部370个用户。四组使用同一批窗口，直接输出未来96点×370用户。例如：

| 样本 | 预测起点 t | 输入首点 → 末点 | 标签首点 → 末点 |
|---|---|---|---|
| 首个训练样本 | 2014-01-02 00:00 | 01-01 00:15 → 01-02 00:00 | 01-02 00:15 → 01-03 00:00 |
| 下一训练样本 | 2014-01-02 06:00 | 01-01 06:15 → 01-02 06:00 | 01-02 06:15 → 01-03 06:00 |
| 首个验证样本 | 2014-09-01 00:00 | 08-31 00:15 → 09-01 00:00 | 09-01 00:15 → 09-02 00:00 |
| 首个测试样本 | 2014-11-01 00:00 | 10-31 00:15 → 11-01 00:00 | 11-01 00:15 → 11-02 00:00 |

训练相邻起点相差6小时，所以相邻输入之间、相邻标签之间均重叠18小时；965个窗口不是965段独立记录。验证和测试每天午夜发起一次预测，同一集合内相邻目标窗口不重叠。允许验证或测试的历史输入跨入前一分区：它只包含预测起点及之前已知的负荷，例如首个测试输入使用10月31日历史；未来标签不作为输入，也不参与训练或模型选择。

### 3.3 处理、训练和评测过程

1. `prepare` 分块读取分号分隔、逗号小数的原文件，检查时间间隔和缺失并缓存 float32 数组。
2. 只用训练区间的数据计算每个用户的均值和标准差；标准差下限为1 kW。三组数据都使用这一组参数标准化，保存于 `scaler.npz`，没有使用验证或测试数据拟合标准化参数。
3. 四种编码器各自训练20轮，训练批次32、训练损失为标准化空间的 MAE。每轮结束后使用61个验证窗口，反标准化为 kW 后计算所有用户、所有96步的验证 MAE；只保存验证误差最低的权重。
4. 加载各自最佳权重，对同样的61个测试窗口进行一次评测，反标准化为 kW；结果及完整预测保存到 `results/`。测试不反向传播、不更新权重，也不用于选择轮次或调整超参数。

## 4. 四种模型与统一配置

公共输入投影 370→32；CNN 是两层时间卷积（核5，膨胀率1/2），GRU 是单层32维递归网络，Transformer 是单层32维、4头、64维前馈层、正弦位置编码、dropout 0.1，Mamba 是单块 Mamba-1 结构（内部64维，状态8维，因果深度卷积核4）。Transformer 只关注历史输入，历史内部双向注意力不会泄漏未来标签。

公共预测头：LayerNorm → 时间维线性映射96→96 → 用户维线性映射32→370。所有模型均预测对“前一天同一时段负荷”的修正，即 `预测 = 输入的前一天曲线 + 学到的修正`。因此本比较是四个编码器在相同日周期残差框架下的比较，而非无跳连版本。额外报告 `seasonal_naive`，直接复制前一天，不训练。

Mamba 使用项目自带纯 PyTorch CPU 实现，包含输入相关 Δ/B/C、负对角 A、状态递推、D 跳连及 SiLU 门控。递推为 `s_t = exp(Δ_t A) s_(t-1) + Δ_t B_t u_t`，`y_t = C_t s_t + D u_t`，随后门控、投影和块残差。不是用普通 RNN 替代，也不依赖 `mamba-ssm` CUDA 扩展；并未声称与官方所有优化路径逐位等价。测试比较了递推与独立闭式展开的输出和梯度，并检查因果性。

公共优化：AdamW，学习率0.001、weight_decay=0、batch32、梯度裁剪1、种子42、20轮、CPU线程4。四组训练样本顺序相同。没有超参数搜索，参数量不同。

{mdtable(pd.DataFrame([dict(模型=NAMES[k],参数量=v['parameters'],最佳轮次=v['best_epoch'],训练秒数=v['seconds'],验证MAE=v['best_val_mae_kw']) for k,v in summaries.items()]))}

## 5. MAE 口径和主结果

预测起点是最后一个历史时间戳。第1/6/24小时分别取第4/24/96步（零基索引3/23/95），不是前1/6/24小时的累计平均。每个时距的 MAE 对61个起点×370用户的绝对误差等权平均。全天 MAE 额外对96步平均。结果均在反标准化后的 kW 空间计算，不裁剪负预测。

{mdtable(mt)}

四个神经模型中，第1小时最低为 **{NAMES[best['mae_1h_kw']]}**，第6小时最低为 **{NAMES[best['mae_6h_kw']]}**，第24小时最低为 **{NAMES[best['mae_24h_kw']]}**；全天最低为 **{NAMES[best['mae_all_kw']]}**。

{chr(10).join(descriptions)}

![各预测时距误差](../results/figures/mae_horizon.png)

这里所有评测起点都是午夜；第1小时是01:00，第6小时是06:00，第24小时是下一日00:00。因此不同预测距离的误差还混合了日内时段差异，误差不必随时距单调增加：例如6小时 MAE 较小，并不能单独证明模型在更远的距离预测得更好。kW MAE 虽然每个起点和用户等权，但大功率用户可能产生数值更大的绝对误差，从而对总体均值贡献较大；各用户结果见 `per_user_metrics.csv`。

## 6. 同用户、同测试时段预测曲线与负荷高峰

展示用户 **{user}**：仅按训练数据选择，在训练期至少95%记录为正的用户中，选择训练均值中位次序用户；没有根据测试效果挑用户。图中展示固定的首个测试日。

![首日五条曲线](../results/figures/forecast_day.png)

首日真实峰值为 **{truth[0,:,uid].max():.3f} kW**，首次峰值时间为 **{dates[int(truth[0,:,uid].argmax())]}**。

从首日图看，四组都能跟随凌晨低谷、早晨上升和深夜下降，但四条预测相互接近，体现了共同的前一天残差基底。当天四组预测最大值都落在16:30，与实际最大值的时间不一致；晚间部分时段也出现共同高估。因此“日变化趋势跟得上”不等于“当天尖峰跟得准”。

### 四条路线如何结合表格和曲线理解

主表的每个数值汇总了 **61个起点、370个用户**；首日曲线只展示 **{user}一个用户**。两者回答的问题不同，不能用单个用户的图推断所有用户的峰值表现。下面的首日实际峰值是 {truth[0,:,uid].max():.3f} kW（12:15）；各模型首日预测峰值的时间都是16:30。

- **1D CNN**：短期1小时 MAE 为 {neural.loc['cnn','mae_1h_kw']:.3f} kW，四组中最低；6小时为 {neural.loc['cnn','mae_6h_kw']:.3f}，略高于 Transformer 的 {neural.loc['transformer','mae_6h_kw']:.3f}。较长的24小时为 {neural.loc['cnn','mae_24h_kw']:.3f}，也是四组最低，全天误差同样最低。对图中首日，CNN 跟住了早晨上升趋势，但预测峰值为 {preds['cnn'][0,:,uid].max():.3f} kW，且峰值时间比真实峰值晚4小时15分。
- **GRU**：1小时 MAE 为 {neural.loc['gru','mae_1h_kw']:.3f}，与 Transformer 接近；6小时为 {neural.loc['gru','mae_6h_kw']:.3f}，略差于复制前一天基线的 {base['mae_6h_kw']:.3f}。24小时为 {neural.loc['gru','mae_24h_kw']:.3f}，优于基线但高于 CNN。首日曲线同样跟随低谷和早晨上升，预测峰值 {preds['gru'][0,:,uid].max():.3f} kW 出现在16:30，未捕捉12:15的真实最高点；仅凭该图不能说 GRU 在所有用户上都错过峰值。
- **Transformer**：1小时 MAE 为 {neural.loc['transformer','mae_1h_kw']:.3f}；6小时为 {neural.loc['transformer','mae_6h_kw']:.3f}，四组中最低。24小时为 {neural.loc['transformer','mae_24h_kw']:.3f}，仅次于 CNN，全天误差也仅次于 CNN。首日预测峰值 {preds['transformer'][0,:,uid].max():.3f} kW、时间16:30；虽然总体6小时表现最好，这一用户当天的最高峰时刻仍预测偏晚。
- **Mamba**：1小时 MAE 为 {neural.loc['mamba','mae_1h_kw']:.3f}，在四组中仅次于 CNN；6小时 {neural.loc['mamba','mae_6h_kw']:.3f} 和24小时 {neural.loc['mamba','mae_24h_kw']:.3f} 均为四组中最高，6小时也高于复制基线。首日预测峰值 {preds['mamba'][0,:,uid].max():.3f} kW，仍出现在16:30。

因此，1小时和24小时总体上 CNN 领先，6小时 Transformer 领先，但差距应结合基线和日内时刻解读。四条曲线都接近前一天形状，也都在首日把最大峰预测在16:30；这一观察支持“总体趋势跟随、单日峰值时刻仍有偏差”，不能据此断定哪种架构普遍更好。

## 7. 限制、验证和复现

这是单随机种子、小模型、一天历史、2014年子集、两个月测试的实验，不能据此断言某个模型家族普遍优于其他家族。一天历史无法直接覆盖周周期；全部模型共用日周期跳连，结果可能主要受重复日模式影响。训练的标准化损失与最终 kW MAE 权重不同。未对零值上线期、夏令时和异常点做领域修复。CPU串行 Mamba 的运行时间不能代表官方 GPU 融合内核效率。

已运行三项单元测试（选择性扫描数值/梯度、Mamba因果性、四模型反向传播），训练时检查有限损失，输出形状为 `[61,96,370]`；报告生成时从保存的预测独立重算主表三项 MAE 并核对。数据审计检查全文件时间间隔和缺失。复现命令及文件结构见 README。

![验证曲线](../results/figures/validation_curves.png)

## 8. 来源

- [UCI 数据与单位/时间说明](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014)，Trindade (2015)，DOI 10.24432/C58C86，CC BY 4.0。
- [Mamba 原论文](https://arxiv.org/abs/2312.00752)。
- [官方 Mamba-1 模块](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py) 与 [选择性扫描参考公式实现](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py)。本项目按公开结构与数学递推独立实现小型CPU版本，未使用官方加速算子。
'''
    (ROOT/'docs/technical_report.md').write_text(report,encoding='utf-8')
    print(table.to_string(index=False))
    print('Report and four figures saved; metric consistency checks passed.')


if __name__=='__main__': main()
