"""Regenerate the compact report figure from the saved training histories."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

report = Path(__file__).resolve().parent
root = report.parent / 'results'
groups = [('ResNet-50', 'task1/resnet50'), ('ViT-B/16', 'task1/vit_b_16'),
          ('ResNet-50 + SE', 'task2/resnet50_se'), ('ViT-B/16 + SE', 'task2/vit_b_16_se')]
plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7, 'xtick.labelsize': 6.5,
                     'ytick.labelsize': 6.5, 'axes.labelsize': 7})
fig, axes = plt.subplots(2, 2, figsize=(3.25, 2.55), sharex=True, sharey=True)
for ax, (title, folder) in zip(axes.flat, groups):
    history = json.loads((root / folder / 'history.json').read_text(encoding='utf-8'))
    metrics = json.loads((root / folder / 'metrics.json').read_text(encoding='utf-8'))
    assert len(history) == 300
    x = [entry['epoch'] for entry in history]
    ax.plot(x, [entry['train_loss'] for entry in history], color='#2563eb', lw=1)
    ax.plot(x, [entry['val_loss'] for entry in history], color='#c56b06', lw=1)
    best = metrics['best_epoch']
    ax.scatter([best], [history[best - 1]['val_loss']], marker='*', s=24,
               color='#b91c1c', zorder=3)
    ax.set_title(title, pad=2)
    ax.set_xticks([0, 150, 300])
    ax.grid(alpha=.18)
fig.supxlabel('Epoch', fontsize=7, y=.02)
fig.supylabel('Cross-entropy loss', fontsize=7, x=.01)
fig.legend([plt.Line2D([0], [0], color='#2563eb'),
            plt.Line2D([0], [0], color='#c56b06'),
            plt.Line2D([0], [0], marker='*', color='#b91c1c', linestyle='')],
           ['Train', 'Validation', 'Selected'], loc='upper center', ncol=3,
           fontsize=6.5, frameon=False, bbox_to_anchor=(.54, 1.01),
           handlelength=1.5, columnspacing=1)
fig.subplots_adjust(left=.13, right=.99, bottom=.14, top=.86, wspace=.13, hspace=.38)
fig.savefig(report / 'figures' / 'loss_curves_column.png', dpi=300)
plt.close(fig)
