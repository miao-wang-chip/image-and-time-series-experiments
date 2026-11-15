"""Render saved losses and export a shared-sample prediction comparison."""
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from PIL import Image


def main():
    output = Path('docs/figures')
    output.mkdir(parents=True, exist_ok=True)
    experiments = [
        ('ResNet-50', Path('results/task1/resnet50')),
        ('ViT-B/16', Path('results/task1/vit_b_16')),
        ('ResNet-50+SE', Path('results/task2/resnet50_se')),
        ('ViT-B/16+SE', Path('results/task2/vit_b_16_se')),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (name, folder) in zip(axes.flat, experiments):
        history = json.loads((folder / 'history.json').read_text(encoding='utf-8'))
        metrics = json.loads((folder / 'metrics.json').read_text(encoding='utf-8'))
        assert len(history) == 300
        epochs = [entry['epoch'] for entry in history]
        axis.plot(epochs, [entry['train_loss'] for entry in history], label='Training', color='#2563eb')
        axis.plot(epochs, [entry['val_loss'] for entry in history], label='Validation', color='#d97706')
        best = metrics['best_epoch']
        axis.scatter([best], [history[best - 1]['val_loss']], marker='*', s=100, color='#b91c1c', zorder=4, label=f'Selected epoch {best}')
        axis.set_title(name)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('Cross-entropy loss')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.suptitle('Frozen features: baseline and SE training histories')
    figure.tight_layout()
    figure.savefig(output / 'loss_curves_full.png', dpi=180)
    plt.close(figure)

    sample_id = 3082102
    comparisons = []
    for name, folder in experiments:
        with (folder / 'predictions.csv').open(encoding='utf-8', newline='') as file:
            row = next(row for row in csv.DictReader(file) if int(row['sample_id']) == sample_id)
        comparisons.append({'model': name, **row})
    assert len({row['row_index'] for row in comparisons}) == 1
    assert len({row['true_label'] for row in comparisons}) == 1
    table = pq.read_table('data/train-00000-of-00001.parquet')
    sample = table.slice(int(comparisons[0]['row_index']), 1).to_pylist()[0]
    assert sample['id'] == sample_id and sample['label'] == int(comparisons[0]['true_label'])
    with Image.open(io.BytesIO(sample['image']['bytes'])) as image:
        image.convert('RGB').save(output / 'sample_3082102.png')
    with (output / 'sample_3082102_predictions.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    print('Saved full loss curves, sample image and four-model prediction record to', output)


if __name__ == '__main__':
    main()
