"""Paired SE ablation on frozen spatial/token summaries. Run from project root."""
import argparse
import copy
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision
from torch import nn

from run_task1 import save, sha


class SpeciesProbe(nn.Module):
    """Train-only standardization, optional SE, then a linear classifier."""

    def __init__(self, mean, std, classes, reduction=16, use_se=False):
        super().__init__()
        dimension = mean.numel()
        hidden = max(1, dimension // reduction)
        self.register_buffer('mean', mean.clone())
        self.register_buffer('std', std.clone())
        self.gate = nn.Identity()
        if use_se:
            self.gate = nn.Sequential(
                nn.Linear(dimension, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dimension),
                nn.Sigmoid(),
            )
        self.classifier = nn.Linear(dimension, classes)
        self.use_se = use_se

    def forward(self, features):
        normalized = (features - self.mean) / self.std
        if self.use_se:
            normalized = normalized * self.gate(normalized)
        return self.classifier(normalized)


def metrics_from_top5(top5, truth, classes):
    matrix = torch.zeros(classes, classes, dtype=torch.int64)
    for target, prediction in zip(truth, top5[:, 0]):
        matrix[target, prediction] += 1
    diagonal = matrix.diag().double()
    f1 = 2 * diagonal / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
    metrics = {
        'top1_pct': 100 * (top5[:, 0] == truth).double().mean().item(),
        'top5_pct': 100 * (top5 == truth[:, None]).any(1).double().mean().item(),
        'macro_f1_pct': 100 * f1.mean().item(),
    }
    return metrics, matrix


def train_probe(features, labels, split, names, rows, folder, use_se, args, metadata):
    train, val, test = [torch.tensor(split[key]) for key in ['train', 'val', 'test']]
    mean = features[train].mean(0)
    std = features[train].std(0, unbiased=False).clamp_min(1e-6)
    # Recreate exactly the same classifier initialization in each pair.
    torch.manual_seed(args.seed)
    initial_classifier = torch.load(Path(args.baseline) / metadata['backbone'] / 'initial_state.pt', weights_only=True)
    torch.manual_seed(args.seed + 1)
    probe = SpeciesProbe(mean, std, len(names), args.reduction, use_se)
    probe.classifier.load_state_dict(initial_classifier)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), folder / 'initial_state.pt')
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=.01)
    history = []
    best_loss = float('inf')
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        probe.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(probe(features[train]), labels[train])
        loss.backward()
        optimizer.step()
        probe.eval()
        with torch.no_grad():
            val_loss = nn.functional.cross_entropy(probe(features[val]), labels[val]).item()
        history.append({'epoch': epoch, 'train_loss': loss.item(), 'val_loss': val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(probe.state_dict())
    training_seconds = time.perf_counter() - start
    probe.load_state_dict(best_state)
    probe.eval()
    with torch.no_grad():
        top5 = probe(features[test]).topk(5, dim=1).indices
    metrics, matrix = metrics_from_top5(top5, labels[test], len(names))
    gate_parameters = sum(parameter.numel() for parameter in probe.gate.parameters())
    result = {
        'model': folder.name,
        'use_se': use_se,
        'feature_dim': features.shape[1],
        'head_parameters': sum(parameter.numel() for parameter in probe.classifier.parameters()),
        'added_se_parameters': gate_parameters,
        'trainable_parameters': sum(parameter.numel() for parameter in probe.parameters()),
        **metrics,
        'best_epoch': best_epoch,
        'best_val_loss': best_loss,
        'training_seconds': training_seconds,
        'train_samples': len(train), 'val_samples': len(val), 'test_samples': len(test),
    }
    save(folder / 'metrics.json', result)
    save(folder / 'history.json', history)
    torch.save({
        'state_dict': best_state, 'use_se': use_se, 'reduction': args.reduction,
        'class_names': names, 'feature_metadata': metadata,
    }, folder / 'head.pt')
    np.savetxt(folder / 'confusion_matrix.csv', matrix.numpy(), fmt='%d', delimiter=',')
    with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['row_index', 'sample_id', 'true_label', 'true_species', 'pred_label', 'pred_species', 'top5'])
        for j, index in enumerate(split['test']):
            target = int(labels[index])
            predicted = int(top5[j, 0])
            writer.writerow([index, rows[index]['id'], target, names[target], predicted, names[predicted], json.dumps(top5[j].tolist())])
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/train-00000-of-00001.parquet')
    parser.add_argument('--baseline', default='results/task1')
    parser.add_argument('--output', default='results/task2')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reduction', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.01)
    args = parser.parse_args()
    if min(args.epochs, args.threads, args.reduction) < 1 or args.lr <= 0:
        parser.error('epochs, threads, reduction and lr must be positive')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(args.data)
    names = json.loads(table.schema.metadata[b'huggingface'])['info']['features']['label']['names']
    rows = table.to_pylist()
    labels = torch.tensor([row['label'] for row in rows])
    split_record = json.loads((Path(args.baseline) / 'split.json').read_text(encoding='utf-8'))
    split = split_record['indices']
    all_indices = [index for key in ['train', 'val', 'test'] for index in split[key]]
    assert sorted(all_indices) == list(range(len(rows))), 'Split must cover each sample exactly once'
    for key in split:
        assert [rows[i]['id'] for i in split[key]] == split_record['sample_ids'][key]
    save(out / 'split.json', split_record)
    save(out / 'class_names.json', names)
    config = {
        'args': vars(args), 'data_sha256': sha(args.data), 'split_sha256': sha(Path(args.baseline) / 'split.json'),
        'samples': len(rows), 'classes': len(names),
        'split_sizes': {key: len(value) for key, value in split.items()},
        'python': platform.python_version(), 'python_executable': sys.executable,
        'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'numpy': np.__version__, 'platform': platform.platform(), 'device': 'cpu',
        'optimizer': 'AdamW, full batch, weight_decay=0.01',
        'selection': 'minimum validation cross entropy over all epochs',
        'gate': 'standardize pooled summary -> Linear(D,D//r,bias) -> ReLU -> Linear(D//r,D,bias) -> Sigmoid; multiply standardized summary',
        'paired_controls': 'same frozen features, splits, classifier initialization, normalization, optimizer, epochs and selection',
    }
    baseline_root = Path(args.baseline)
    baseline_config = json.loads((baseline_root / 'config.json').read_text(encoding='utf-8'))
    assert baseline_config['data_sha256'] == config['data_sha256']
    assert baseline_config['args']['epochs'] == args.epochs
    assert baseline_config['seed'] == args.seed
    assert baseline_config['optimizer'] == 'AdamW lr=0.01 weight_decay=0.01; full batch' and args.lr == .01
    config['baseline_config_sha256'] = sha(baseline_root / 'config.json')
    config['baseline_root'] = str(baseline_root)
    save(out / 'config.json', config)
    results = []
    for name in ['resnet50', 'vit_b_16']:
        baseline_folder = baseline_root / name
        cache_path = baseline_folder / 'features.pt'
        cached = torch.load(cache_path, weights_only=True)
        features = cached['features']
        assert cached['data_sha256'] == config['data_sha256']
        assert torch.equal(cached['labels'], labels)
        expected_pooling = 'layer4 spatial mean over H,W' if name == 'resnet50' else 'encoder final LayerNorm output; mean of 196 patch tokens excluding CLS'
        assert cached['pooling'] == expected_pooling
        metadata = {
            'backbone': name, 'baseline_root': str(baseline_root), 'features_path': str(cache_path),
            'features_sha256': sha(cache_path),
            'baseline_metrics_sha256': sha(baseline_folder / 'metrics.json'),
            'pooling': cached['pooling'], 'weights_sha256': cached['weights_sha256'],
            'data_sha256': cached['data_sha256'],
        }
        baseline_result = json.loads((baseline_folder / 'metrics.json').read_text(encoding='utf-8'))
        baseline_result.update({
            'use_se': False, 'feature_dim': features.shape[1],
            'head_parameters': baseline_result['trainable_parameters'], 'added_se_parameters': 0,
            'result_source': str(baseline_folder / 'metrics.json'),
        })
        results.append(baseline_result)
        print(f'{name}: baseline and features loaded from Task 1; training SE only', flush=True)
        result = train_probe(features, labels, split, names, rows, out / (name + '_se'), True, args, metadata)
        result['result_source'] = str(out / (name + '_se') / 'metrics.json')
        results.append(result)
        save(out / 'summary.json', results)
    with (out / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(key for result in results for key in result)))
        writer.writeheader()
        writer.writerows(results)
    lines = ['| 模型 | 分类头参数 | 新增 SE 参数 | 总可训练参数 | Top-1 (%) | Top-5 (%) | Macro-F1 (%) |', '|---|---:|---:|---:|---:|---:|---:|']
    for result in results:
        lines.append(f"| {result['model']} | {result['head_parameters']:,} | {result['added_se_parameters']:,} | {result['trainable_parameters']:,} | {result['top1_pct']:.2f} | {result['top5_pct']:.2f} | {result['macro_f1_pct']:.2f} |")
    (out / 'comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    differences = []
    for base, enhanced in [(results[0], results[1]), (results[2], results[3])]:
        differences.append({'pair': base['model'], **{key + '_delta_pp': enhanced[key] - base[key] for key in ['top1_pct', 'top5_pct', 'macro_f1_pct']}})
    save(out / 'paired_deltas.json', differences)


if __name__ == '__main__':
    main()
