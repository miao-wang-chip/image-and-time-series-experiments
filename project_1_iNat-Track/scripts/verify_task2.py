"""Re-run saved probes on frozen summaries and audit paired controls."""
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from run_task1 import sha
from run_task2 import SpeciesProbe, metrics_from_top5


def main():
    root = Path('results/task2')
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(config['args']['threads'])
    record = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    split = record['indices']
    train, val, test = [torch.tensor(split[key]) for key in ['train', 'val', 'test']]
    assert sorted(split['train'] + split['val'] + split['test']) == list(range(config['samples']))
    rows = pq.read_table(config['args']['data'], columns=['id', 'label']).to_pylist()
    assert sha(config['args']['data']) == config['data_sha256']
    baseline_root = Path(config['baseline_root'])
    assert sha(baseline_root / 'split.json') == config['split_sha256']
    assert sha(baseline_root / 'config.json') == config['baseline_config_sha256']
    for key in split:
        assert [rows[i]['id'] for i in split[key]] == record['sample_ids'][key]
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    checks = []
    for backbone in ['resnet50', 'vit_b_16']:
        folder = baseline_root / backbone
        cache = torch.load(folder / 'features.pt', weights_only=True)
        metadata = json.loads((folder / 'backbone.json').read_text(encoding='utf-8'))
        assert metadata['trainable_backbone_parameters'] == 0
        assert cache['pooling'] == metadata['pooling']
        features, labels = cache['features'], cache['labels']
        assert torch.equal(labels, torch.tensor([row['label'] for row in rows]))
        initial_states = []
        for use_se in [False, True]:
            variant = backbone + ('_se' if use_se else '')
            output = root / variant if use_se else baseline_root / backbone
            checkpoint = torch.load(output / 'head.pt', weights_only=True)
            if use_se:
                state = checkpoint['state_dict']
                reference = checkpoint['feature_metadata']
                assert reference['features_sha256'] == sha(folder / 'features.pt')
                assert reference['baseline_metrics_sha256'] == sha(folder / 'metrics.json')
                assert reference['pooling'] == cache['pooling']
            else:
                state = {
                    'mean': checkpoint['feature_mean'], 'std': checkpoint['feature_std'],
                    'classifier.weight': checkpoint['state_dict']['weight'],
                    'classifier.bias': checkpoint['state_dict']['bias'],
                }
            assert torch.equal(state['mean'], features[train].mean(0))
            assert torch.equal(state['std'], features[train].std(0, unbiased=False).clamp_min(1e-6))
            model = SpeciesProbe(state['mean'], state['std'], config['classes'], config['args']['reduction'], use_se)
            model.load_state_dict(state)
            model.eval()
            initial = torch.load(output / 'initial_state.pt', weights_only=True)
            if not use_se:
                initial = {'classifier.weight': initial['weight'], 'classifier.bias': initial['bias'], 'mean': state['mean'], 'std': state['std']}
            initial_states.append(initial)
            with torch.no_grad():
                top = model(features[test]).topk(5, dim=1).indices
                validation_loss = torch.nn.functional.cross_entropy(model(features[val]), labels[val]).item()
            metrics, matrix = metrics_from_top5(top, labels[test], config['classes'])
            with (output / 'predictions.csv').open(encoding='utf-8', newline='') as file:
                predictions = list(csv.DictReader(file))
            assert [int(row['row_index']) for row in predictions] == split['test']
            assert [int(row['sample_id']) for row in predictions] == record['sample_ids']['test']
            assert torch.equal(top, torch.tensor([json.loads(row['top5']) for row in predictions]))
            for row, index in zip(predictions, split['test']):
                assert int(row['true_label']) == rows[index]['label']
                assert int(row['pred_label']) == json.loads(row['top5'])[0]
                assert row['true_species'] == checkpoint['class_names'][int(row['true_label'])]
                assert row['pred_species'] == checkpoint['class_names'][int(row['pred_label'])]
            assert np.array_equal(matrix.numpy(), np.loadtxt(output / 'confusion_matrix.csv', delimiter=','))
            saved = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
            summary_row = next(item for item in summary if item['model'] == variant)
            assert all(saved[key] == summary_row[key] for key in saved)
            for key, value in metrics.items():
                assert abs(saved[key] - value) < 1e-4
            assert saved['trainable_parameters'] == sum(p.numel() for p in model.parameters())
            assert summary_row['added_se_parameters'] == sum(p.numel() for p in model.gate.parameters())
            history = json.loads((output / 'history.json').read_text(encoding='utf-8'))
            assert len(history) == config['args']['epochs']
            best = min(history, key=lambda item: item['val_loss'])
            assert saved['best_epoch'] == best['epoch']
            assert abs(saved['best_val_loss'] - validation_loss) < 1e-6
            assert saved['best_val_loss'] == best['val_loss']
            checks.append(f'{variant}: PASS; predictions reproduced from saved features and head; metrics, parameters, best epoch and normalization verified')
        for key in ['classifier.weight', 'classifier.bias', 'mean', 'std']:
            assert torch.equal(initial_states[0][key], initial_states[1][key])
        checks.append(f'{backbone} pair: PASS; identical feature cache, classifier initialization, normalization and test IDs')
    (root / 'verification.txt').write_text('\n'.join(checks) + '\n', encoding='utf-8')
    print('\n'.join(checks))


if __name__ == '__main__':
    main()
