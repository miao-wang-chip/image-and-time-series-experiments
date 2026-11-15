"""Prepare data, train four models, evaluate saved best validation checkpoints."""
import argparse
import hashlib
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from models import Forecaster

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
FAMILIES = ['cnn', 'gru', 'transformer', 'mamba']


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def prepare(cfg):
    cache = ROOT / 'data' / 'processed.npz'
    if not cache.exists():
        source = ROOT / 'data/electricityloaddiagrams20112014/LD2011_2014.txt'
        digest = hashlib.sha256()
        with source.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
        frames, total, missing, first, last = [], 0, 0, None, None
        for chunk in pd.read_csv(source, sep=';', decimal=',', index_col=0, chunksize=8192):
            chunk.index = pd.to_datetime(chunk.index)
            assert chunk.shape[1] == 370
            assert (chunk.index.to_series().diff().dropna() == pd.Timedelta(minutes=15)).all()
            if last is not None:
                assert chunk.index[0] - last == pd.Timedelta(minutes=15)
            first = chunk.index[0] if first is None else first
            last = chunk.index[-1]
            total += len(chunk)
            missing += int(chunk.isna().sum().sum())
            selected = chunk.loc[(chunk.index >= cfg['start']) & (chunk.index <= cfg['end'])]
            frames.append(selected.astype('float32'))
        df = pd.concat(frames)
        assert total == 140256 and missing == 0
        assert np.isfinite(df.values).all()
        np.savez_compressed(cache, values=df.values, timestamps=df.index.values, users=df.columns.values.astype(str))
        save_json(OUT / 'raw_audit.json', dict(rows=total, users=370, first=str(first), last=str(last),
                  missing=missing, sha256=digest.hexdigest(), selected_rows=len(df),
                  selected_first=str(df.index[0]), selected_last=str(df.index[-1]),
                  zero_fraction=float((df.values == 0).mean())))
    data = np.load(cache)
    times = pd.DatetimeIndex(data['timestamps'])
    assert str(times[0]) == cfg['start'] and str(times[-1]) == cfg['end'], 'Delete processed.npz when changing date range'
    values, users = data['values'], data['users']
    ntrain = int((times <= cfg['train_end']).sum())
    nval = int((times <= cfg['val_end']).sum())
    mean = values[:ntrain].mean(0, dtype=np.float64).astype('float32')
    std = values[:ntrain].std(0, dtype=np.float64).astype('float32')
    std = np.maximum(std, 1.)
    normalized = (values - mean) / std
    h, f = cfg['history'], cfg['horizon']
    splits = {'train': np.arange(h, ntrain - f + 1, cfg['train_stride']),
              'val': np.arange(ntrain, nval - f + 1, cfg['eval_stride']),
              'test': np.arange(nval, len(times) - f + 1, cfg['eval_stride'])}
    for name, starts in splits.items():
        pd.DataFrame({'origin': times[starts - 1], 'target_first': times[starts],
                      'target_last': times[starts + f - 1], 'array_start': starts}).to_csv(OUT / f'{name}_windows.csv', index=False)
    np.savez(OUT / 'scaler.npz', mean=mean, std=std, users=users)
    # Select example using training data only: median training mean among active users.
    eligible = np.flatnonzero((values[:ntrain] > 0).mean(0) >= .95)
    example = int(eligible[np.argsort(mean[eligible], kind='stable')[len(eligible)//2]])
    save_json(OUT / 'protocol.json', dict(config=cfg, users=users.tolist(), users_count=len(users),
              selected_rows=len(values), ntrain=ntrain, nval_boundary=nval,
              split_samples={k: len(v) for k, v in splits.items()},
              constant_train_users=users[values[:ntrain].std(0) == 0].tolist(),
              example_user=str(users[example]), example_user_index=example,
              example_rule='median training mean among users with >=95% positive training records',
              python=platform.python_version(), torch=torch.__version__, device='cpu'))
    return normalized.astype('float32'), values, times, users, mean, std, splits


class Windows(Dataset):
    def __init__(self, values, starts, cfg):
        self.values, self.starts, self.cfg = values, starts, cfg
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, i):
        t = self.starts[i]
        return (torch.from_numpy(self.values[t-self.cfg['history']:t]),
                torch.from_numpy(self.values[t:t+self.cfg['horizon']]))


def evaluate(model, loader, std, mean):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            predictions.append(model(x).numpy() * std + mean)
            targets.append(y.numpy() * std + mean)
    return np.concatenate(predictions), np.concatenate(targets)


def metrics(name, pred, truth):
    err = np.abs(pred.astype('float64') - truth)
    return dict(model=name, mae_1h_kw=err[:, 3].mean(), mae_6h_kw=err[:, 23].mean(),
                mae_24h_kw=err[:, 95].mean(), mae_all_kw=err.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--models', nargs='+', default=FAMILIES, choices=FAMILIES)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    cfg = json.loads((ROOT / 'config.json').read_text())
    assert cfg['history'] == cfg['horizon'] == 96, 'Daily residual requires 96 steps'
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    print('Preparing and auditing data...', flush=True)
    z, raw, times, users, mean, std, splits = prepare(cfg)
    print('Windows:', {k:len(v) for k,v in splits.items()}, flush=True)
    loaders = {k: DataLoader(Windows(z, v, cfg), batch_size=cfg['batch_size'], shuffle=False) for k,v in splits.items()}
    truth = np.stack([raw[t:t+96] for t in splits['test']])
    baseline = np.stack([raw[t-96:t] for t in splits['test']])
    np.savez_compressed(OUT / 'test_targets.npz', truth=truth, origins=times[splits['test']-1].values, users=users)
    rows = [metrics('seasonal_naive', baseline, truth)]
    for family in args.models:
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        model = Forecaster(family, len(users), cfg)
        folder = OUT / family
        folder.mkdir(exist_ok=True)
        if not args.evaluate_only:
            generator = torch.Generator().manual_seed(cfg['seed'])
            train = DataLoader(Windows(z, splits['train'], cfg), batch_size=cfg['batch_size'], shuffle=True, generator=generator)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.)
            best, logs, started = float('inf'), [], time.perf_counter()
            for epoch in range(1, cfg['epochs']+1):
                model.train(); loss_sum = 0.
                for x,y in train:
                    opt.zero_grad(set_to_none=True)
                    loss = (model(x)-y).abs().mean()
                    assert torch.isfinite(loss), 'Non-finite training loss'
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    opt.step()
                    loss_sum += loss.item() * len(x)
                vp, vt = evaluate(model, loaders['val'], std, mean)
                val_mae = float(np.abs(vp-vt).mean())
                logs.append(dict(epoch=epoch, train_normalized_mae=loss_sum/len(train.dataset), val_mae_kw=val_mae,
                                 elapsed_seconds=time.perf_counter()-started))
                if val_mae < best:
                    best, best_epoch = val_mae, epoch
                    torch.save(dict(state_dict=model.state_dict(), family=family, config=cfg, users=users.tolist(), epoch=epoch), folder/'best.pt')
                pd.DataFrame(logs).to_csv(folder/'training_log.csv', index=False)
                print(f'{family} epoch {epoch:02d}: train={logs[-1]["train_normalized_mae"]:.4f} val={val_mae:.3f} kW elapsed={logs[-1]["elapsed_seconds"]:.1f}s', flush=True)
            save_json(folder/'training_summary.json', dict(parameters=sum(p.numel() for p in model.parameters()),
                best_epoch=best_epoch, best_val_mae_kw=best, seconds=time.perf_counter()-started))
        checkpoint = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        pred, _ = evaluate(model, loaders['test'], std, mean)
        assert pred.shape == truth.shape and np.isfinite(pred).all()
        np.savez_compressed(folder/'predictions.npz', predictions=pred, origins=times[splits['test']-1].values, users=users)
        row = metrics(family, pred, truth)
        rows.append(row); save_json(folder/'metrics.json', row)
        pd.DataFrame({'user':users, 'mae_1h_kw':abs(pred[:,3]-truth[:,3]).mean(0),
                      'mae_6h_kw':abs(pred[:,23]-truth[:,23]).mean(0),
                      'mae_24h_kw':abs(pred[:,95]-truth[:,95]).mean(0),
                      'mae_all_kw':abs(pred-truth).mean((0,1))}).to_csv(folder/'per_user_metrics.csv',index=False)
        print(row, flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    print('Finished. Results:', OUT, flush=True)


if __name__ == '__main__':
    main()
