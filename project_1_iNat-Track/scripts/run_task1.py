"""Frozen ImageNet backbones + linear species probes; run from project root."""
import argparse, csv, hashlib, io, json, platform, random, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision.models import resnet50, vit_b_16, ResNet50_Weights, ViT_B_16_Weights

def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def extract_summary(model, images, name):
    """Squeeze the last spatial map or final normalized patch tokens."""
    if name == 'resnet50':
        features = model.conv1(images)
        features = model.bn1(features)
        features = model.relu(features)
        features = model.maxpool(features)
        features = model.layer1(features)
        features = model.layer2(features)
        features = model.layer3(features)
        features = model.layer4(features)
        return features.mean(dim=(2, 3))
    patches = model._process_input(images)
    cls = model.class_token.expand(images.shape[0], -1, -1)
    tokens = model.encoder(torch.cat([cls, patches], dim=1))
    return tokens[:, 1:, :].mean(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train-00000-of-00001.parquet')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = Path('results/task1'); out.mkdir(parents=True, exist_ok=True)
    p = pq.ParquetFile(a.data)
    names = json.loads(p.schema_arrow.metadata[b'huggingface'])['info']['features']['label']['names']
    rows = p.read().to_pylist(); labels = torch.tensor([r['label'] for r in rows])
    rng = np.random.default_rng(seed)
    split = {k: [] for k in ['train', 'val', 'test']}
    for c in range(len(names)):
        ids = np.where(labels.numpy()==c)[0]; rng.shuffle(ids)
        assert len(ids)==10, 'Expected 10 examples per species'
        for k, v in [('train',ids[:6]),('val',ids[6:8]),('test',ids[8:])]: split[k].extend(map(int,v))
    save(out/'split.json', {'seed':seed,'indices':split,'sample_ids':{k:[rows[i]['id'] for i in v] for k,v in split.items()}})
    save(out/'class_names.json', names)
    config = {'args':vars(a),'seed':seed,'data_sha256':sha(a.data),'samples':len(rows),'classes':len(names),'split_sizes':{k:len(v) for k,v in split.items()},'torch':torch.__version__,'torchvision':torchvision.__version__,'python':platform.python_version(),'device':'cpu','feature_pooling':'resnet layer4 spatial mean; vit final patch token mean excluding CLS','platform':platform.platform(),'head':'Linear + bias; train-only feature standardization','optimizer':'AdamW lr=0.01 weight_decay=0.01; full batch','selection':'minimum validation cross entropy; test evaluated once after selection'}
    save(out/'config.json',config)
    Path('weights').mkdir(exist_ok=True)
    results=[]
    specs=[('resnet50',resnet50,ResNet50_Weights.IMAGENET1K_V2,2048),('vit_b_16',vit_b_16,ViT_B_16_Weights.IMAGENET1K_V1,768)]
    for name,ctor,w,dim in specs:
        d=out/name; d.mkdir(exist_ok=True)
        weight=Path('weights')/w.url.rsplit('/',1)[-1]
        if not weight.exists():
            torch.hub.download_url_to_file(w.url,str(weight),hash_prefix=weight.stem.rsplit('-',1)[-1])
        digest=sha(weight); assert digest.startswith(weight.stem.rsplit('-',1)[-1])
        model=ctor(weights=None); model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True))
        if name=='resnet50': model.fc=nn.Identity()
        else: model.heads=nn.Identity()
        model.requires_grad_(False); model.eval()
        transform=w.transforms()
        pooling = 'layer4 spatial mean over H,W' if name == 'resnet50' else 'encoder final LayerNorm output; mean of 196 patch tokens excluding CLS'
        metadata={'weights':str(w),'url':w.url,'sha256':digest,'transform':str(transform),'pooling':pooling,'feature_dim':dim,'frozen_parameters':sum(x.numel() for x in model.parameters()),'trainable_backbone_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad)}
        save(d/'backbone.json',metadata)
        start=time.perf_counter(); chunks=[]
        cache=d/'features.pt'
        cached=torch.load(cache,weights_only=True) if cache.exists() else None
        if cached is not None and cached.get('data_sha256')==config['data_sha256'] and cached.get('weights_sha256')==digest and cached.get('pooling')==pooling and cached.get('transform')==str(transform):
            features=cached['features']
            seconds=cached['feature_seconds']
            print(f'{name}: verified feature cache reused',flush=True)
        else:
          with torch.inference_mode():
            for start_i in range(0,len(rows),a.batch_size):
                ims=[transform(Image.open(io.BytesIO(r['image']['bytes'])).convert('RGB')) for r in rows[start_i:start_i+a.batch_size]]
                chunks.append(extract_summary(model, torch.stack(ims), name).cpu())
                print(f'{name}: features {min(start_i+a.batch_size,len(rows))}/{len(rows)}',flush=True)
          features=torch.cat(chunks).clone(); seconds=time.perf_counter()-start
        torch.save({'features':features,'labels':labels,'data_sha256':config['data_sha256'],'weights_sha256':digest,'pooling':pooling,'transform':str(transform),'feature_seconds':seconds},d/'features.pt')
        del model, chunks
        train,val,test=[torch.tensor(split[k]) for k in ['train','val','test']]
        mean=features[train].mean(0); std=features[train].std(0,unbiased=False).clamp_min(1e-6)
        x=(features-mean)/std
        torch.manual_seed(seed); head=nn.Linear(dim,len(names))
        torch.save(head.state_dict(), d/'initial_state.pt')
        optimizer=torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.01)
        history=[]; best=float('inf'); best_state=None
        for epoch in range(1,a.epochs+1):
            head.train(); optimizer.zero_grad(); loss=nn.functional.cross_entropy(head(x[train]),labels[train]); loss.backward(); optimizer.step()
            head.eval()
            with torch.no_grad():
                v=nn.functional.cross_entropy(head(x[val]),labels[val]).item()
            history.append({'epoch':epoch,'train_loss':loss.item(),'val_loss':v})
            if v<best:
                best=v; best_epoch=epoch; best_state={k:t.detach().clone() for k,t in head.state_dict().items()}
        head.load_state_dict(best_state)
        with torch.no_grad(): probs=head(x[test]).softmax(1); top=probs.topk(5,dim=1).indices
        truth=labels[test]; pred=top[:,0]
        cm=torch.zeros(len(names),len(names),dtype=torch.int64)
        for y,h in zip(truth,pred): cm[y,h]+=1
        tp=cm.diag().float(); f1=2*tp/(cm.sum(0)+cm.sum(1)).clamp_min(1)
        result={'model':name,'trainable_parameters':sum(t.numel() for t in head.parameters()),'top1_pct':100*(pred==truth).float().mean().item(),'top5_pct':100*(top==truth[:,None]).any(1).float().mean().item(),'macro_f1_pct':100*f1.mean().item(),'best_epoch':best_epoch,'best_val_loss':best,'feature_seconds':seconds,'train_samples':len(train),'val_samples':len(val),'test_samples':len(test)}
        save(d/'metrics.json',result); save(d/'history.json',history)
        torch.save({'state_dict':best_state,'feature_mean':mean,'feature_std':std,'class_names':names,'backbone':metadata},d/'head.pt')
        np.savetxt(d/'confusion_matrix.csv',cm.numpy(),fmt='%d',delimiter=',')
        with (d/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['row_index','sample_id','true_label','true_species','pred_label','pred_species','top5'])
            for j,i in enumerate(split['test']): writer.writerow([i,rows[i]['id'],int(truth[j]),names[int(truth[j])],int(pred[j]),names[int(pred[j])],json.dumps(top[j].tolist())])
        results.append(result); save(out/'summary.json',results)
        print(json.dumps(result),flush=True)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    lines=['| 特征提取器 | 训练部分 | 可训练参数量 | Top-1 (%) | Top-5 (%) | Macro-F1 (%) |','|---|---|---:|---:|---:|---:|']
    for r in results: lines.append(f"| {r['model']} | 物种分类头 | {r['trainable_parameters']:,} | {r['top1_pct']:.2f} | {r['top5_pct']:.2f} | {r['macro_f1_pct']:.2f} |")
    (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__': main()
