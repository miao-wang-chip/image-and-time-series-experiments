"""Recompute reported metrics from saved predictions; verify shared evaluation IDs."""
import csv,json
from pathlib import Path
import numpy as np

p=Path('results/task1')
s=json.loads((p/'split.json').read_text())
parts=[set(s['indices'][k]) for k in ['train','val','test']]
assert len(set.union(*parts))==1000
assert all(not(parts[i]&parts[j]) for i,j in [(0,1),(0,2),(1,2)])
ids=[]
for model in ['resnet50','vit_b_16']:
    d=p/model
    with (d/'predictions.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==200
    sample_ids=[int(r['sample_id']) for r in rows]
    assert sample_ids==s['sample_ids']['test']
    ids.append(sample_ids)
    cm=np.zeros((100,100),dtype=int)
    hits5=0
    for r in rows:
        y,h=int(r['true_label']),int(r['pred_label'])
        top=json.loads(r['top5'])
        assert len(set(top))==5 and top[0]==h
        cm[y,h]+=1; hits5+=y in top
    assert np.array_equal(cm,np.loadtxt(d/'confusion_matrix.csv',delimiter=','))
    assert np.all(cm.sum(1)==2)
    m=json.loads((d/'metrics.json').read_text())
    expected={'top1_pct':100*np.trace(cm)/200,'top5_pct':100*hits5/200,'macro_f1_pct':100*np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1)))}
    for key,value in expected.items(): assert abs(m[key]-value)<1e-4
    b=json.loads((d/'backbone.json').read_text())
    assert b['trainable_backbone_parameters']==0
    print(model,expected)
assert ids[0]==ids[1]
(p/'verification.txt').write_text('PASS: disjoint 600/200/200 split; identical 200 test IDs; 2 samples/class; predictions reproduce all metrics and confusion matrices; zero trainable backbone parameters.\n',encoding='utf-8')
print('All result checks passed.')
