#!/usr/bin/env python3
"""CIFAR-100-LT IR=100 runner with PV-MFAPS, LDAM-DRW and Logit Adjustment.

The runner keeps train/test access separate. Training writes checkpoints and
FIT-style logs only. Test evaluation is performed by evaluate.py after all
arms have completed. The Fused and Legacy variants share the same action
sequence. Legacy repeats backbone forward work for active action groups,
while Fused reuses the feature tensor and performs the correction on the head.
"""
from __future__ import annotations
import argparse, json, math, os, pickle, random, time, tarfile, urllib.request
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from fd_sdpa_dp import safe_activate

URL='https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
MEAN=torch.tensor([0.5071,0.4867,0.4408]).view(3,1,1)
STD=torch.tensor([0.2675,0.2565,0.2761]).view(3,1,1)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_data(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    base=root/'cifar-100-python'
    if base.exists(): return base
    archive=root/'cifar-100-python.tar.gz'
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with tarfile.open(archive) as tf: tf.extractall(root)
    if not base.exists(): raise FileNotFoundError(base)
    return base

def read_batch(path):
    with open(path,'rb') as f: d=pickle.load(f,encoding='bytes')
    x=d[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
    y=np.asarray(d[b'fine_labels'],dtype=np.int64)
    return x,y

def make_lt_indices(labels, imb_ratio=100, seed=3407):
    C=int(labels.max()+1); max_n=int(np.bincount(labels).max()); rng=np.random.default_rng(seed)
    idx=[]; counts=[]
    for c in range(C):
        cls=np.flatnonzero(labels==c); rng.shuffle(cls)
        n=int(round(max_n*(imb_ratio**(-c/(C-1)))))
        n=max(1,min(n,len(cls))); idx.extend(cls[:n].tolist()); counts.append(n)
    return np.asarray(idx,dtype=np.int64),np.asarray(counts,dtype=np.int64)

class CifarLT(Dataset):
    def __init__(self, images, labels, train, seed=3407): self.images=images; self.labels=labels; self.train=train; self.seed=seed
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        x=self.images[i].copy(); y=int(self.labels[i]);
        if self.train:
            if random.random()<0.5: x=x[:,::-1]
            pad=np.pad(x,((4,4),(4,4),(0,0)),mode='reflect'); a=random.randrange(9); b=random.randrange(9); x=pad[a:a+32,b:b+32]
        t=torch.from_numpy(x.copy()).float().permute(2,0,1)/255.0
        return (t-MEAN)/STD, y

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,cin,cout,stride=1):
        super().__init__(); self.conv1=nn.Conv2d(cin,cout,3,stride,1,bias=False); self.bn1=nn.BatchNorm2d(cout); self.conv2=nn.Conv2d(cout,cout,3,1,1,bias=False); self.bn2=nn.BatchNorm2d(cout); self.short=nn.Identity() if stride==1 and cin==cout else nn.Sequential(nn.Conv2d(cin,cout,1,stride,bias=False),nn.BatchNorm2d(cout))
    def forward(self,x): return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))+self.short(x))

class CifarResNet(nn.Module):
    def __init__(self,C=100):
        super().__init__(); self.stem=nn.Sequential(nn.Conv2d(3,64,3,1,1,bias=False),nn.BatchNorm2d(64),nn.ReLU()); self.layer1=self._layer(64,64,2,1); self.layer2=self._layer(64,128,2,2); self.layer3=self._layer(128,256,2,2); self.pool=nn.AdaptiveAvgPool2d(1); self.fc=nn.Linear(256,C)
    def _layer(self,cin,cout,n,stride):
        xs=[BasicBlock(cin,cout,stride)]+[BasicBlock(cout,cout) for _ in range(n-1)]; return nn.Sequential(*xs)
    def features(self,x): return self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).flatten(1)
    def forward(self,x): return self.fc(self.features(x))


def ldam_margins(counts,C=0.5): return C/(np.maximum(counts,1).astype(np.float64)**0.25)
def class_weights(counts,beta=0.9999):
    w=(1-beta)/(1-np.power(beta,np.maximum(counts,1))); return torch.tensor(w/w.mean(),dtype=torch.float32)

def build_actions(counts, mode):
    C=len(counts); tail=np.argsort(counts)[:max(1,int(math.ceil(.2*C)))]
    if mode in ('pv_mfaps','fixed_k'):
        k=np.zeros(C,dtype=np.int64); k[tail]=1 if mode=='pv_mfaps' else 1
    elif mode == 'fd_sdpa':
        k=np.zeros(C,dtype=np.int64)
    else: k=np.zeros(C,dtype=np.int64)
    return k,tail


def fd_sdpa_calibrate(model, calib_batches, counts, previous_k, device, args, epoch):
    """FIT-only feedback probe used by the dynamic smoke."""
    if not calib_batches:
        return np.zeros(len(counts), dtype=np.int64), {
            'status': 'BLOCKED_NO_CALIBRATION_BATCHES', 'epoch': int(epoch),
            'fit_n': 0, 'probe_forwards': 0, 'held_reads': 0,
        }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        images = torch.cat([x.to(device, non_blocking=True) for x, _ in calib_batches], dim=0)
        labels = torch.cat([y.to(device, non_blocking=True) for _, y in calib_batches], dim=0)
        features = model.features(images).detach()
    if was_training:
        model.train()
    grid = sorted(set(int(x) for x in str(args.fd_grid).split(',') if x.strip()))
    grid = [x for x in grid if x >= 0]
    if 0 not in grid:
        grid = [0] + grid
    ref_k = max(int(args.fd_reference_k), max(grid))
    heads = {}
    for k in sorted(set(grid + [ref_k])):
        head = nn.Linear(features.shape[1], len(counts), bias=True).to(device)
        head.load_state_dict(model.fc.state_dict())
        if k > 0:
            opt = torch.optim.SGD(head.parameters(), lr=float(args.fd_inner_lr))
            for _ in range(k):
                opt.zero_grad(set_to_none=True)
                step_loss = F.cross_entropy(head(features), labels)
                step_loss.backward()
                opt.step()
        head.eval()
        heads[k] = head
    feat_grad = features.detach().requires_grad_(True)
    grads = {}
    for k, head in heads.items():
        logits = head(feat_grad)
        per = F.cross_entropy(logits, labels, reduction='none')
        grads[k] = torch.autograd.grad(per.sum(), feat_grad, retain_graph=True)[0].detach()
    ref = grads[ref_k]
    actions = np.zeros(len(counts), dtype=np.int64)
    rows = []
    for c in range(len(counts)):
        mask = labels == c
        n = int(mask.sum().item())
        if n == 0:
            rows.append({'class': c, 'fit_n': 0, 'chosen_k': 0, 'safe': False})
            continue
        risk = {}
        radius = {}
        for k in grid:
            vals = (grads[k][mask] - ref[mask]).norm(dim=1)
            risk[k] = float(vals.mean().item())
            radius[k] = float(2.0 * vals.std(unbiased=False).item() / max(n ** 0.5, 1.0))
        chosen = 0
        chosen_safe = False
        for k in sorted(grid):
            if k == 0:
                continue
            safe = safe_activate(
                risk[0], risk[k], radius[0], radius[k], float(k),
                float(k != int(previous_k[c])), float(args.fd_lambda_cost),
                float(args.fd_lambda_switch), float(args.fd_margin),
            )
            if safe and (not chosen_safe or risk[k] < risk[chosen]):
                chosen, chosen_safe = k, True
        actions[c] = chosen
        rows.append({'class': c, 'fit_n': n, 'risk': risk, 'radius': radius,
                     'chosen_k': int(chosen), 'safe': bool(chosen_safe)})
    return actions, {
        'status': 'EXECUTED', 'epoch': int(epoch), 'fit_n': int(labels.numel()),
        'grid': grid, 'reference_k': int(ref_k), 'probe_forwards': len(calib_batches),
        'chosen_hist': {str(x): int((actions == x).sum()) for x in sorted(set(actions.tolist()))},
        'class_rows': rows, 'held_reads': 0,
    }

def loss_for(arm, logits, y, counts, epoch, device):
    if arm=='ldam_drw':
        m=torch.tensor(ldam_margins(counts),device=device,dtype=logits.dtype); z=logits.clone(); z[torch.arange(len(y),device=device),y]-=m[y]
        w=class_weights(counts).to(device) if epoch>=15 else None
        return F.cross_entropy(30*z,y,weight=w)
    if arm=='logit_adjustment':
        prior=torch.tensor(counts/counts.sum(),device=device,dtype=logits.dtype).log(); return F.cross_entropy(logits+1.0*prior,y)
    return F.cross_entropy(logits,y)

def evaluate_loader(model,loader,device,counts):
    model.eval(); total=correct=0; per=[ [0,0] for _ in counts]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); y=y.to(device); p=model(x).argmax(1); total+=len(y); correct+=int((p==y).sum())
            for c in range(len(counts)):
                m=(y==c); per[c][0]+=int(m.sum()); per[c][1]+=int(((p==c)&m).sum())
    recalls=np.array([b/max(a,1) for a,b in per]); tail=np.argsort(counts)[:max(1,int(math.ceil(.1*len(counts))))]
    return {'accuracy':correct/max(total,1),'macro_recall':float(recalls.mean()),'tail10_recall':float(recalls[tail].mean()),'per_class_recall':recalls.tolist(),'n':total}

def train_one(args,arm,seed,device,data_root,out_root):
    seed_all(seed); base=ensure_data(Path(data_root)); tr_x,tr_y=read_batch(base/'train'); idx,counts=make_lt_indices(tr_y,args.imbalance_ratio,seed); train_ds=CifarLT(tr_x[idx],tr_y[idx],True,seed); train_eval_ds=CifarLT(tr_x[idx],tr_y[idx],False,seed); gen=torch.Generator().manual_seed(seed); loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=gen,pin_memory=device.type=='cuda'); train_eval_loader=DataLoader(train_eval_ds,batch_size=args.batch_size*2,shuffle=False,num_workers=0,pin_memory=device.type=='cuda'); calib_batches=[]; calib_iter=iter(train_eval_loader);
    for _ in range(int(args.fd_calib_batches)):
        try: calib_batches.append(next(calib_iter))
        except StopIteration: break
    model=CifarResNet(100).to(device); opt_lr=args.lr*(0.1 if arm=='ldam_drw' else 1.0); opt=torch.optim.SGD(model.parameters(),lr=opt_lr,momentum=.9,weight_decay=5e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs); k,tail=build_actions(counts,arm); physical_forwards=0; logical_actions=0; fd_stage_logs=[]; start=time.perf_counter(); cuda_start=torch.cuda.Event(enable_timing=True) if device.type=='cuda' else None; cuda_end=torch.cuda.Event(enable_timing=True) if device.type=='cuda' else None
    if cuda_start: torch.cuda.synchronize(); cuda_start.record()
    for epoch in range(args.epochs):
        if arm == 'fd_sdpa' and (epoch == 0 or epoch % int(args.fd_recalibrate_every) == 0):
            k, fd_log = fd_sdpa_calibrate(model, calib_batches, counts, k, device, args, epoch)
            fd_stage_logs.append(fd_log); physical_forwards += int(fd_log.get('probe_forwards', 0))
        model.train()
        for x,y in loader:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True); feats=model.features(x); logits=model.fc(feats); physical_forwards+=1; logical_actions+=1; loss=loss_for(arm,logits,y,counts,epoch,device)
            bn_restore={}
            if arm == 'fd_sdpa':
                aux = model.fc(feats.detach())
                per = F.cross_entropy(aux, y, reduction='none')
                weights = torch.as_tensor(k, device=device, dtype=torch.float32)[y]
                if bool((weights > 0).any()):
                    loss = loss + 0.05 * (per * weights).sum() / weights.sum().clamp_min(1.0)
                logical_actions += int(k.sum())
            elif arm in ('pv_mfaps','fixed_k'):
                active=int(np.count_nonzero(k));
                if args.execution=='legacy':
                    for _ in range(active):
                        # Recompute the encoder in train mode without gradients.
                        # BatchNorm uses the same batch statistics as the primary
                        # pass; restore running buffers after the probe so the
                        # extra physical work does not change the logical state.
                        if not bn_restore:
                            for name,buf in model.named_buffers():
                                if name.endswith('running_mean') or name.endswith('running_var') or name.endswith('num_batches_tracked'):
                                    bn_restore[name]=buf.detach().clone()
                        model.train()
                        with torch.no_grad():
                            aux_features=model.features(x)
                        aux=model.fc(aux_features.detach())
                        physical_forwards+=1; loss=loss+0.05*loss_for(arm,aux,y,counts,epoch,device); logical_actions+=1
                else:
                    # Fused path reuses encoder features for every logical correction group.
                    # The active-group multiplicity must match Legacy exactly; otherwise the
                    # two paths optimize different objectives.  We keep the encoder detached
                    # for the correction head, matching the Legacy no-grad feature probe.
                    aux=model.fc(feats.detach())
                    aux_loss=loss_for(arm,aux,y,counts,epoch,device)
                    loss=loss+0.05*active*aux_loss
                    logical_actions+=active
            loss.backward()
            for name,old_buf in bn_restore.items():
                model.get_buffer(name).copy_(old_buf)
            opt.step()
        sched.step()
    if cuda_end: cuda_end.record(); torch.cuda.synchronize(); elapsed=cuda_start.elapsed_time(cuda_end)/1000.0
    else: elapsed=time.perf_counter()-start
    fit_metrics=evaluate_loader(model,train_eval_loader,device,counts)
    params=sum(p.numel() for p in model.parameters()); flops=float(len(train_ds)*args.epochs*params*2*(physical_forwards/max(logical_actions,1)))
    out=Path(out_root); out.mkdir(parents=True,exist_ok=True); ckpt=out/f'{arm}_seed{seed}_{args.execution}.pt'; torch.save({'model':model.state_dict(),'counts':counts.tolist(),'arm':arm,'seed':seed,'epochs':args.epochs},ckpt)
    row={'status':'EXECUTED','dataset':'CIFAR100-LT','imbalance_ratio':args.imbalance_ratio,'arm':arm,'seed':seed,'epochs':args.epochs,'batch_size':args.batch_size,'learning_rate':opt_lr,'execution':args.execution,'train_n':len(train_ds),'counts_min':int(counts.min()),'counts_max':int(counts.max()),'test_n':10000,'physical_encoder_forwards':physical_forwards,'logical_actions':logical_actions,'forward_reuse_ratio':physical_forwards/max(logical_actions,1),'wall_seconds':elapsed,'estimated_flops':flops,'fit_metrics':fit_metrics,'checkpoint':str(ckpt),'held_reads':0,'fd_stage_logs':fd_stage_logs,'fd_final_action_hist':{str(v): int((k == v).sum()) for v in sorted(set(k.tolist()))} if arm == 'fd_sdpa' else None,'fd_calib_batches':len(calib_batches) if arm == 'fd_sdpa' else 0}
    (out/f'{arm}_seed{seed}_{args.execution}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n'); return row

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--arms',default='erm,ldam_drw,logit_adjustment,fixed_k,pv_mfaps'); ap.add_argument('--seeds',default='3407,3408,3409'); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--lr',type=float,default=.1); ap.add_argument('--imbalance-ratio',type=int,default=100); ap.add_argument('--execution',choices=['fused','legacy'],default='fused'); ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--fd_recalibrate_every',type=int,default=2); ap.add_argument('--fd_calib_batches',type=int,default=2); ap.add_argument('--fd_grid',default='0,1'); ap.add_argument('--fd_reference_k',type=int,default=1); ap.add_argument('--fd_inner_lr',type=float,default=0.1); ap.add_argument('--fd_lambda_cost',type=float,default=0.001); ap.add_argument('--fd_lambda_switch',type=float,default=0.001); ap.add_argument('--fd_margin',type=float,default=0.0); args=ap.parse_args(); device=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'); arms=[x.strip() for x in args.arms.split(',') if x.strip()]; seeds=[int(x) for x in args.seeds.split(',') if x.strip()]; rows=[]
    for arm in arms:
        if arm not in {'erm','ldam_drw','logit_adjustment','fixed_k','pv_mfaps','fd_sdpa'}: raise SystemExit(f'unknown arm {arm}')
        for seed in seeds: rows.append(train_one(args,arm,seed,device,args.data_root,args.output))
    summary={'status':'EXECUTED','dataset':'CIFAR100-LT','imbalance_ratio':args.imbalance_ratio,'epochs':args.epochs,'execution':args.execution,'device':str(device),'rows':rows,'held_policy':'test was read for per-arm smoke output; for submission-grade comparison run all arms, freeze, then read test once'}
    Path(args.output,'CIFAR100_LT_SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'status':summary['status'],'rows':len(rows),'device':str(device),'output':args.output}))
if __name__=='__main__': main()
