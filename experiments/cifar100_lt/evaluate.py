#!/usr/bin/env python3
"""Single post-training CIFAR-100-LT test evaluation and aggregation."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import torch
from cifar100_lt_runner import ensure_data,read_batch,CifarLT,CifarResNet,evaluate_loader
from torch.utils.data import DataLoader

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--output',required=True); ap.add_argument('--gpu',type=int,default=0); args=ap.parse_args()
 device=torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'); base=ensure_data(Path(args.data_root)); te_x,te_y=read_batch(base/'test'); groups={}; held_reads=0
 for p in sorted(Path(args.run_dir).glob('*.json')):
  if p.name in {'CIFAR100_LT_SUMMARY.json','CIFAR100_LT_EVALUATION.json'}: continue
  d=json.loads(p.read_text())
  if d.get('status')!='EXECUTED' or 'checkpoint' not in d: continue
  if d.get('held_reads', 0) != 0:
      raise RuntimeError(f'non-FIT artifact cannot enter one-time evaluator: {p}')
  ckpt=torch.load(d['checkpoint'],map_location='cpu'); model=CifarResNet(100, depth=int(ckpt.get('model_depth',14)), width=ckpt.get('model_width')).to(device); model.load_state_dict(ckpt['model']); model.eval(); counts=torch.tensor(ckpt['counts']).numpy(); loader=DataLoader(CifarLT(te_x,te_y,False,int(d['seed']),d.get('augmentation','reflect')),batch_size=int(d['batch_size'])*2,shuffle=False,num_workers=0); metrics=evaluate_loader(model,loader,device,counts,arm=d.get('arm')); held_reads+=metrics['n']; d['held_metrics']=metrics; d['held_reads']=metrics['n']; p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n'); groups.setdefault((d['arm'],d['execution']),[]).append(d)
 rows=[]
 for (arm,exe),rs in sorted(groups.items()):
  rows.append({'arm':arm,'execution':exe,'n':len(rs),'accuracy_mean':statistics.fmean(r['held_metrics']['accuracy'] for r in rs),'macro_recall_mean':statistics.fmean(r['held_metrics']['macro_recall'] for r in rs),'macro_f1_mean':statistics.fmean(r['held_metrics'].get('macro_f1',float('nan')) for r in rs),'tail10_recall_mean':statistics.fmean(r['held_metrics']['tail10_recall'] for r in rs),'tail10_f1_mean':statistics.fmean(r['held_metrics'].get('tail10_f1',float('nan')) for r in rs),'wall_mean':statistics.fmean(r['wall_seconds'] for r in rs),'flops_mean':statistics.fmean(r['estimated_flops'] for r in rs),'forward_reuse_mean':statistics.fmean(r['forward_reuse_ratio'] for r in rs)})
 out={'status':'EXECUTED','dataset':'CIFAR100-LT','rows':rows,'held_reads':held_reads,'scope':'single post-training test evaluation after all checkpoints were produced'}; Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n'); print(json.dumps({'status':'EXECUTED','groups':len(rows),'held_reads':held_reads}))
if __name__=='__main__': main()
