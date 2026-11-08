#!/usr/bin/env python3
"""Independent evaluation entry point for FIT summaries and post-freeze HELD."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path


def fit_summary(run_dir: Path, output: Path) -> None:
    rows=[]
    for p in sorted((run_dir/'matched').glob('*.json')):
        try: d=json.loads(p.read_text())
        except Exception: continue
        if d.get('status') != 'EXECUTED': continue
        m=d.get('fit_metrics_native_head') or d.get('fit_metrics_fixed_head') or {}
        rows.append({'file':str(p.relative_to(run_dir)),'arm':d.get('algorithm_arm') or d.get('arm'),'partition':d.get('partition'),'training_seed':d.get('training_seed'),'macro_f1':m.get('macro_f1'),'tail10':m.get('tail_recall_worst10'),'wall':d.get('wall_clock_training_seconds'),'flops':d.get('estimated_training_flops'),'held_reads':d.get('held_reads')})
    if not rows: raise SystemExit(f'no executed FIT rows under {run_dir / "matched"}')
    groups={}
    for r in rows: groups.setdefault(r['arm'],[]).append(r)
    summary=[]
    for arm,rs in sorted(groups.items()):
        mean=lambda key: statistics.fmean(float(r[key]) for r in rs if r[key] is not None)
        summary.append({'arm':arm,'n':len(rs),'macro_f1_mean':mean('macro_f1'),'tail10_mean':mean('tail10'),'wall_mean':mean('wall'),'flops_mean':mean('flops'),'held_reads_max':max(r['held_reads'] or 0 for r in rs)})
    output.write_text(json.dumps({'status':'EXECUTED','split':'fit','scope':'FIT rows only','rows':rows,'summary':summary},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'EXECUTED','split':'fit','n_rows':len(rows),'output':str(output)},ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',required=True)
    ap.add_argument('--split',choices=['fit','held'],default='fit')
    ap.add_argument('--output',required=True)
    ap.add_argument('--gpu',type=int,default=0)
    ap.add_argument('--freeze-manifest',default=None)
    args=ap.parse_args()
    if args.split=='held':
        from validated_runtime.held_eval import main as held_main
        import sys
        sys.argv=['held_eval.py','--freeze-manifest',args.freeze_manifest or str(Path(args.run_dir)/'PR_FIT_FREEZE_MANIFEST.json'),'--output',args.output,'--gpu',str(args.gpu)]
        held_main()
    else:
        fit_summary(Path(args.run_dir),Path(args.output))

if __name__=='__main__': main()
