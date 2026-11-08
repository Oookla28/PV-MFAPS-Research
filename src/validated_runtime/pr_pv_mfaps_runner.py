#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT_CODE = Path(__file__).resolve().parent
# Existing validated runner lives in sibling code directory; import it explicitly.
BASE_CODE = ROOT_CODE
sys.path = [str(BASE_CODE)] + [p for p in sys.path if p not in (str(ROOT_CODE), '')]
from pivot_runner import get_endpoint
from bs_plugin_runner import pv_policy, bs_fixed_policy
from tfgps_gpu_runner import make_policy_for_arm, run_single_arm
from tfgps_gpu_lib import write_json

ARMS = ('pr_pv_mfaps','bs_pcfa','bs_fixed_k_matched_B02','effective_number')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--action-policy',required=True)
    ap.add_argument('--partitions',default='0,1'); ap.add_argument('--seeds',default='3407,3408,3409')
    ap.add_argument('--steps',type=int,default=8); ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--arms',default=','.join(ARMS))
    ap.add_argument('--phase',default='matched')
    args=ap.parse_args()
    root=Path(args.root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    (out/'matched').mkdir(exist_ok=True); (out/'rows').mkdir(exist_ok=True)
    arms=[x.strip() for x in args.arms.split(',') if x.strip()]
    if any(x not in ARMS for x in arms): raise SystemExit(f'unsupported arms {arms}')
    # PR and bs_pcfa use the same validated fused kernel and action vector.
    # Keep one copy in the independent PR freeze to avoid state-tag collision;
    # the prior plugin matrix already contains the bs_pcfa control evidence.
    duplicate_control_omitted = False
    if 'pr_pv_mfaps' in arms and 'bs_pcfa' in arms:
        arms = [a for a in arms if a != 'bs_pcfa']
        duplicate_control_omitted = True
    parts=[int(x) for x in args.partitions.split(',') if x.strip()]
    seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    pr_policy=json.loads(Path(args.action_policy).read_text())
    rows=[]
    for part in parts:
        runner, base, head, x, labels, fit, held, counts, cal, ckpt, payload, device = get_endpoint('raw',3407,part,args.gpu,args.batch_size,out/'calibration')
        for ts in seeds:
            for arm in arms:
                if arm=='pr_pv_mfaps':
                    pol=pv_policy(counts,pr_policy,balanced=True)
                    internal_arm='bs_pcfa'
                elif arm=='bs_pcfa':
                    # Registered nominal BS plugin policy from the existing control path.
                    pol=pv_policy(counts,pr_policy,balanced=True)
                    internal_arm='bs_pcfa'
                elif arm=='bs_fixed_k_matched_B02':
                    pol=bs_fixed_policy(counts,cal,ts,args.batch_size); internal_arm=arm
                else:
                    pol=make_policy_for_arm(arm,counts,cal,args.batch_size,ts); internal_arm=arm
                prefix = 'PR_PV_PR' if arm == 'pr_pv_mfaps' else 'PR_PV_' + arm
                r=run_single_arm(internal_arm,base,runner,head,x,labels,fit,counts,cal,'raw',part,3407,ts,args.steps,args.batch_size,device,'matched',out/'matched',len(held),int(payload.get('epoch',-1)),str(ckpt),policy_override=pol,output_prefix=prefix,pcfa_fused=True if internal_arm=='bs_pcfa' else None)
                row={'status':r.get('status'),'algorithm_arm':arm,'internal_arm':internal_arm,'partition':part,'training_seed':ts,
                     'fit_macro_f1':(r.get('fit_metrics_native_head') or {}).get('macro_f1'),'fit_tail10':(r.get('fit_metrics_native_head') or {}).get('tail_recall_worst10'),
                     'flops':r.get('estimated_training_flops'),'wall':r.get('wall_clock_training_seconds'),'cuda':r.get('cuda_training_seconds'),
                     'held_reads':r.get('held_reads',0),'pcfa_fused':r.get('pcfa_fused'),
                     'file':str(out/'matched'/f'PR_PV_raw_p{part}_s3407_{arm}_train{ts}.json')}
                rows.append(row)
                write_json(out/'rows'/f'PR_PV_SUMMARY_p{part}_s{ts}_{arm}.json',{'status':'EXECUTED','kind':'PR_PV_ROW','held_reads':0,'row':row})
                print(json.dumps(row,ensure_ascii=False),flush=True)
    ok=sum(1 for r in rows if r['status']=='EXECUTED')
    manifest={'status':'EXECUTED' if ok==len(rows) and rows else 'INVALID_CONFIG','kind':'PR_PV_MFAPS_MATCHED_MANIFEST','scope':'shared partition-robust risk policy FIT-only; HELD untouched','held_reads':0,
      'registered_config':{'arms':arms,'duplicate_control_omitted':duplicate_control_omitted,'omitted_equivalent_control':'bs_pcfa' if duplicate_control_omitted else None,'partitions':parts,'seeds':seeds,'steps':args.steps,'batch_size':args.batch_size,'variant':'a100_fp32_fused','action_policy':str(args.action_policy),'shared_policy':True,'cost_claim':'measured post-run only'},
      'execution_manifest':{'n_rows':len(rows),'n_executed':ok,'n_invalid':len(rows)-ok,'rows':rows},
      'coverage_status':{'status':'fit_only','note':'Held efficacy requires freeze and one-time HELD evaluation'}}
    write_json(out/'PR_PV_MFAPS_MATCHED_MANIFEST.json',manifest)
    print(json.dumps({'status':manifest['status'],'n':len(rows),'executed':ok},ensure_ascii=False))
    if manifest['status']!='EXECUTED': sys.exit(2)
if __name__=='__main__': main()
