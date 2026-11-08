#!/usr/bin/env python3
"""K-MFAPS threshold allocation contract on stored FIT class proxies."""
from pathlib import Path
import json, math
import numpy as np
ROOT=Path('$WORKSPACE_ROOT/feedback_geometry_pipeline_2026-11-04'); D=ROOT/'diagnostics'
KGRID=np.array([0,1,2,4],int); B=0.2; KMAX=4; GAMMA=.2; TAU=.75; BETA=0.0

def solve(A,sigma,rho,counts):
 ref=np.maximum(2*sigma-A,1e-8); kap=A/ref; b=(kap+BETA*sigma)**2; rho=np.clip(rho,.7,.999999); q=counts.astype(float)**(1-TAU); q/=q.sum(); tail=np.argsort(counts)[:max(1,int(math.ceil(GAMMA*len(counts))))]
 def kc(z): return np.clip(np.log(np.maximum(b/z,1.0))/(-2*np.log(rho)),0,KMAX)
 def cost(z): return float(np.sum(q[tail]*kc(z)[tail]))
 lo=max(float(b.min())*1e-8,1e-12); hi=float(max(b.max(),1e-6))
 for _ in range(100):
  mid=math.sqrt(lo*hi)
  if cost(mid)>B: lo=mid
  else: hi=mid
 z=hi; kcont=kc(z)
 # round down/up candidates and greedily repair to a registered grid under budget.
 kdisc=np.asarray([KGRID[np.argmin(np.abs(KGRID-x))] for x in kcont],int)
 while float(np.sum(q[tail]*kdisc[tail]))>B+1e-9:
  candidates=[c for c in tail if kdisc[c]>0]
  if not candidates: break
  c=max(candidates,key=lambda j:kdisc[j])
  lower=KGRID[KGRID<kdisc[c]]
  kdisc[c]=lower[-1] if len(lower) else 0
 risk0=b[tail]; riskg=b[tail]*rho[tail]**(2*kdisc[tail]);
 return {'count_min':int(counts.min()),'count_max':int(counts.max()),'tail_size':int(len(tail)),'tail_indices':tail.tolist(),'q_tail_mass':float(q[tail].sum()),'z_star':float(z),'budget':B,'continuous_cost':float(np.sum(q[tail]*kcont[tail])),'discrete_cost':float(np.sum(q[tail]*kdisc[tail])),'k_continuous':kcont.tolist(),'k_discrete':kdisc.tolist(),'k_discrete_unique':sorted(set(kdisc.tolist())),'k_dispersion':float(np.std(kdisc[tail])),'tail_risk_global_k1':float(np.max(b[tail]*rho[tail]**2)),'tail_risk_kmfaps':float(np.max(riskg)),'tail_risk_reduction_vs_global_k1':float(np.max(b[tail]*rho[tail]**2)/(np.max(riskg)+1e-12)),'kappa_proxy_range':[float(kap.min()),float(kap.max())]}
rows=[]
for p in [D/'TFGPS_RAW_PILOT_p0_s3407.json',D/'TFGPS_RAW_PILOT_p1_s3407.json']:
 r=json.loads(p.read_text()); counts=np.asarray(r['counts'],float); A=np.asarray(r['class_proxy']['A'],float); sig=np.asarray(r['class_proxy']['sigma'],float); rho=np.asarray(r['class_proxy']['rho'],float)
 # Replace the clipped one-step rho on p0 tail classes with the executed
 # multi-step rho probe when available; p1 remains the one-step proxy.
 rho_file=D/'KAPPA_RHO_MULTI_STEP_P0.json'
 if 'p0_s3407' in p.stem and rho_file.exists():
  rr=json.loads(rho_file.read_text())
  for x in rr['rows']: rho[int(x['class'])]=float(x['rho12'])
 row=solve(A,sig,rho,counts); row.update({'endpoint':p.stem,'status':'PASS','rho_source':'multi_step_probe_for_p0_tail' if 'p0_s3407' in p.stem else 'one_step_proxy_for_p1','scope':'FIT class proxies; threshold allocation only; no model update/HELD'}); rows.append(row)
result={'status':'PASS','kind':'KAPPA_MINIMAX_ALLOCATION_CONTRACT','scope':'FIT proxy allocation; no HELD reads; no end-to-end claim','tau':TAU,'gamma':GAMMA,'budget':B,'k_grid':KGRID.tolist(),'rows':rows,'interpretation':'A nonconstant k vector and lower proxy worst-tail risk are necessary but not sufficient for a real algorithm claim.'}
(D/'KAPPA_MINIMAX_CONTRACT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
md=['# K-MFAPS proxy allocation contract','',f"Status: **{result['status']}**",'', '| endpoint | k unique | dispersion | global k=1 risk | K-MFAPS risk | reduction |', '|---|---:|---:|---:|---:|---:|']
for x in rows: md.append(f"| {x['endpoint']} | {x['k_discrete_unique']} | {x['k_dispersion']:.4f} | {x['tail_risk_global_k1']:.4f} | {x['tail_risk_kmfaps']:.4f} | {x['tail_risk_reduction_vs_global_k1']:.3f}x |")
md += ['', '- This is a FIT proxy allocation contract, not a training result.', '- The algorithm is only interesting if k dispersion persists and matched-compute updates improve held-out action utility.']
(D/'KAPPA_MINIMAX_CONTRACT.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'status':result['status'],'rows':rows},ensure_ascii=False))
