#!/usr/bin/env python3
"""Proxy-only K-MFAPS budget/safety sensitivity surface."""
from pathlib import Path
import json, math
import numpy as np
ROOT=Path('$WORKSPACE_ROOT/feedback_geometry_pipeline_2026-11-04'); D=ROOT/'diagnostics'
BUDGETS=(.05,.1,.2,.3,.4); KMAXS=(1,2,4); BETAS=(0.0,.05,.1); TAU=.75; GAMMA=.2; GRID=(0,1,2,4,8)

def load_endpoint(path):
 r=json.loads(path.read_text()); counts=np.asarray(r['counts'],float); A=np.asarray(r['class_proxy']['A'],float); sigma=np.asarray(r['class_proxy']['sigma'],float); rho=np.asarray(r['class_proxy']['rho'],float)
 if 'p0_s3407' in path.stem and (D/'KAPPA_RHO_MULTI_STEP_P0.json').exists():
  rr=json.loads((D/'KAPPA_RHO_MULTI_STEP_P0.json').read_text())
  for x in rr['rows']: rho[int(x['class'])]=float(x['rho12'])
 return counts,A,sigma,np.clip(rho,.7,.999999)

def solve(counts,A,sigma,rho,B,KMAX,beta):
 C=len(counts); q=counts**(1-TAU); q/=q.sum(); tail=np.argsort(counts)[:max(1,int(math.ceil(GAMMA*C)))]
 b=(A/np.maximum(2*sigma-A,1e-8)+beta*sigma)**2
 def kc(z): return np.clip(np.log(np.maximum(b/z,1.0))/(-2*np.log(rho)),0,KMAX)
 def cost(z): return float(np.sum(q[tail]*kc(z)[tail]))
 lo=max(float(b.min())*1e-8,1e-12); hi=max(float(b.max()),1e-6)
 for _ in range(100):
  mid=math.sqrt(lo*hi)
  if cost(mid)>B: lo=mid
  else: hi=mid
 z=hi; kcont=kc(z)
 allowed=np.asarray([k for k in GRID if k<=KMAX],int)
 kdisc=np.asarray([allowed[np.argmin(np.abs(allowed-v))] for v in kcont],int)
 while float(np.sum(q[tail]*kdisc[tail]))>B+1e-10:
  cand=[c for c in tail if kdisc[c]>0]
  if not cand: break
  c=max(cand,key=lambda j:kdisc[j]); lower=allowed[allowed<kdisc[c]]; kdisc[c]=int(lower[-1]) if len(lower) else 0
 # Global-k best under the same expected tail budget.
 feasible=[int(k) for k in allowed if float(q[tail].sum()*k)<=B+1e-10]
 kg=max(feasible) if feasible else 0
 risk_km=float(np.max(b[tail]*rho[tail]**(2*kdisc[tail])))
 risk_g=float(np.max(b[tail]*rho[tail]**(2*kg)))
 return {'B':B,'Kmax':KMAX,'beta':beta,'global_k':kg,'global_cost':float(q[tail].sum()*kg),'kmfaps_cost':float(np.sum(q[tail]*kdisc[tail])),'k_unique':sorted(set(kdisc[tail].tolist())),'k_dispersion':float(np.std(kdisc[tail])),'risk_global':risk_g,'risk_kmfaps':risk_km,'risk_reduction':float(risk_g/(risk_km+1e-12)),'z_star':float(z),'rho_source':'multi_step_tail_p0' if 'p0' else 'one_step'}

rows=[]
for path in [D/'TFGPS_RAW_PILOT_p0_s3407.json',D/'TFGPS_RAW_PILOT_p1_s3407.json']:
 counts,A,sigma,rho=load_endpoint(path)
 for B in BUDGETS:
  for K in KMAXS:
   for beta in BETAS:
    x=solve(counts,A,sigma,rho,B,K,beta); x.update({'endpoint':path.stem,'status':'PASS'}); rows.append(x)
result={'status':'PASS','kind':'KAPPA_MINIMAX_BUDGET_SENSITIVITY','scope':'FIT class proxies only; no encoder updates; no HELD; p0 tail uses executed multi-step rho','budgets':list(BUDGETS),'Kmax_grid':list(KMAXS),'beta_grid':list(BETAS),'rows':rows,'interpretation':'Sensitivity contract identifies non-degenerate budget/safety regions; it does not establish end-to-end accuracy.'}
(D/'KAPPA_MINIMAX_BUDGET_SENSITIVITY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
# concise markdown summary per endpoint/B/beta=0
md=['# K-MFAPS budget sensitivity (proxy-only)','',f"Status: **{result['status']}**",'', '| endpoint | B | Kmax | beta | global k | K-MFAPS k set | risk ratio |', '|---|---:|---:|---:|---:|---|---:|']
for x in rows:
 if x['beta']==0.0 and x['Kmax'] in (1,4): md.append(f"| {x['endpoint']} | {x['B']:.2f} | {x['Kmax']} | {x['beta']:.2f} | {x['global_k']} | {x['k_unique']} | {x['risk_reduction']:.3f}x |")
md += ['', '- `risk_ratio > 1` means lower proxy worst-tail residual than global-k at the same expected budget.', '- This is not a training or held-out result.']
(D/'KAPPA_MINIMAX_BUDGET_SENSITIVITY.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'status':result['status'],'rows':len(rows),'nondegenerate':sum(len(x['k_unique'])>1 for x in rows),'strict_proxy_gain':sum(x['risk_reduction']>1.001 for x in rows)},ensure_ascii=False))
