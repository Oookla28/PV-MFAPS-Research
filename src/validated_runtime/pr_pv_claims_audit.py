#!/usr/bin/env python3
import argparse, json, math, hashlib, statistics
from pathlib import Path
from collections import defaultdict, Counter

def load(p):
    try: return json.loads(Path(p).read_text())
    except Exception as e: return {"_error":repr(e),"_path":str(p)}

def mean(xs):
    ys=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(ys)/len(ys) if ys else None

def sd(xs):
    ys=[float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return (sum((x-mean(ys))**2 for x in ys)/len(ys))**0.5 if ys else None

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def metric(a,k): return a.get("fit_metrics_native_head",{}).get(k)
def heldmetric(h,k): return h.get("held_native_head",{}).get(k)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--out",required=True); ap.add_argument("--summary",required=True); ap.add_argument("--audit-json",required=True); ap.add_argument("--audit-md",required=True); ap.add_argument("--policy",default=None); ap.add_argument("--preflight",default=None); ap.add_argument("--registration",default=None); ap.add_argument("--selection",default=None); ap.add_argument("--gate",default=None); args=ap.parse_args()
    root=Path(args.root); out=Path(args.out)
    files=sorted((out/"matched").glob("*.json")); artifacts=[load(p) | {"_file":str(p)} for p in files]
    freeze=load(out/"PR_FIT_FREEZE_MANIFEST.json"); held=load(out/"HELD_ONE_TIME_EVAL.json")
    sel=load(args.selection or (out/"PR_PV_MFAPS_POLICY_SELECTION_EPS001.json"))
    reg=load(args.registration or (out/"PR_PV_MFAPS_REGISTRATION.json"))
    pre=load(args.preflight or (out/"PR_PV_MFAPS_PREFLIGHT.json"))
    policy=load(args.policy or (out/"action_policy/PR_POLICY_eps001.json"))
    gate=load(args.gate or (out/"partition_heterogeneity_gate/PR_PARTITION_HETEROGENEITY_GATE.json"))
    errors=[]; checks=[]
    # FIT artifact validity
    if len(artifacts)!=24: errors.append(f"expected 24 matched artifacts, got {len(artifacts)}")
    if any(a.get("status")!="EXECUTED" for a in artifacts): errors.append("one or more matched artifacts not EXECUTED")
    if any(a.get("held_reads")!=0 for a in artifacts): errors.append("FIT artifact held_reads nonzero")
    keys=[(a.get("algorithm_arm") or a.get("arm"),a.get("partition"),a.get("training_seed")) for a in artifacts]
    if len(set(keys))!=24: errors.append("duplicate FIT identity keys")
    else: checks.append("24 unique FIT identities; held_reads=0")
    # Freeze/held gates
    fexec=freeze.get("execution_manifest",{})
    if freeze.get("status")!="EXECUTED" or fexec.get("artifact_count")!=24 or len(fexec.get("invalid",[]))!=0 or len(fexec.get("missing_state",[]))!=0 or len(fexec.get("held_leaks",[]))!=0:
        errors.append("PR FIT freeze gate failed")
    else: checks.append("PR FIT freeze EXECUTED, 24 artifacts, no invalid/missing/held leak")
    hman=held.get("execution_manifest",{})
    if held.get("status")!="EXECUTED" or held.get("held_reads")!=25104 or hman.get("n_states")!=24 or hman.get("held_eval_passes")!=24 or hman.get("errors"):
        errors.append("PR one-time HELD gate failed")
    else: checks.append("one-time HELD EXECUTED, 24 states, 25104 reads, zero errors")
    hres=held.get("results",[])
    hkeys=[(r.get("arm"),r.get("partition"),r.get("training_seed")) for r in hres]
    if len(hres)!=24 or len(set(hkeys))!=24: errors.append("HELD identity coverage/collision failure")
    else: checks.append("HELD identity coverage unique across PR and controls")
    # policy and registration
    if sel.get("status")!="EXECUTED" or sel.get("selected_epsilon")!=0.01 or sel.get("held_reads")!=0: errors.append("FIT-only epsilon selection invalid")
    else: checks.append("FIT-only epsilon selection selected epsilon=0.01 before HELD")
    if policy.get("status")!="EXECUTED" or policy.get("held_reads")!=0: errors.append("selected PR policy invalid")
    cert=policy.get("certificate",{}); kgrid=policy.get("registered_config",{}).get("k_grid",[0,1,2,4]); kh=policy.get("k_hist",{})
    if not cert.get("exact_quantized",False) or set(map(int,kh.keys()))-set(kgrid): errors.append("policy certificate/grid inconsistency")
    else: checks.append(f"shared finite-action DP certificate present; k_hist={kh}; exact quantized={cert.get('exact_quantized')}")
    if pre.get("status")!="EXECUTED": errors.append("PR preflight not EXECUTED")
    if gate.get("status") not in ("EXECUTED","PASS") and gate.get("gate_status") not in ("PASS","EXECUTED"): errors.append("partition heterogeneity gate unavailable")
    else: checks.append("partition heterogeneity gate supported PR registration")
    # aggregate FIT/HELD
    byheld={(r.get("arm"),r.get("partition"),r.get("training_seed")):r for r in hres}
    groups=defaultdict(list)
    for a in artifacts: groups[a.get("algorithm_arm") or a.get("arm")].append(a)
    rows=[]
    for arm,aa in sorted(groups.items()):
        hh=[byheld.get((arm,a.get("partition"),a.get("training_seed"))) for a in aa]
        rows.append({"arm":arm,"n":len(aa),"fit_macro":mean([metric(a,"macro_f1") for a in aa]),"fit_macro_sd":sd([metric(a,"macro_f1") for a in aa]),"fit_tail":mean([metric(a,"tail_recall_worst10") for a in aa]),"fit_tail_sd":sd([metric(a,"tail_recall_worst10") for a in aa]),"held_macro":mean([heldmetric(h,"macro_f1") if h else None for h in hh]),"held_macro_sd":sd([heldmetric(h,"macro_f1") if h else None for h in hh]),"held_tail":mean([heldmetric(h,"tail_recall_worst10") if h else None for h in hh]),"held_tail_sd":sd([heldmetric(h,"tail_recall_worst10") if h else None for h in hh]),"wall":mean([a.get("wall_clock_training_seconds") for a in aa]),"flops":mean([a.get("estimated_training_flops") for a in aa]),"fused_all":all(a.get("pcfa_implementation_variant")=="fused_single_encoder_graph" for a in aa if arm=="pr_pv_mfaps")})
    rmap={r["arm"]:r for r in rows}; pr=rmap.get("pr_pv_mfaps"); fx=rmap.get("bs_fixed_k_matched_B02"); eff=rmap.get("effective_number")
    # paired cost
    prmap={(a.get("partition"),a.get("training_seed")):a for a in artifacts if (a.get("algorithm_arm") or a.get("arm"))=="pr_pv_mfaps"}; fxmap={(a.get("partition"),a.get("training_seed")):a for a in artifacts if a.get("arm")=="bs_fixed_k_matched_B02"}
    ratios=[]
    for k,a in sorted(prmap.items()):
        b=fxmap.get(k)
        if not b: errors.append(f"missing fixed-k pair {k}"); continue
        ratios.append({"partition":k[0],"seed":k[1],"flops_ratio":a.get("estimated_training_flops")/b.get("estimated_training_flops"),"wall_ratio":a.get("wall_clock_training_seconds")/b.get("wall_clock_training_seconds"),"pr_flops":a.get("estimated_training_flops"),"fixed_flops":b.get("estimated_training_flops")})
    maxf=max((x["flops_ratio"] for x in ratios),default=None); maxw=max((x["wall_ratio"] for x in ratios),default=None)
    tail_gain=bool(pr and fx and pr["held_tail"] is not None and fx["held_tail"] is not None and pr["held_tail"]>fx["held_tail"])
    macro_ok=bool(pr and fx and pr["held_macro"] is not None and fx["held_macro"] is not None and pr["held_macro"]>=fx["held_macro"])
    wall_ok=maxw is not None and maxw<=1.05+1e-12; flops_ok=maxf is not None and maxf<=1.05+1e-12
    gates=[]
    if tail_gain and macro_ok and wall_ok and flops_ok: gates.append("ALGORITHM_CLAIM_REOPENED")
    else:
        gates.append("ALGORITHM_BOUNDARY: PR held tail/macro/cost joint gate not passed")
        if not tail_gain: gates.append("NO_TAIL_GAIN: PR HELD tail-10% is not above fixed-k")
        elif not macro_ok: gates.append("TAIL_MACRO_TRADEOFF: PR tail up but HELD Macro-F1 below fixed-k")
        if not wall_ok: gates.append("WALL_BUDGET_NOT_PASSED")
        if not flops_ok: gates.append("FLOPS_BUDGET_NOT_PASSED")
    if pr and fx and pr["fit_tail"]>fx["fit_tail"] and not tail_gain: gates.append("FIT_TAIL_UP_HELD_NONREPRO")
    # Explicit duplicate baseline caveat
    if "bs_pcfa" in rmap and "pr_pv_mfaps" in rmap and abs((rmap["bs_pcfa"]["held_macro"] or 0)-(rmap["pr_pv_mfaps"]["held_macro"] or 0))<1e-12: gates.append("DUPLICATE_CONTROL: bs_pcfa used same PR policy/action path; not an independent algorithm baseline")
    checks.append(f"paired PR/fixed cost ratios computed for {len(ratios)} partition-seed pairs")
    summary=["# PR-PV-MFAPS Final Summary","","Status: deterministic execution and claims summary; semantic reviewer unavailable.","","The final PR artifact is the robust-risk/shared-cost finite-action variant. Its KL dual and two-partition risk-vector Pareto DP are registered; the empirical generalization theorem and alpha-aware optimality remain unverified.","","## Matched FIT/HELD results","","| arm | n | FIT Macro-F1 | FIT tail-10% | HELD Macro-F1 | HELD tail-10% | wall s | FLOPs |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows: summary.append(f"| {r['arm']} | {r['n']} | {r['fit_macro']:.4f} | {r['fit_tail']:.4f} | {r['held_macro']:.4f} | {r['held_tail']:.4f} | {r['wall']:.4f} | {r['flops']:.3e} |")
    summary += ["","## Paired PR/fixed-k cost ratios","","| partition | seed | FLOPs ratio | wall ratio |","|---:|---:|---:|---:|"]
    for x in ratios: summary.append(f"| {x['partition']} | {x['seed']} | {x['flops_ratio']:.4f} | {x['wall_ratio']:.4f} |")
    summary += ["",f"max FLOPs ratio **{maxf:.4f}**, max wall ratio **{maxw:.4f}**.","", "## Automated gates"]+[f"- {g}" for g in gates]+["","## Evidence boundary","- FIT artifacts were frozen with held_reads=0 before the corrected one-time HELD read.","- HELD was read once for the corrected arm-labelled freeze; earlier invalid HELD attempts are quarantined and excluded.","- The shared finite-action policy uses a two-partition risk vector and quantized cost; robust-cost and generalization claims are not made.","- No SOTA, universal improvement, cross-task transfer, or independent semantic-review claim.","- `bs_pcfa` is retained as a control artifact but is not independent of PR when run with the same action policy."]
    Path(args.summary).write_text("\n".join(summary)+"\n")
    classification={"EXECUTED":["partition heterogeneity gate","PR registration/preflight","math smoke","FIT-only epsilon selection","PR-only 6 matched arms","18 controls","corrected FIT freeze (24 states)","corrected one-time HELD (24 states, 25104 reads)","deterministic claims audit"],"REGISTERED_NOT_EXECUTED":["CIFAR-LT/ImageNet-LT cross-domain runner","learnable Power(p) geometry","independent second SPD project"],"UNVERIFIED":["PR generalization theorem","SOTA/universal improvement","cross-task transfer","semantic review"],"INVALID_CONFIG":["contaminated PR/bs_pcfa collision runs","first HELD attempt without PLANT_E01_ROOT","second HELD attempt with duplicate arm labels"],"BLOCKED_ENV":[]}
    audit={"status":"EXECUTED" if not errors else "INVALID_CONFIG","scope":"PR-PV-MFAPS robust-risk/shared-cost final claims-evidence audit","held_reads":held.get("held_reads",0),"registered_config":{"variant":"robust-risk/shared-cost finite-action PR-PV-MFAPS","selected_epsilon":0.01,"k_grid":kgrid,"budget":policy.get("registered_config",{}).get("budget"),"cost_semantics":"shared measured q*k proxy; no robust cost claim","dp_state":"two-partition risk-vector Pareto frontier","held_policy":"single post-freeze read","semantic_review":"REVIEW_UNAVAILABLE"},"execution_manifest":{"checks":checks,"errors":errors,"n_artifacts":len(artifacts),"n_states":hman.get("n_states"),"held_eval_passes":hman.get("held_eval_passes"),"summary":rows,"cost_ratios":ratios,"max_flops_ratio":maxf,"max_wall_ratio":maxw,"gates":gates,"duplicate_control_not_independent":True},"coverage_status":{"status":"EXECUTED_HELD_UNVERIFIED_EFFICACY","note":"Execution chain is complete; algorithm efficacy/generalization remains bounded by reported held metrics and gates."},"classification":classification,"review_status":"REVIEW_UNAVAILABLE"}
    Path(args.audit_json).write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n")
    md=["# PR-PV-MFAPS Claims/Evidence Audit","",f"Status: **{audit['status']}**","","Deterministic audit only; semantic reviewer: **REVIEW_UNAVAILABLE**.","","## Checks"]+[f"- {c}" for c in checks]+["","## Errors"]+([f"- {e}" for e in errors] or ["- none"])+["","## Gates"]+[f"- {g}" for g in gates]+["","## Classification"]+[f"- **{k}**: {len(v)}" for k,v in classification.items()]+["","No SOTA, generalization, universal improvement, or cross-task claim is supported."]
    Path(args.audit_md).write_text("\n".join(md)+"\n")
    print(json.dumps({"status":audit["status"],"errors":len(errors),"held_reads":held.get("held_reads"),"gates":gates,"max_flops_ratio":maxf,"max_wall_ratio":maxw}))
if __name__=="__main__": main()
