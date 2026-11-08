#!/usr/bin/env python3
"""Single one-time HELD evaluation after the FIT freeze manifest is PASS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tfgps_gpu_lib import (
    E0, choose_device, counts_and_prior, eval_model, hash_file, set_deterministic,
    write_json,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    freeze = json.loads(Path(args.freeze_manifest).read_text())
    if freeze.get("status") not in ("PASS", "EXECUTED"):
        raise SystemExit("freeze manifest is not PASS; refusing HELD read")
    executed = freeze["execution_manifest"]["executed"]
    if not executed:
        raise SystemExit("no executed FIT artifacts")
    for e in executed:
        obj = json.loads(Path(e["path"]).read_text())
        if obj.get("held_reads") != 0:
            raise SystemExit(f"held_reads already nonzero: {e['path']}")
    device = choose_device(args.gpu)
    set_deterministic(3407)
    results = []
    errors = []
    held_n = None
    for e in executed:
        obj = json.loads(Path(e["path"]).read_text())
        state_path = Path(e["state_path"])
        st = torch.load(state_path, map_location="cpu")
        method = obj["method"]; part = int(obj["partition"]); ckpt_seed = int(obj["checkpoint_seed"])
        try:
            runner, base, ckpt, payload = E0.load_checkpoint(method, ckpt_seed, part, device=str(device))
            missing, unexpected = base.load_state_dict(st["encoder"], strict=False)
            base.eval()
            for module in base.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            head = nn.Linear(int(st["head_in_features"]), int(st["head_out_features"])).to(device)
            head.load_state_dict(st["head"])
            head.eval()
            x, labels, fit, held = E0.load_data(part)
            E0.labels_global = labels
            held = np.asarray(held, dtype=np.int64)
            if held_n is None:
                held_n = int(len(held))
            elif held_n != int(len(held)):
                errors.append(f"held count changed {held_n} -> {len(held)}")
            fit = np.asarray(fit, dtype=np.int64)
            counts, _ = counts_and_prior(labels, fit, int(base.n_classes))
            log_prior = None
            policy = st.get("policy", {})
            if obj.get("arm") == "logit_adjustment" and "log_prior" in policy:
                log_prior = torch.as_tensor(policy["log_prior"], dtype=torch.float32, device=device)
            fixed = eval_model(base, runner, x, labels, held, head, method, device, counts,
                               batch_size=32, use_fixed_head=True)
            native = eval_model(base, runner, x, labels, held, None, method, device, counts,
                                batch_size=32, log_prior=log_prior, use_fixed_head=False)
            results.append({
                "arm": obj.get("arm"), "method": method, "phase": obj.get("phase"),
                "partition": part, "checkpoint_seed": ckpt_seed,
                "training_seed": obj.get("training_seed"),
                "state_path": str(state_path), "state_sha256": hash_file(state_path),
                "fit_artifact": e["path"],
                "held_fixed_head": fixed, "held_native_head": native,
                "load_missing_keys": list(missing), "load_unexpected_keys": list(unexpected),
            })
        except Exception as exc:
            errors.append({"path": e["path"], "error": repr(exc)})
    out = {
        "status": "EXECUTED" if results and not errors else "UNVERIFIED",
        "scope": "one-time HELD evaluation after all FIT artifacts were frozen; no fitting or selection",
        "held_reads": int((held_n or 0) * len(results)),
        "registered_config": {
            "freeze_manifest": str(args.freeze_manifest),
            "freeze_manifest_sha256": hash_file(args.freeze_manifest),
            "held_policy": "single read after FIT freeze; fixed FIT-calibrated head; no HELD tuning",
            "eval_classifier_primary": "FIT-calibrated fixed affine reference head",
            "eval_classifier_secondary": "native dual head (with logit adjustment only for the LA arm)",
        },
        "execution_manifest": {
            "n_states": len(results), "held_unique_samples": int(held_n or 0),
            "held_eval_passes": len(results), "errors": errors,
            "device": str(device),
        },
        "coverage_status": {
            "status": "proxy_only_unverified",
            "note": "HELD is a single post-freeze read; it is not a cross-task or SOTA claim",
        },
        "results": results,
    }
    write_json(args.output, out)
    print(json.dumps({"status": out["status"], "held_reads": out["held_reads"],
                      "n_states": len(results), "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
