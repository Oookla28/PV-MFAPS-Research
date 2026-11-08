# Top-conference upgrade tracker

| ID | Block | Status | Required evidence |
|---|---|---|---|
| A1 | CIFAR-100-LT 30-epoch Fused | EXECUTED | Current metrics and held evaluation |
| A2 | LDAM-DRW custom recipe | EXECUTED / FAILED INTEGRATION | Must not be used as a valid baseline |
| A3 | Legacy PV 30-epoch | RUNNING | Same seeds/actions; physical forward and CUDA time |
| A4 | Official LDAM-DRW recipe | REGISTERED_NOT_EXECUTED | Official-equivalent implementation and sanity curve |
| B1 | Hard-floor stage rule | REGISTERED_NOT_EXECUTED | Negative control for forced k>=1 |
| B2 | Certified marginal-gain gate | REGISTERED_NOT_EXECUTED | Fresh calibration, activation inequality |
| B3 | Stage-aware robust DP | CPU_EXECUTED + GPU_SMOKE_EXECUTED / MATCHED_REGISTERED_NOT_EXECUTED | DP exactness and 8-epoch FIT-only smoke passed, held evaluation pending |
| C1 | 8/30/50 horizon | REGISTERED_NOT_EXECUTED | Fixed budget and per-class tail overlap |
| D1 | CIFAR-10-LT IF=100 | REGISTERED_NOT_EXECUTED | Same code, pre-registered seeds |
| E1 | Fused vs Legacy | CPU/GPU EXECUTED | One-step equivalence passed; sequential seed timing complete; full trajectory equivalence remains unclaimed |
| E2 | torch.compile/CUDA Graph control | REGISTERED_NOT_EXECUTED | Compiler baseline or explicit unavailable status |
| F1 | Five-seed confirmation | REGISTERED_NOT_EXECUTED | Frozen hyperparameters, confidence intervals |
