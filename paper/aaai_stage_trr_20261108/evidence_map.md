# Draft evidence map

This draft uses only frozen artifacts and explicitly labels exploratory results.

| Section | Claim | Evidence |
|---|---|---|
| Feedback diagnosis | LA leaves a measurable backward gap | TRY/work/results/diag_param_s3407.json |
| CIFAR-100-LT 30 | Stage-TRR-10 improves Macro and Tail | artifacts/results/TRR_AUDITED_PREFIX10_VS_LA_SUMMARY.json |
| CIFAR-10-LT 30 | Fixed policy transfers to another long-tailed benchmark | artifacts/results/TRR_AUDITED_C10_PREFIX10_VS_LA_SUMMARY.json |
| Horizon boundary | 50 epoch Tail gain disappears | artifacts/results/TRR_AUDITED_HORIZON50_SUMMARY.json |
| PlantSPD | Real-scene SPD branch and fused execution | artifacts/results/pcfa_fused_latency_2026-11-05/ and PlantSPD summaries |
| Theory | Conditional dynamic-stage surrogate bound | docs/ftrr_stage_switching_bound_20261108.md |

## Evidence states

* Verified means the value is present in an executed JSON artifact with a freeze and
  one-time evaluator record.
* Exploratory means the final group is internally audited but a pilot or direct
  test read influenced the choice of a setting.
* Unverified means the experiment is planned or the result cannot be traced to
  a frozen artifact.
