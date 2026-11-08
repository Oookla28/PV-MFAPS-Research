# Balanced Softmax plugin

The plugin line studies a long-tail objective as a drop-in target-function component. The Balanced Softmax prior is applied consistently to native loss, reference probe loss, detached inner-solver loss, and the fused correction loss. This keeps the feedback comparison on the same label objective.

The plugin runner shares the checkpoint, backbone, optimizer, steps, batch size, partitions, seeds and cost accounting with the main line. Its arm list is declared in `configs/plugin/bs_pcfa.yaml`. The action policy files are the FIT-only policies under `configs/policies/`.

Run the plugin after the core chain has its own output directory.

```bash
export OUTPUT_ROOT=$REPO_ROOT/artifacts/results/run_plugin
bash scripts/run_plugin.sh
export RUN_DIR=$OUTPUT_ROOT/plugin
bash scripts/freeze.sh
SPLIT=held bash scripts/evaluate.sh
```

The plugin results are read from `artifacts/results/plugin/` and remain separate from the primary PR-PV-MFAPS evidence. The plugin registration document records its shared inputs and arm semantics.
