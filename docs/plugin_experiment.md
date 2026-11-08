# Balanced Softmax plugin

The plugin line studies a long-tail objective as a drop-in target-function component. The Balanced Softmax prior is applied consistently to native loss, reference probe loss, detached inner-solver loss, and the fused correction loss. This keeps the feedback comparison on the same label objective.

The plugin runner shares the checkpoint, backbone, optimizer, steps, batch size, partitions, seeds and cost accounting with the main line. Its arm list is declared in `configs/plugin/bs_pcfa.yaml`. The action policy files are the FIT-only policies under `configs/policies/`.

Run the plugin after the core chain has its own output directory.

~~~bash
export DATA_ROOT=/path/to/plantspd-data
export OUTPUT_ROOT=/path/to/output/plugin
PYTHONPATH=src/validated_runtime python src/validated_runtime/bs_plugin_runner.py \
  --root "$DATA_ROOT" --out "$OUTPUT_ROOT" \
  --arms balanced_softmax,bs_fixed_k_matched_B02,bs_pcfa,native_balanced,effective_number,logit_adjustment \
  --seeds 3407,3408,3409
PYTHONPATH=src/validated_runtime python src/validated_runtime/make_freeze_manifest.py \
  --root "$OUTPUT_ROOT" --output "$OUTPUT_ROOT/FIT_FREEZE_MANIFEST.json"
PYTHONPATH=src/validated_runtime python src/validated_runtime/held_eval.py \
  --freeze-manifest "$OUTPUT_ROOT/FIT_FREEZE_MANIFEST.json" \
  --output "$OUTPUT_ROOT/HELD_EVAL.json"
~~~

The plugin results remain separate from the primary PR-PV-MFAPS evidence. The
plugin registration document records its shared inputs and arm semantics.
