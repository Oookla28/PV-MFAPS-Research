# CIFAR-100-LT IR=100 protocol

This benchmark uses the official CIFAR-100 Python archive. The public
training split contains 50,000 images, with 500 images per fine-label class.
For imbalance ratio 100, class `c` receives

\[
n_c=\operatorname{round}\left(500\,100^{-c/99}\right),\qquad c=0,\ldots,99.
\]

The resulting long-tailed training set has 100 classes, counts from 500 down
to 5, and 10,899 selected images. The official 10,000-image test split is
balanced and is never opened by the training runner.

## Shared training contract

The ERM, Balanced Softmax, Logit Adjustment, LDAM-DRW, fixed-k and PV-MFAPS
arms use the same class construction, random seeds, CIFAR augmentation,
batch size, SGD optimizer, selected learning-rate schedule, and model. The
baseline runner writes FIT-only JSON with `held_reads: 0`. The standard 200-epoch matrix uses a 5-epoch linear warmup and decays at epochs 160 and 180; historical short runs retain cosine scheduling. `evaluate.py` is a
separate post-freeze entry point.

The 200-epoch matrix uses the CIFAR ResNet-32 setting (`--depth 32
--width 16`) with zero-padding augmentation. This matches the depth and
width convention used by the public LDAM-DRW reference. The older 30/50-epoch
summaries used the repository's wider depth-14 model and are retained as a
separate historical protocol. The new matrix selects five-epoch linear
warmup followed by 160/180 multistep decays, matching the reference schedule.

The four standard long-run comparators are:

- `erm`: ordinary cross-entropy.
- `balanced_softmax`: cross-entropy on `logits + log(n_c)`.
- `logit_adjustment`: cross-entropy on `logits + tau * log(n_c / sum_j n_j)`.
  At `tau=1`, this objective is mathematically identical to Balanced Softmax
  up to a class-independent constant. Both names remain available so that the
  implementation and literature terminology can be compared explicitly.
- `ldam_drw`: LDAM margins with deferred effective-number reweighting. The
  200-epoch configuration switches the reweighting on at epoch 160, matching
  the public LDAM-DRW reference schedule.

Stage-TRR is a separate runner because it changes the encoder feedback path.
It uses the same long-tailed samples and optimizer family, with a diagnostic
calibration subset that is FIT-only. `--trr-epochs 10` means ten initial
epochs with detached one-step readout refinement followed by joint Logit
Adjustment training.

## 1. Environment and data

```bash
python -m pip install -r requirements.txt
export REPO_ROOT=/path/to/PV-MFAPS-Research
export CIFAR_ROOT=/path/to/cifar-cache
```

`ensure_data` downloads and extracts the official archive into
`$CIFAR_ROOT` when it is absent. For a disconnected machine, place
`cifar-100-python.tar.gz` in that directory before running the commands.

## 2. Preflight smoke

This command reads the train batch only, checks the 100-class IR=100 count
contract, runs one finite-loss backward pass for all four standard losses, and
writes a machine-readable manifest. It does not read the test batch.

```bash
export PREFLIGHT_OUT=$REPO_ROOT/artifacts/runs/cifar100_ir100_200/preflight.json
PYTHONPATH=$REPO_ROOT/experiments/cifar100_lt \
python $REPO_ROOT/experiments/cifar100_lt/cifar100_lt_preflight.py \
  --data-root "$CIFAR_ROOT" \
  --output "$PREFLIGHT_OUT" \
  --imbalance-ratio 100 --seed 3407 --gpu 0
```

The preflight must report `class_count: 100`, `counts_min: 5`,
`counts_max: 500`, `train_n_long_tail: 10899`, and `held_reads: 0`.

## 3. Standard 200-epoch baseline matrix

Run the four baseline arms together so that they share the same data and
optimizer configuration. The output directory must be new and physically
separate from any earlier 30/50-epoch runs.

```bash
export BASELINE_OUT=$REPO_ROOT/artifacts/runs/cifar100_ir100_200/baseline_fused
PYTHONPATH=$REPO_ROOT/experiments/cifar100_lt \
python $REPO_ROOT/experiments/cifar100_lt/cifar100_lt_runner.py \
  --data-root "$CIFAR_ROOT" \
  --output "$BASELINE_OUT" \
  --arms erm,balanced_softmax,ldam_drw,logit_adjustment \
  --seeds 3407,3408,3409 \
  --epochs 200 --batch-size 128 --lr 0.1 \
  --depth 32 --width 16 --augmentation zero \
  --schedule standard_multistep --warmup-epochs 5 --milestones 160 180 \
  --weight-decay 0.0002 \
  --ldam-switch-epoch 160 --ldam-C 0.5 --ldam-beta 0.9999 \
  --ldam-scale 30 --logit-adjustment-tau 1.0 \
  --execution fused --gpu 0
```

After every checkpoint has been written, read the test split once for this
matrix:

```bash
PYTHONPATH=$REPO_ROOT/experiments/cifar100_lt \
python $REPO_ROOT/experiments/cifar100_lt/evaluate.py \
  --run-dir "$BASELINE_OUT" --data-root "$CIFAR_ROOT" \
  --output "$BASELINE_OUT/HELD_EVAL.json" --gpu 0
```

Before the evaluator runs, all per-arm JSON files should still contain
`held_reads: 0`. The evaluator records top-1 accuracy, macro recall, macro F1, tail-10% recall, tail-10% F1, wall time and estimated FLOPs for each arm and seed.

## 4. Stage-TRR-10 at the same horizon

Run each Stage-TRR seed into one shared directory. This keeps the evaluator's
input set explicit while preserving separate checkpoint names.

```bash
export TRR_OUT=$REPO_ROOT/artifacts/runs/cifar100_ir100_200/stage_trr10
mkdir -p "$TRR_OUT"
for seed in 3407 3408 3409; do
  PYTHONPATH=$REPO_ROOT/experiments/cifar100_lt \
  python $REPO_ROOT/experiments/cifar100_lt/trr_audited_runner.py \
    --data-root "$CIFAR_ROOT" --output "$TRR_OUT" \
    --epochs 200 --trr-epochs 10 --batch-size 128 \
    --depth 32 --width 16 --augmentation zero \
    --schedule standard_multistep --warmup-epochs 5 --milestones 160 180 \
    --weight-decay 0.0002 \
    --lr-encoder 0.1 --lr-head 0.05 \
    --calibration-fraction 0.1 --seed "$seed" --gpu 0
done
```

The corresponding LA-only control uses the same command with
`--trr-epochs 0`, written to a different directory. Evaluate the Stage-TRR
directory after all three seeds finish:

```bash
PYTHONPATH=$REPO_ROOT/experiments/cifar100_lt \
python $REPO_ROOT/experiments/cifar100_lt/trr_audited_evaluate.py \
  --run-dir "$TRR_OUT" --data-root "$CIFAR_ROOT" \
  --output "$TRR_OUT/HELD_EVAL.json" --gpu 0
```

## 5. Fused and Legacy system check

The algorithm comparison uses the Fused path. To isolate the execution axis,
run one seed and one short horizon in a separate directory with
`--execution legacy`, then compare the logical action count and checkpoint
equivalence with `equivalence_smoke.py`. Do not mix Legacy artifacts into the
200-epoch accuracy table unless the two paths use the same seed, actions,
optimizer and horizon.

```bash
PYTHONPATH=$REPO_ROOT/experiments/cifar100_lt \
python $REPO_ROOT/experiments/cifar100_lt/cifar100_lt_runner.py \
  --data-root "$CIFAR_ROOT" \
  --output "$REPO_ROOT/artifacts/runs/cifar100_ir100_system/legacy" \
  --arms erm,pv_mfaps --seeds 3407 --epochs 2 --batch-size 128 \
  --execution legacy --gpu 0
```

## 6. Metrics and reporting

Report top-1 accuracy, all-class macro recall, and recall averaged over the
10 least frequent training classes. Report seed mean and standard deviation,
not a single seed. Balanced Softmax and Logit Adjustment with `tau=1` should
be described as equivalent objectives rather than counted as independent
algorithmic wins. The Stage-TRR comparison must keep the 200-epoch horizon and
the same three seeds as the baselines.

The 30/50-epoch runs in `artifacts/results/cifar100_lt/` are historical
reproduction summaries. The 200-epoch matrix described here is a fresh,
independently named run and should be evaluated only after its own freeze.
