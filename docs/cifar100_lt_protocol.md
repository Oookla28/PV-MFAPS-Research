# CIFAR-100-LT extension

This extension uses the public CIFAR-100 train/test split with imbalance ratio 100. The class counts are constructed from 500 examples for class 0 down to 5 examples for class 99, giving 100 classes and approximately 12,631 training samples. The test split remains balanced with 10,000 images.

The registered 30-epoch arms are:

- `erm` as the empirical-sampler reference.
- `ldam_drw` with LDAM margins `m_c = 0.5 / n_c^(1/4)` and deferred re-weighting from epoch 15.
- `logit_adjustment` with fixed `tau=1.0` and the train-prior offset.
- `fixed_k` with one correction action for the registered tail set.
- `pv_mfaps` with the shared finite action policy and the same expected action budget.

The runner has two execution variants. `legacy` repeats the backbone forward for each active correction group. `fused` reuses the feature tensor and runs the correction on the classifier head. The two variants share seeds, data order, optimizer, actions, epochs and batch size. Their equivalence manifest records logical action counts, physical forward counts, checkpoint hashes and CUDA event wall time.

Use the following commands on a CUDA machine.

~~~bash
export DATA_ROOT=/path/to/cifar-cache
export OUTPUT_ROOT=/path/to/output/cifar100_ir100_fused
python experiments/cifar100_lt/cifar100_lt_runner.py \
  --data-root "$DATA_ROOT" --output "$OUTPUT_ROOT" \
  --arms erm,ldam_drw,logit_adjustment,fixed_k,pv_mfaps \
  --seeds 3407,3408,3409 --epochs 30 --batch-size 128 \
  --execution fused --gpu 0
~~~

To measure the execution axis with the same protocol, use a separate output
folder and set execution to legacy.

~~~bash
export OUTPUT_ROOT=/path/to/output/cifar100_ir100_legacy
python experiments/cifar100_lt/cifar100_lt_runner.py \
  --data-root "$DATA_ROOT" --output "$OUTPUT_ROOT" \
  --arms erm,ldam_drw,logit_adjustment,fixed_k,pv_mfaps \
  --seeds 3407,3408,3409 --epochs 30 --batch-size 128 \
  --execution legacy --gpu 0
~~~

The CIFAR extension is a separate benchmark run. Its Fused measurement describes repeated backbone work in this CIFAR runner and does not reuse the PlantSPD timing numbers. Test evaluation was performed after all 30-epoch checkpoints were produced. The final artifacts are under `artifacts/results/cifar100_lt/`, with earlier mismatched Legacy runs retained under explicit invalid-config records.
