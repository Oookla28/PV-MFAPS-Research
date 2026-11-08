# Figures

Run:

```bash
python figures/generate_figures.py
```

The script reads the frozen JSON summaries in `artifacts/results/` and writes
vector PDF figures. The fused-system numbers are stored in
`fused_system_data.json`, which records the real-checkpoint equivalence smoke
used for the system plot. No training or evaluation result is hard-coded in the
plotting logic.
