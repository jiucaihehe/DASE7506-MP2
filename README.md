# DASE7506 MP2 — Selecting Reliable LLM Reasoning

**Test accuracy: 81.80% (382/467) · validation 93.48% · assets 13.8 KB · CPU evaluation ≈ 0.3 s**

Method: a linear **group-softmax ranker** over 46 hand-crafted features + the
supplied 1024-d candidate embeddings. Training maximizes the probability mass
that a softmax over the four candidate scores assigns to the *set of correct
candidates* per question — directly matching the "any correct pick scores"
rule. Training is deterministic (zero-init full-batch Adam) and uses the train
split only; model selection used hard validation groups only; test references
were read exactly once, by the frozen pipeline.

- **Report:** [`report/REPORT.md`](report/REPORT.md)
- **Method code:** [`code/submission.py`](code/submission.py), [`code/features.py`](code/features.py)
- **Training code:** [`code/train_final.py`](code/train_final.py)
- **Checkpoint (inference asset, 13.8 KB):** [`code/runs/model_bundle.npz`](code/runs/model_bundle.npz)
- **Official evaluation JSONs:** [`code/runs/test-score.json`](code/runs/test-score.json), [`code/runs/validation-score.json`](code/runs/validation-score.json)
- **Experiment log:** [`code/EXPERIMENTS.md`](code/EXPERIMENTS.md)
- Original task materials: [`guide/GUIDE.md`](guide/GUIDE.md), [`code/README.md`](code/README.md)

## Reproduce the submitted score (no retraining)

```bash
cd code
python -m venv .venv && source .venv/bin/activate    # Python 3.12
python -m pip install -r requirements.txt            # numpy==2.5.3 only
python check_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python predict.py --method submission.py --checkpoint runs/model_bundle.npz \
  --split test --output runs/test-predictions.json
python score_predictions.py --split test \
  --predictions runs/test-predictions.json --output runs/test-score.json
# -> accuracy 81.79871520342613
```

## Retrain from scratch (bit-for-bit deterministic)

```bash
cd code && source .venv/bin/activate
python train_final.py    # ~23 s CPU; rewrites runs/model_bundle.npz
```

## Asset hash

```
shasum -a 256 code/runs/model_bundle.npz
# d1cdc3a3c209980acc43a70c9d428be105f2c2782a608dbbf6833b5c227f08da
```

See `code/runs/train_metrics.json` for hyperparameters and training metrics.
