# Experiment log (development scripts)

This file describes the original development round. Later rounds are recorded in
`OPTIMIZATION_ROUND2.md` and `OPTIMIZATION_ROUND3.md`. The current submitted
checkpoint remains the 83.94% contextual model; round three did not replace it.

These scripts document the development process. They train only on the train
split and select on validation; test references were touched exactly once, by
the frozen final pipeline (`predict.py --split test` + `score_predictions.py`).

| Script | Purpose |
|---|---|
| `explore.py` | Data statistics + baselines (first candidate, majority vote, random). |
| `train.py` | Round 1: pointwise logistic regression, hand vs hand+embedding features, argmax vs cluster selection. |
| `train2.py` | Round 2: hard-group-focused evaluation; GBM/MLP comparisons; ensemble and agreement-bonus probes. |
| `train3.py` | Round 3: group-normalized features, hard-group sample weighting, hard-only fitting. |
| `train4.py` | Group-softmax objective (mini-batch Adam): motivation + first results. |
| `train5.py` | Seed-variance study for mini-batch training; weight/probability ensembling. |
| `train6.py` | Deterministic full-batch group-softmax; hyperparameter plateau (lam, lr, epochs). |
| `train7.py` | Consistency features, PCA-reduced embeddings, tiny MLP with group-softmax loss. |
| `train8.py` (+ `textvec.py`) | TF-IDF unigram+bigram augmentation: overfits (train-hard 99–100%, val-hard drops 3.5–5.5 pts). |
| `train9.py` | Independent-agreement features, block-wise L2, single-correct-group upweighting: all within noise or worse; rejected. |
| `train10.py` (+ `arith.py`) | Arithmetic verification of "expr = value" claims in traces: weak signal (73.7% vs 75.8% base), no val-hard gain; rejected. |
| `train11.py` | Ensemble of group-softmax and pointwise-LR probabilities: no gain over pure group-softmax; rejected. |
| `train12.py` (+ `chain.py`) | Reasoning-chain coherence ("magic number") features: strong univariate signal (80.4% vs 67.8%) but no model gain; rejected. |
| `train13.py` | Pairwise GBM Bradley–Terry preference model: val-hard 80.67, overfits; rejected. |
| `train14.py` | Cluster-level two-stage answer model: val-hard 73.27, aggregation loses candidate detail; rejected. |
| `train15.py` | kNN retrieval scoring (PCA-64, cosine, k=10–200): ≤68.74 alone, no ensemble gain; rejected. |
| `train16.py` | Smooth margin listwise loss ± 4× permutation augmentation: ≤87.11; rejected. |
| `train_final.py` | The submitted model: deterministic full-batch group-softmax, writes `runs/model_bundle.npz`. |
| `analyze.py` | Calibration (confidence/margin vs accuracy) and failure analysis on validation. |

Requirements for experiments: `numpy==2.5.3`, `scikit-learn==1.9.1` (rounds 1–3, 7),
`scipy==1.18.1` (analysis only). The **final** training and inference pipeline needs **numpy only**.
