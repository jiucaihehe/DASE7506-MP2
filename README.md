# DASE7506 MP2 - Selecting Reliable LLM Reasoning

**Frozen contextual model: official test accuracy 83.94004282655246% (392/467).**

**Course submission completed: student ID 3036707472, [issue #276](https://github.com/xudongwu-0/xudongwu-0.github.io/issues/276). Score and encrypted artifact links were received. See [submission receipt](code/COURSE_SUBMISSION.md).**

**For instructors: start with [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md).** After installing `code/requirements-inference.txt`, run `python code/reproduce_final.py --split test` from the repository root. This fixed entrypoint verifies and evaluates only the final contextual checkpoint; it does not load the historical baseline.

Validation accuracy is 93.6090% (747/798); mixed-correctness validation accuracy is 87.8282% (368/419). The model was selected using training/validation only, frozen, and then evaluated on test. A separate isolated reproduction with no reference files available during prediction produced all 467 identical choices. Test references are for scoring/reproduction only, never tuning.

The method augments the original linear group-softmax ranker with strongly regularized interactions between shared trace context and candidate reliability features. Training uses only the supplied train bank. Inference is offline, CPU-only and NumPy-only. All inference assets total 307,183 bytes (~0.293 MiB). The recorded complete test prediction plus separate scoring took 0.914 s on an Intel Core Ultra 9 275HX, four threads; reference-machine eligibility remains subject to instructor verification.

## Current artifacts

- Report: [English PDF](output/pdf/REPORT_V2.pdf), [editable Markdown](report/REPORT_V2.md).
- Inference: [submission_candidate.py](code/submission_candidate.py), [method_bilinear.py](code/method_bilinear.py), [features.py](code/features.py), and the unchanged official [common.py](code/common.py).
- Complete assets: [runs/final_candidate/](code/runs/final_candidate/) or [checkpoint ZIP](output/checkpoint/context_candidate.zip). Both the NPZ and config are required.
- Official score: [test_score.json](code/runs/final_evaluation/test_score.json).
- Resources: [test_resources.json](code/runs/final_evaluation/test_resources.json).
- Fixed train-only reproduction: [train_candidate.py](code/train_candidate.py).
- Development: [NEW_EXPERIMENTS.md](code/NEW_EXPERIMENTS.md), [OPTIMIZATION_ROUND2.md](code/OPTIMIZATION_ROUND2.md).
- Audit: [SUBMISSION_AUDIT.md](code/SUBMISSION_AUDIT.md), [isolated reproduction](code/runs/submission_audit/reproduction_audit.json).
- Official materials: [Guide](guide/GUIDE.md), [code/data README](code/README.md).

Original `code/submission.py`, `code/runs/model_bundle.npz`, `report/REPORT.md` and original test JSONs are the **81.80% historical baseline**. Their commands do not reproduce the new score. Do not overwrite them when following the current instructions.

## Reproduce without retraining

Python 3.12. Preserve original data bytes on Windows:

```bash
git -c core.autocrlf=false clone https://github.com/jiucaihehe/DASE7506-MP2.git
cd DASE7506-MP2/code
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-inference.txt
python check_data.py
```

Set thread limits before prediction/scoring. In PowerShell:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:BLIS_NUM_THREADS='4'
python predict.py --method submission_candidate.py --checkpoint runs/final_candidate --split test --output runs/reproduced_test_predictions.json
python score_predictions.py --split test --predictions runs/reproduced_test_predictions.json --output runs/reproduced_test_score.json
```

In a POSIX shell, use `export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 BLIS_NUM_THREADS=4`, then run the same Python commands. Expected official `accuracy`: **83.94004282655246**. This is the leaderboard value. These commands reproduce a frozen result; do not use test results for method selection.

## Train the fixed model

```bash
python -m pip install -r requirements-new-experiments.txt
python train_candidate.py --output runs/retrained_context_candidate.npz
```

Training reads only train inputs/references and refuses to overwrite an existing output. On the development machine the regenerated arrays were bitwise identical to the frozen checkpoint. Floating-point decomposition details may vary across platforms; use the supplied checkpoint for exact-score reproduction. `experiment_bilinear.py` reproduces train-only CV. `train_final.py` belongs to the historical baseline and writes the old checkpoint path.

Checkpoint SHA-256:

```text
cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537
```

## Submission status

The submitted source and checkpoint URLs are pinned to `590532406a947c8c9d7eaf775887a8a069ec64d4`, including all later experiment disclosures. The fixed model and instructor helper are unchanged from the earlier independently verified version `8cc53d5637f947ed648000cb039c5e9c054a86a7`. The helper reproduced 392/467 in a clean NumPy-only checkout and with historical models absent. [COURSE_SUBMISSION.md](code/COURSE_SUBMISSION.md) records the actual completed issue, exact submitted URLs and receipt checks; [PUBLICATION_CHECK.md](code/PUBLICATION_CHECK.md) preserves the earlier artifact audit.

Use the same GitHub account/student ID for updates and provide both links by **31 October 2026, 23:59 (UTC+8)**. AI assistance and failed experiments are disclosed in the English report. Supplied data licenses and attribution remain intact.
