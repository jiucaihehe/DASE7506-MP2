# New-machine continuation: contextual reliability and executable program transfer

Historical record of the first continuation, before the owner authorized a new test run. For the later frozen **83.94%** test result and current reproduction instructions, see `OPTIMIZATION_ROUND2.md` and `../report/REPORT_V2.md`.

Date: 27 October 2026 (UTC+8). These are **development results**, not a replacement submission or a new test score. The frozen `submission.py` and `runs/model_bundle.npz` were not edited. No test evaluation was run; no test reference file was opened by the new experiments. The original report and experiment log are preserved.

## Outcome

| Selected method | Train OOF hard | Validation hard | Validation all | Paired validation change vs frozen |
|---|---:|---:|---:|---:|
| Frozen original | — | 367/419, 87.59% | 746/798, 93.48% | — |
| Shrinkage quadratic residual model + refit linear model | 2102/2513, 83.65% | 364/419, 86.87% | 743/798, 93.11% | 3 rescued, 6 regressed |
| **Question-conditioned hand-feature weights** | **2096/2513, 83.41%** | **368/419, 87.83%** | **747/798, 93.61%** | **2 rescued, 1 regressed** |
| Executable arithmetic-program transfer | 2084/2513, 82.93% | 367/419, 87.59% | 746/798, 93.48% | disabled by train-only selection |

The original-Adam baseline's OOF result is 2084/2513 (82.93%). The quadratic experiment uses a more converged L-BFGS baseline, whose OOF accuracy is 2088/2513 (83.09%) and validation hard accuracy is 365/419. These are distinct optimization controls and should not be mixed when interpreting improvements.

The best new configuration gains **one** hard validation question, or **0.239 percentage points**. This falls within the original project's stated ±0.5-point noise region. It is a small positive result, **not evidence of a reliable breakthrough**. Its training OOF predictions rescue 21 questions and regress 9 versus the matched baseline, but configuration selection used these same folds. The unadjusted paired binomial p-value (~0.043) is exploratory and is not a valid confirmatory significance claim after searching nine configurations. On validation the corresponding two-sided exact p-value is 1.0 (2 versus 1).

Keep the original submission frozen. Any decision to replace it and any new test evaluation require the owner's approval. No claim is made that the candidate improves the existing 81.80% test score.

## Protocol

All fits, normalization statistics, PCA bases, vocabulary/IDF and template extraction use the supplied **training split only**. Three folds split whole questions, stratifying on the number of correct candidates, with seed 7506. The same split generator is used across experiments. All-correct groups remain in the training objective and have zero group-softmax loss.

Each script writes its predeclared grid to `runs/<experiment>/protocol.json`, then saves `cv_selection.json` **before** loading validation. Selection maximizes pooled OOF hard accuracy. Ties choose the earlier contextual configuration; program-transfer ties choose the larger, more conservative similarity threshold. Validation does not fit parameters or select numerical settings. For the first experiment only, three explicitly declared representation ablations are also reported on validation; they are not used to replace the CV selection. Experiments were performed sequentially, so the three hypotheses are exploratory development, not three independent held-out confirmations.

Training-only OOF predictions use models trained on 2/3 of the training bank. Their absolute accuracy need not equal the full-training model's validation accuracy. IDs serve alignment only; new methods do not use IDs as predictive features.

## 1. Residual geometry and quadratic discrimination

Let E_j be one candidate embedding and c the mean of the four embeddings. Residuals R_j = E_j - c remove shared problem content. We tested three representations: shrinkage-normalized residuals, unit-length residual directions, and concatenation of raw features with normalized residuals. These did not improve the baseline. Validation hard scores were 353, 361 and 354 out of 419 respectively.

The structurally different alternative is a Gaussian log-likelihood ratio over residuals. Fit a train-only residual PCA/whitening transform, then separate correct/incorrect Gaussian covariances on mixed training groups. Shrink each covariance toward identity. The log-likelihood difference is quadratic in the residual, so it can use covariance patterns unavailable to a global linear score. Blend it with a refit group-softmax score using training-score RMS scales.

The finite CV grid used 64/128/256 dimensions, covariance shrinkage 0.5/0.9, and fusion weights 0.15/0.35. CV selected 256, 0.5, 0.35. The resulting validation score was worse, so this approach is rejected. `experiment_context.py` contains training; `method_context.py` is NumPy-only inference.

## 2. Question-conditioned preferences: the positive candidate

The old model assigns a globally fixed weight to features such as answer agreement, integer status and answer magnitude. The new hypothesis is that the usefulness of a feature changes with the kind of question.

Compute the four-candidate embedding centroid c. Center it using the training mean and project it through a training-fitted, whitened PCA basis to get a context vector u. For each candidate, form outer products between u and its existing 46 hand features h_j. The score is:

`score_j = w_base · standardized(original_1070_features_j) + w_context · standardized(vec(u ⊗ h_j))`.

Equivalently, the hand-feature coefficients depend linearly on the shared context. Context is a proxy derived from the supplied trace embeddings; it is **not** an independently encoded question representation. The model does not prove that any principal component corresponds to a named mathematical topic.

We also tested outer products between centroid PCA and residual PCA, rather than hand features. The predeclared grid comprised these two interaction families, context dimensions 8/16 and contextual L2 penalties 0.2/1.0, plus the baseline. Training retains the original zero-initialized full-batch Adam setup: 1500 steps, learning rate 0.02, base L2 0.02. The gradient is computed with matrix multiplication to avoid allocating a large broadcast tensor every step.

CV selected **8 context dimensions × 46 hand features = 368 interactions**, with contextual L2 = 1.0. This is much stronger regularization than the base block. The final score has 1438 feature coefficients. It is trained on train only and frozen before validation. The model rescues 2 validation errors while introducing 1 new error. The context weight norm is only ~0.055, versus ~1.51 for the base coefficients, consistent with a small correction rather than a wholesale model change.

Files: `experiment_bilinear.py`, `method_bilinear.py`, and `runs/bilinear_experiment/selected_bundle.npz`. Inference needs NumPy and the existing `features.py`/`common.py`; training does not run at inference.

## 3. Transfer executable programs from training reference reasoning

Earlier methods used training reference answers mainly to derive correctness labels. This experiment instead uses the reference reasoning's `<<expression=result>>` annotations to build arithmetic programs.

Replace literals corresponding to question quantities with occurrence-indexed variables. Substitute earlier computed expressions into later steps. Only basic arithmetic and a small generic constant set are permitted. Require a program to use at least one input variable, reproduce its source training answer, stay within a size limit and have matching quantity arity on a new question. No final-answer table, source reference text or question ID is stored for inference.

We extracted 3846 executable templates from 4755 training references. A training-fitted word/bigram TF-IDF index retrieves a number-normalized question template. Its program is executed with the **new question's quantities**, and can promote an existing candidate with the resulting numeric answer. If no supported answer exists or similarity is too low, use the original ranker. This differs from the old candidate-embedding kNN: the retrieved object is a reusable computation, not a correctness label.

Three-fold training OOF results at similarity thresholds 0.35/0.45/0.55 were 2079/2081/2083 hard correct, below 2084 for the baseline. Higher thresholds tied the baseline. The predeclared tie rule selected 1.01, which disables transfer. Validation consequently remains exactly the baseline. This is a negative result: sparse text similarity did not identify sufficiently transferable arithmetic structure. It does not establish that reference-reasoning supervision in general is useless.

`program_transfer.py` contains extraction, retrieval and frozen prediction. Four semantic unit tests cover numerical substitution, rejection of constant-only answers, arity/division errors and asset round-tripping. The alternative-number test confirms that the program recomputes an answer instead of returning the source answer.

## Resources and verification

Local machine: Windows, Intel Core Ultra 9 275HX, Python 3.12, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1. Four-thread environment variables and training threadpool limits were applied. Approximate training/development script wall times: residual/QDA 58 s, bilinear search 327 s, program transfer 3.4 s. Costs include their respective folds, full fits and validation analysis.

| Method | Uncompressed inference asset files | Fresh validation prediction + separate scoring |
|---|---:|---:|
| Residual QDA | 3,178,010 bytes (3.031 MiB) | 1.12 s |
| Contextual hand weights | 306,806 bytes (0.293 MiB) | 0.95 s |
| Program transfer including baseline | 2,290,966 bytes (2.185 MiB) | 2.47 s |

Assets were saved with uncompressed `np.savez`. Times include both fresh process startups, imports, input/asset loading, preprocessing and the independent scorer, on **798 validation questions**. These are local validation measurements, **not official test-bank timings or reference-Xeon certification**. All assets are below 16 MiB. No external inference model/API or additional pretrained weights are used. Each method passed prediction checks for reordered subsets, repeated questions, unseen IDs and an empty bank without retraining.

The exact local timings, official validation scoring JSONs, asset hashes and integrity checks are in `runs/new_experiment_verification/verification.json`. The best candidate's checkpoint SHA-256 is `cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537`.

## Reproduce on Windows PowerShell

Use Python 3.12. A clone should preserve original data bytes; Windows automatic CRLF conversion breaks official data hashes. For a fresh clone, use:

```powershell
git -c core.autocrlf=false clone https://github.com/jiucaihehe/DASE7506-MP2.git
cd DASE7506-MP2
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r code/requirements-new-experiments.txt
cd code
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
..\.venv\Scripts\python.exe experiment_context.py
..\.venv\Scripts\python.exe experiment_bilinear.py
..\.venv\Scripts\python.exe experiment_programs.py
..\.venv\Scripts\python.exe -m unittest test_program_transfer -v
..\.venv\Scripts\python.exe verify_new_experiments.py
```

These new files and assets currently live in the local checkout; the public remote will not contain them until they are explicitly committed and pushed. On this machine the local `.venv` is already installed, so the experiment and inference commands can be run directly. The third experiment consumes matching baseline OOF scores saved by the second; run them in order.

To evaluate the positive candidate from its saved checkpoint **without retraining**:

```powershell
..\.venv\Scripts\python.exe predict.py --method method_bilinear.py --checkpoint runs/bilinear_experiment/selected_bundle.npz --split validation --output runs/bilinear_validation_predictions.json
..\.venv\Scripts\python.exe score_predictions.py --split validation --predictions runs/bilinear_validation_predictions.json --output runs/bilinear_validation_score.json
```

Every command above explicitly uses validation or train. Do not substitute the test split without the owner's approval. Do not run `train_final.py`: its existing default output overwrites the original frozen bundle.

## Disclosure and handoff

OpenAI Codex proposed and implemented these three experimental hypotheses, ran train/validation experiments, authored the semantic tests and drafted this English supplement. The student should review the implementation and interpretations before incorporating them into the final report. This file is a development supplement; it has not been merged into or rendered as the final report, whose overall limit remains ten pages. No new leaderboard issue, remote commit or public submission was created.
