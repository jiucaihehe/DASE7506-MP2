# Authorized final evaluation: 83.94%

27 October 2026 (UTC+8). The owner authorized further optimization and a new final test evaluation. Test references were used only by the official independent scorer after method selection and freezing. No model or configuration was changed based on this new test result.

## Result

| Method | Validation hard | Validation all | Official test |
|---|---:|---:|---:|
| Original frozen linear group-softmax | 367/419 (87.59%) | 746/798 (93.48%) | 382/467 (81.80%) |
| Frozen contextual hand-feature model | **368/419 (87.83%)** | **747/798 (93.61%)** | **392/467 (83.94004282655246%)** |

The new test score is **83.94%**, an increase of 10 correct selections and 2.14 percentage points over the recorded original score. The new method was selected for its validation result before testing; it was not selected by comparing test scores from multiple new models. The result remains one fixed-bank observation, not proof of universal improvement.

## Additional experiments before testing

Each of the first three families used train-only three-fold group CV (seed 7506) to select one configuration. Only that selected configuration was then evaluated on validation. The bootstrap ensemble used all eight predeclared members without seed selection.

| New experiment | Selected configuration | Validation hard |
|---|---|---:|
| Alternative objectives | Uniform probability targets on correct candidates; L2 0.005 | 364/419 |
| Conditional RBF basis | Hand/context latent inputs; width 2; extra L2 0.2 | 366/419 |
| Structural mathematical ngrams | Quantity-role normalization, relation/unit vocabulary; uniform loss; extra L2 0.5 | 364/419 |
| Bayesian-bootstrap ensemble | Eight Exponential(1) group-weight fits; equal predictive-probability average | 365/419 |

A development diagnostic also mixed the uniform-target model with the original or contextual model at weights 0.1/0.25/0.5/0.75; none exceeded the contextual model's validation accuracy. No such mixture was tested on test.

The contextual model from the first continuation therefore remained the validation-selected winner. `runs/candidate_freeze.json` records the compared configurations, validation counts, source hashes and frozen checkpoint before any new test prediction. The final checkpoint is copied to `runs/final_candidate/selected_bundle.npz`; its SHA-256 is:

`cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537`

## Method and resources

Use the original 1070 candidate features plus 368 interactions between eight principal components of the four-candidate embedding centroid and the 46 handcrafted/group-normalized features. These interactions allow the weights on reliability signals to depend on the shared question/trace context. Fit only on training, using the original zero-initialized, 1500-step full-batch Adam group-softmax objective. Base L2 is 0.02; interaction L2 is 1.0. Inference uses fixed train-fitted transforms, normalization and weights; it requires no fitting, reference files, external models, generation or API.

Local test measurement: Windows, Intel Core Ultra 9 275HX, Python 3.12, four threads. Fresh prediction process including imports/loading/preprocessing: 0.610 s. Separate scoring process: 0.304 s. Combined: **0.914 s**. Inference assets, including config: **307,183 bytes (~0.293 MiB)**, uncompressed. These satisfy the local resource checks; the reference Xeon remains the eligibility authority. Saved input-contract tests confirm the model can process reordered subsets, repeated questions, new IDs and empty banks without retraining.

Original `submission.py`, `runs/model_bundle.npz`, original test predictions and original test score remain unchanged. The new evaluation is stored separately:

- `runs/final_evaluation/test_score.json`: official scorer output.
- `runs/final_evaluation/test_predictions.json`: frozen method's choices.
- `runs/final_evaluation/test_resources.json`: complete-process wall times, asset hash and size.
- `runs/final_candidate/`: all inference assets for the new entrypoint.

## Reproduce from the frozen checkpoint

From `code/`, after installing `requirements-new-experiments.txt` in Python 3.12:

```powershell
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
..\.venv\Scripts\python.exe predict.py --method submission_candidate.py --checkpoint runs/final_candidate --split test --output runs/reproduced_candidate_test_predictions.json
..\.venv\Scripts\python.exe score_predictions.py --split test --predictions runs/reproduced_candidate_test_predictions.json --output runs/reproduced_candidate_test_score.json
```

Expected accuracy: `83.94004282655246` (392/467). These are reproduction commands, not permission to use test results for additional tuning. To regenerate the fixed model from training only, without repeating the development search:

```powershell
..\.venv\Scripts\python.exe train_candidate.py --output runs/retrained_context_candidate.npz
```

The fixed training command was independently rerun into `runs/reproduction_check/`; every saved checkpoint array was bitwise identical to the frozen checkpoint. This was a reproduction check, not additional parameter search, and no second test evaluation was performed.

For a fresh Windows clone, disable automatic data line-ending conversion (`git -c core.autocrlf=false clone ...`). Data restored during migration was copied byte-for-byte from Git; official data/parser/scorer content was not changed.

## Leaderboard text and status when this experiment concluded

Use **83.94004282655246** for the accuracy field, or **83.94%** when rounding for display. Suggested English method description:

> Context-conditioned group-softmax ranker using fixed embeddings, handcrafted reliability features, and strongly regularized interactions between shared trace context and candidate features. Trained on the supplied training split only; all inference is offline on CPU.

No new GitHub issue or leaderboard submission was created by this experiment. At its conclusion the new files were local. Subsequent repository publication and verified artifact links are recorded in `PUBLICATION_CHECK.md`. Student ID and the actual course submission issue remain the owner's responsibility.

## Disclosure

OpenAI Codex proposed, implemented and evaluated these additional experiments, authored the new inference/training scripts and drafted this English record and the updated report. The student remains responsible for reviewing and understanding the implementation and claims. No external data or pretrained weights were introduced. The earlier development report is a historical record; `report/REPORT_V2.md` describes the new frozen result.
