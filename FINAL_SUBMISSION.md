# Instructor guide: the final 83.94% submission

**The final method is the contextual ranker. Expected official full-test result: 392/467, accuracy 83.94004282655246%.**

## One-command evaluation after installation

Use Python 3.12. From the repository root:

```text
python -m pip install -r code/requirements-inference.txt
python code/reproduce_final.py --split test
```

A virtual environment is recommended, as described in `README.md`. The evaluation helper sets four-thread limits and invokes two fresh processes: the official prediction runner, then the unchanged independent scorer. It verifies the exact final checkpoint hash before starting. It never searches for another checkpoint or falls back to the historical method. It prints the actual scorer output and writes fresh results under `code/runs/instructor_reproduction/`; no cached predictions are used to produce the score.

Expected scorer fields:

```json
{"correct": 392, "n": 467, "accuracy": 83.94004282655246}
```

## Exactly what is loaded

| Role | Final file |
|---|---|
| Model entrypoint | `code/submission_candidate.py` |
| Selected implementation | `code/method_bilinear.py` |
| Feature extraction | `code/features.py` |
| Complete inference assets | `code/runs/final_candidate/config.json` and `selected_bundle.npz` |
| Final English report | `output/pdf/REPORT_V2.pdf` (4 pages) |
| Frozen official score record | `code/runs/final_evaluation/test_score.json` |
| Train-only reproduction | `code/train_candidate.py` |

The configuration selects only `bilinear`. The final checkpoint SHA-256 is:

```text
cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537
```

The official runner imports the file specified by `--method` and passes the explicit `--checkpoint` path to it. Merely keeping other files in the repository does not cause them to execute or be loaded. The new model uses 368 context/hand-feature interactions in addition to the original feature vector; the old checkpoint does not contain these final learned parameters.

## Direct official commands

From `code/`, with thread limits set to four as in `README.md`:

```text
python predict.py --method submission_candidate.py --checkpoint runs/final_candidate --split test --output runs/reproduced_test_predictions.json
python score_predictions.py --split test --predictions runs/reproduced_test_predictions.json --output runs/reproduced_test_score.json
```

The helper above runs these same official tools with absolute paths. No retraining, model download or external API is needed. Training code is separate; inference needs only NumPy.

## Historical baseline is not the final entrypoint

The files `code/submission.py`, `code/runs/model_bundle.npz`, `report/REPORT.md` and `code/runs/test-score.json` describe the preserved **81.80% baseline** for experimental comparison. They are not loaded by either final evaluation command above. The final report is `REPORT_V2.pdf`, not the historical report. All final paths are explicit to prevent this ambiguity.

The isolated audit copied only `common.py`, `features.py`, `method_bilinear.py`, `submission_candidate.py`, the official prediction/scoring runners and the final assets into a temporary directory. Neither historical `submission.py` nor historical `model_bundle.npz` was present. References were also absent during prediction, and network/training imports were blocked. That directory still reproduced all 467 final choices and scored 392/467. See `code/runs/submission_audit/reproduction_audit.json` and `code/audit_submission.py`.

`code/PUBLICATION_CHECK.md` records an independent GitHub clean-clone reproduction using a new NumPy-only environment and provides immutable version/checkpoint links. Historical models remain for provenance and ablation only. Keeping a baseline is consistent with comparing initial and improved methods; only the explicit final assets are used for this result.

The fixed instructor helper was additionally fetched from published commit `8cc53d5637f947ed648000cb039c5e9c054a86a7` and run in both the clean checkout and a minimal directory with no historical code/checkpoint or saved predictions. Both returned 392/467. Records are in `code/runs/submission_audit/instructor_entrypoint_clean_clone.json` and `instructor_entrypoint_verification.json`.
