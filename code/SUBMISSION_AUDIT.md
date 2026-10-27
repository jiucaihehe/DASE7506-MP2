# Submission compliance and reproduction audit

**Audit date: 28 October 2026 (UTC+8).**

**Technical result: PASS on the local audited checks.** This document records the initial pre-publication audit. Publication and clean-clone follow-up are tracked in `PUBLICATION_CHECK.md`. Creating the final course submission issue remains a separate owner action.

The unchanged frozen contextual model reproduces **392/467 = 83.94004282655246%**. Do not submit validation accuracy or the old model's score as this model's result. This audit did not fit a model, choose a configuration using test, or change inference code/weights.

## Authoritative materials checked

- Live submission form and JavaScript: <https://xudongwu-0.github.io/courses/dase7506/mp2/>
- Current official Guide: <https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md>
- Current official README: <https://github.com/xudongwu-0/DASE7506-MP2/blob/main/code/README.md>
- Official leaderboard snapshot: <https://raw.githubusercontent.com/xudongwu-0/xudongwu-0.github.io/main/courses/dase7506/data/leaderboard.json>

The form accepts official full-test accuracy in percent, student ID and HTTPS code/checkpoint links. Preparing the form does not complete submission: the generated GitHub issue must actually be created. Use the same account/student ID for updates. Final links and score are due **31 October 2026, 23:59 (UTC+8)**. Follow the Guide's deadline even if a later form phase permits collecting missing links.

## Checks and evidence

| Requirement | Audited result | Evidence |
|---|---|---|
| Official bank/parser/scorer unchanged | PASS | `check_data.py` passes all split hashes, shapes, alignment and disjointness. `common.py`, `predict.py`, `score_predictions.py`, `check_data.py` and data manifest match live upstream after source line-ending normalization. |
| Training split only | PASS for inspected final training path | `train_candidate.py` calls only train input/reference loaders. Training imports do not execute development `main()` functions. Final normalization/PCA/coefficients are train-fitted. |
| Validation method selection; test used for scoring/reproduction | PASS in recorded continuation | `candidate_freeze.json` stores the validation comparison and frozen hashes. Test result was first obtained after selection. Later audits reproduce that exact method. Historical activity outside these recorded files cannot be independently proven by a code audit. |
| No references at prediction | PASS | Isolated directory initially contains only candidate inputs/manifest/assets/sources. An audit hook prohibits reference opens. Only after prediction finishes are test references copied in for the separate official scorer. |
| Offline, no external model/API | PASS | Network calls and imports of training/model libraries are blocked in the prediction audit. Final inference needs only NumPy and standard-library modules. |
| Full test score | PASS | Every one of the 467 candidate choices equals the frozen saved prediction, and the unchanged independent scorer returns 392/467. |
| New bank support | PASS | Reordered subsets, repeated questions, new IDs and empty bank accepted without fitting or cached answer lookup. Candidate order remains the contract's fixed order. |
| <=16 MiB inference assets | PASS | NPZ plus required config: **307183 bytes**; NPZ members contain 305916 uncompressed bytes. ZIP includes exactly the same NPZ/config. |
| <=30 s CPU, four threads | PASS locally; reference-machine eligibility pending | Original fresh prediction plus independent scoring: **0.914 s**. Audit observes a single NumPy OpenBLAS pool with **4 threads**. The reference Xeon is the authority; it was not available here. |
| <=10-page report | PASS | `output/pdf/REPORT_V2.pdf` has **4 pages**, visually inspected, with comparisons, ablation, resources, reproduction and AI disclosure. |
| Frozen reproducible assets | PASS locally | Fixed train-only refit previously produced bitwise-identical arrays. Source/asset checksums verified. The current checkpoint digest remains `cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537`. |
| Public code/checkpoint links and identified version | NOT READY | New inference/config/report paths returned HTTP 404 from the public remote. New files have not been committed/pushed. |
| Actual final leaderboard entry | NOT COMPLETE | No final issue was created by this chat. The audit snapshot has no best-row entry under GitHub account `jiucaihehe`. A valid student ID must be supplied by its owner. |

Machine-readable evidence is in `runs/submission_audit/`: `upstream_integrity.json`, `data_check.json`, `reproduction_audit.json`, `reproduced_score.json`, `public_artifact_access.json` and `leaderboard_analysis.json`. The complete original resource record remains in `runs/final_evaluation/test_resources.json`. The isolated audit adds contract checks, so its wall time is not substituted for the original fresh prediction timing.

## Packaging issues found and fixed locally

1. The root README still described only the 81.80% baseline. It now points to the correct entrypoint, complete assets and 83.94% result, while clearly preserving the historical baseline.
2. Added `requirements-inference.txt` with NumPy 2.5.4, so evaluators do not need training libraries for frozen inference.
3. Added `.gitattributes` protection for exact JSONL bytes. Git automatic CRLF conversion had broken official data hashes during migration. No official data content or scorer logic was changed.
4. Source checksum helpers originally compared physical CRLF bytes and could reject an equivalent LF checkout. Added separate LF-normalized digests while retaining original raw digests. Copied inference sources were normalized to LF in the isolated audit, and all predictions remained identical. This changes verification portability, not the model.
5. Produced the checked four-page English PDF and `output/checkpoint/context_candidate.zip`, containing exactly `config.json` and `selected_bundle.npz`. These are local deliverables; creation is not publication.

## Why leaderboard ties do not prove a performance ceiling

The downloaded snapshot was generated at **2026-10-27 17:18:37.403 UTC**, or **28 October 2026, 01:18:37.403 UTC+8**. Applying the site's best-per-account/student grouping gives 32 rows, all labeled **Self-reported** at that snapshot. Displayed ties occur at 84.5824, 84.3683, 84.1541, 83.7259, 83.5118, 83.2976 and 81.5846 percent. The browser formats percentages to four decimal places; displayed equality can also hide tiny stored numeric differences. These are observations about that snapshot, not an assertion of the board's future state or verified ranks.

There are only 467 binary scored questions, so official accuracy changes in increments of **100/467 = 0.2141327623 percentage points**. Examples:

- 390 correct -> 83.5117773019272% -> displayed 83.5118%.
- 391 correct -> 83.72591006423983% -> displayed 83.7259%.
- 392 correct -> 83.94004282655246% -> displayed 83.9400%.

Different methods can have the same number correct while making mistakes on different questions, or choose different candidates with equivalent final numeric answers. The scalar score discards those differences. Shared embeddings, data and resource constraints can plausibly encourage similar methods, but establishing that requires released implementations and prediction/error comparisons. A leaderboard tie alone establishes neither methodological equivalence nor an information-theoretic ceiling. The earlier report's strong claim that the plateau was "information-limited, not method-limited" should be read as a hypothesis, not a demonstrated bound; the updated report avoids that conclusion.

## Pre-publication findings and final owner actions

1. Commit and push the selected code, full checkpoint/config, official JSON, experiment records and final report. Identify an immutable commit/version. At audit time remote HEAD was `ac9a045bdfd032d5c44dc615f7d45bb970921007`, whose README still advertised the old model.
2. Verify the public code link and complete-checkpoint link from an unauthenticated context. The checkpoint link must deliver both required files, or the complete checked ZIP.
3. The student reviews the implementation/report/disclosure and supplies the correct student ID.
4. Enter official `accuracy` **83.94004282655246**, both verified links, and the student ID in the course form; confirm and create the resulting GitHub issue.
5. Check that the issue and eventual leaderboard update correspond to the intended code version, checkpoint and score. Do not use any later test reproduction to alter the method.

No push, public issue or leaderboard mutation was performed during this initial audit. The owner subsequently authorized repository publication; see `PUBLICATION_CHECK.md` for the resulting commit, verified public links and fresh-clone reproduction. Earlier NOT READY rows above describe the initial inspection, not the follow-up publication status.
