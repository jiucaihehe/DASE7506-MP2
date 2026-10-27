# Public repository and clean-clone verification

**28 October 2026 (UTC+8).** The owner authorized publishing the contextual model, its complete inference assets, source code, experiment records and English report to `jiucaihehe/DASE7506-MP2`.

**Status: repository publication and clean-clone reproduction PASSED.** Publishing the repository is not itself a completed course submission.

## Immutable published version and submission fields

Verified source/artifact commit: **`8a3e89043888bdd0a57e8572c63fba7cee463a94`**.

- Code link: <https://github.com/jiucaihehe/DASE7506-MP2/tree/8a3e89043888bdd0a57e8572c63fba7cee463a94>
- Complete checkpoint ZIP: <https://raw.githubusercontent.com/jiucaihehe/DASE7506-MP2/8a3e89043888bdd0a57e8572c63fba7cee463a94/output/checkpoint/context_candidate.zip>
- Alternative complete-checkpoint folder: <https://github.com/jiucaihehe/DASE7506-MP2/tree/8a3e89043888bdd0a57e8572c63fba7cee463a94/code/runs/final_candidate>
- Official score JSON: <https://github.com/jiucaihehe/DASE7506-MP2/blob/8a3e89043888bdd0a57e8572c63fba7cee463a94/code/runs/final_evaluation/test_score.json>
- English report PDF: <https://github.com/jiucaihehe/DASE7506-MP2/blob/8a3e89043888bdd0a57e8572c63fba7cee463a94/output/pdf/REPORT_V2.pdf>

Enter **83.94004282655246** in the course accuracy field. Use the student's correct ID and the same GitHub account for updates. The ZIP contains exactly `config.json` and `selected_bundle.npz`; extract both together. Source and asset links above are pinned to the same verified commit. Later documentation-only updates do not change that source/artifact version.

## Clean-clone result

The version was cloned from GitHub into a separate checkout rather than copied from the development directory. A new Python 3.12 virtual environment installed only `numpy==2.5.4`. SciPy and scikit-learn were absent. The commands from the repository documentation were run against the published assets, without retraining.

| Check | Result |
|---|---|
| All official bank hashes, alignment, shapes and split disjointness | PASS |
| Frozen source/checkpoint digests | PASS |
| Full-test independent official score | **392/467, 83.94004282655246%** |
| Match to original frozen predictions | **All 467 choices identical** |
| Fresh complete prediction process | 0.5234 s |
| Separate complete scoring process | 0.2167 s |
| Combined wall time | **0.7401 s**, four threads |
| Inference assets, including configuration | **307183 bytes (~0.293 MiB)** |
| Unauthenticated raw GitHub downloads | **9/9 files passed**, byte-identical to clean checkout |
| Complete ZIP payload | Both required files match the published inference folder |
| README local artifact links | All resolve in the published tree |
| Clean checkout after verification | No tracked/untracked project changes |
| Report | English, **4 visually inspected pages**, below 10-page limit |

Machine-readable evidence: `runs/submission_audit/clean_clone_verification.json`, `publication_access.json`, `clean_clone_score.json` and `clean_clone_predictions.json`. `verify_publication.py` documents the check. Existing isolated reference-free/network-blocked inference evidence remains in `SUBMISSION_AUDIT.md` and its linked records. This verification reproduced a fixed method; it did not use test results to select or tune a model.

The measured machine is the development Windows Intel Core Ultra 9 275HX. Only the instructor can certify the Xeon reference-machine runtime or results on an unreleased audit bank. Local asset/time checks pass by a wide margin.

The frozen target result is **83.94004282655246% (392/467)**. The checkpoint SHA-256 remains `cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537`. No model fitting, tuning or candidate selection is performed in this publication check.

## Course requirements and remaining submission actions

The published repository contains training code, fixed inference code/assets, official score JSON, experiment/resource records, seeds/settings, hashes, English report, reproduction commands and AI/reuse disclosure. The official data, parser and scorer are unchanged. The original baseline is explicitly distinguished from the current model. The inspected training path uses train only; the recorded selection uses validation; test is used for frozen final scoring and reproduction.

The student should review and understand the implementation and report. The final form still needs the correct student ID and an actual GitHub issue created from the prepared form by **31 October 2026, 23:59 (UTC+8)**, with both verified links. No leaderboard issue was created during repository publication. Thus **repository artifact readiness passes**, while **actual course submission remains pending**.
