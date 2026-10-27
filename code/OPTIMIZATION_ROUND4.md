# Five new families: validation-only development

**Follow-up:** after the initial development phase below, the owner explicitly
requested test evaluation of the two 369/419 candidates. Both frozen results are
reported in the final section. The initial validation-only records are preserved.

28 October 2026. Two new families reach **369/419 (88.07%)** on mixed-correctness
validation groups, compared with **368/419 (87.83%)** for the frozen submission.
This is one additional correct selection, not a demonstrated test improvement.
**The submitted checkpoint and its verified test score of 83.94% are unchanged.**

## Scope and novelty

The user requested five methods not previously tested. We checked the original
experiment log and all three continuation rounds. All five retain the contextual
model as a common foundation so that their additions can be compared. They are:

1. **Piecewise-linear additive functions.** Add positive-part hinge functions of
   each of the 46 hand/group-normalized features at its training 20th, 40th, 60th,
   and 80th percentiles. This learns threshold/saturation effects as additive
   functions, rather than trees, an MLP, a radial basis model, or quadratic
   embedding features. There are 184 new coordinates.
2. **Context-cluster distributionally robust optimization.** K-means assigns
   training question centroids to eight clusters in whitened PCA context space.
   Minimize a temperature-smoothed maximum of the eight mean group losses instead
   of the overall mean. Cluster weights change with current losses. This differs
   from earlier fixed weighting by number of correct candidates and Bayesian
   bootstrap weights. The clusters are context proxies, not verified topic labels.
   No cluster assignment or additional fitting is needed at inference.
3. **Class-conditional Markov sequence likelihood.** Hash number-normalized tokens
   into 127 states plus a boundary state; fit separate 128-by-128 transition
   distributions for correct and incorrect training traces with Laplace smoothing.
   Eight likelihood summaries cover total, first/second-half, variability and
   tail behavior. This is generative conditional sequence modeling, not the
   previously tested discriminative TF-IDF or structural ngram vectors.
4. **Reference-operation plan supervision.** Train ridge regression from hashed
   question unigram/bigram counts to five log-count targets from training reference
   reasoning: written addition, subtraction, multiplication, division and equality
   occurrences. Compare predicted counts with candidate counts via signed/absolute
   deviations and products (15 features). This uses reference reasoning as an
   auxiliary supervised target; it neither retrieves a reference nor executes a
   transferred program, verifies equations, or builds an answer table. Written
   operator counts are coarse proxies for plans, not an exact semantic parser.
5. **Nuisance-suppressed embedding projection.** Let W be within-question residual
   covariance and B be question-centroid covariance. Solve the generalized
   eigenproblem for W against B plus 0.1 times its average diagonal as ridge.
   Add 64 residual coordinates maximizing within/between variance ratio. Unlike
   ordinary PCA, this explicitly penalizes directions dominated by shared context;
   unlike the earlier Gaussian model it does not fit correctness-class covariances.

## Selection protocol

All parameter fitting uses the supplied train split only. Three-fold group CV is
stratified by number of correct candidates, seed 7506. Each fold refits PCA,
normalization, quantiles, clusters, auxiliary models and projection as applicable.
The Markov and reference-plan features used to train a ranker are themselves
two-fold cross-fitted (seed 7510) inside that training fold; its held-out predictions
use an auxiliary model fitted on the whole training fold. Final training follows
the same cross-fitting procedure and saves a full-training auxiliary model for
inference. No validation labels train these auxiliary models.

Each family has two predeclared settings. Additional-feature L2 is 0.2 or 2.0;
robust-loss temperature is 0.1 or 0.5. Base L2 remains 0.02 and contextual-block
L2 remains 1.0. Training uses zero-initialized L-BFGS, maximum 350 iterations,
ftol 1e-11 and gtol 1e-7. The matched contextual control uses this same optimizer;
the submitted model retains its original Adam fit.

Select one setting per family by pooled training OOF hard correct count, with the
earlier setting breaking ties. Save selection and complete all full-training fits
before opening validation. A Python audit hook refuses reads of either test-bank
file. No test scoring, test-based selection, mixture-weight search or subsequent
parameter adjustment was performed. The entire search/refit/validation run took
74.16 seconds locally; all 33 CV fits and six full fits reported convergence.

## Results

| Method | Selected setting | Train OOF hard /2513 | Validation hard /419 | Validation all /798 | Rescued / regressed vs frozen |
|---|---:|---:|---:|---:|---:|
| Frozen contextual model, prior record | — | 2096 | 368 (87.83%) | 747 | — |
| Matched L-BFGS contextual control | — | 2087 | 368 (87.83%) | 747 | 0 / 0 |
| Piecewise-linear additive model | L2 2.0 | 2082 | 367 (87.59%) | 746 | 0 / 1 |
| Context-cluster robust training | temperature 0.1 | 2085 | **369 (88.07%)** | **748** | 3 / 2 |
| Markov likelihood features | L2 2.0 | 2085 | 368 (87.83%) | 747 | 0 / 0 |
| Reference-operation plan supervision | L2 0.2 | 2086 | **369 (88.07%)** | **748** | 1 / 0 |
| Nuisance-suppressed projection | L2 0.2 | 2089 | 367 (87.59%) | 746 | 1 / 2 |

The reference-plan model fixes one validation error without creating a new one;
the robust model fixes three and creates two. Their advantage is only 0.239
percentage points on the 419 hard groups. Neither beats the matched control in
training OOF accuracy. The validation bank has been reused across many development
rounds, so these results do not establish a statistically reliable gain. We keep
both as development candidates and do not replace the final submission or claim
a new test score. The result also does not establish that the old score is a ceiling.

## Independent reproduction and constraints

`method_five_new.py` performs frozen inference using NumPy only. All five methods
were run through separate fresh official `predict.py` and `score_predictions.py`
processes on the complete **validation** bank. Every choice matched the development
diagnostics. Each also passed empty-bank, unseen-ID, reordered and repeated-question
checks without retraining. A separate NumPy-only Python environment was used for
the recorded reproduction. Local hardware: Windows, Intel Core Ultra 9 275HX,
Python 3.12, four numerical threads.

| Method | Uncompressed checkpoint bytes |
|---|---:|
| Piecewise-linear additive model | 181,856 |
| Context-cluster robust training | 175,734 |
| Markov likelihood features | 438,318 |
| Reference-operation plan supervision | 196,884 |
| Nuisance-suppressed projection | 701,850 |

Each asset includes all its coefficients and learned transforms/statistics, and
is below 16 MiB. Exact fresh-process prediction plus scoring times and checkpoint
hashes are in `runs/five_new_experiment/verification.json`; each local validation
measurement is below 30 seconds. These are validation measurements, not test-bank
measurements or certification on the reference Xeon. Ordinary source code and
provided data are excluded from asset sizes as specified by the guide.

Mean-loss and robust-loss gradients were checked by finite differences; maximum
absolute discrepancy was about 1.03e-10. The operator-count feature and test-file
access guard were also checked (`semantic_checks.json`). Official data, numeric
parser and scoring code were not edited.

## Commands and files

From `code/`, training/search uses the pinned experiment requirements:

```bash
python -m pip install -r requirements-new-experiments.txt
python experiment_five_new.py
```

To reproduce a saved development candidate without training, NumPy alone suffices:

```bash
python -m pip install -r requirements-inference.txt
python predict.py --method method_five_new.py --checkpoint runs/five_new_experiment/reference_plan_bundle.npz --split validation --output runs/reference_plan_validation_predictions.json
python score_predictions.py --split validation --predictions runs/reference_plan_validation_predictions.json --output runs/reference_plan_validation_score.json
```

Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and NUMEXPR_NUM_THREADS
to 4 when running the two commands directly. `python verify_five_new.py` sets these
limits automatically, reproduces all five official validation scores and checks
the input contract. Its additional choice-by-choice assertions use checked-in
development diagnostic NPZ files, after the fresh official prediction and scoring
processes have completed. These diagnostic arrays are not loaded by inference.
Both the verification script and individual official commands work without retraining.

All six fitted checkpoints (five candidates plus control), protocols, CV results,
validation summaries/diagnostics, official score JSONs and resource checks are saved
under `runs/five_new_experiment/`. Local OOF arrays and regenerated prediction JSONs
are ignored by Git. The active teacher entrypoint remains
`reproduce_final.py`, documented in `../FINAL_SUBMISSION.md`.

OpenAI Codex proposed and implemented these five experiments, performed the checks
and drafted this English record. No external dataset, pretrained weight or API was
used. The student remains responsible for understanding the work.

## Authorized frozen test follow-up

On 28 October 2026, after all five development results and checkpoints were
published at commit `ff452f10febe6612e9e6bda6e2241063d44ead43`, the owner explicitly
requested official test scores for the two candidates with 369/419 validation
hard accuracy. Neither method, checkpoint nor hyperparameter was changed.

`evaluate_round4_frozen.py` first verifies both checkpoint hashes against the
pre-existing validation verification record and writes `freeze.json`. It completes
both prediction files before either independent official scoring process runs.
Both results are reported; no further fitting, adjustment or test-based model
selection is performed. The existing submitted method remains unchanged, as
specified before this follow-up evaluation.

| Frozen method | Official test correct /467 | Official test accuracy | Fresh prediction + scoring |
|---|---:|---:|---:|
| Existing submitted contextual model (previously verified) | 392 | 83.94004282655246% | Previous measurement |
| Context-cluster robust training | 389 | 83.29764453961457% | 0.803 s |
| Reference-operation plan supervision | 388 | 83.08351177730192% | 0.883 s |

These one-question validation gains did not transfer to higher test accuracy.
This is an evaluation finding, not a basis for adjusting the models or selecting
between them on test. No new leaderboard score is claimed. All earlier statements
about no test access refer to the initial development phase, before this explicit
follow-up request. This follow-up is disclosed rather than presented as an
untouched one-shot test protocol for the entire project.

The exact official score JSONs, complete predictions, source/asset hashes and
fresh-process resource measurements are in `runs/round4_frozen_test/`. The runtime
was the same NumPy-only Python 3.12 environment on the local Windows CPU, with
four numerical threads. No reference Xeon timing is claimed. To reproduce both
unchanged checkpoints without training, from `code/`:

```bash
python evaluate_round4_frozen.py --output runs/round4_test_reproduction
```

The output directory must be new to prevent overwriting existing evidence.
