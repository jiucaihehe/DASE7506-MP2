# Additional development: candidate relations and joint correctness

28 October 2026. **No improvement selected; the frozen 83.94% submission is unchanged.**
These are development results, not additional test scores. Neither test inputs
nor test references were loaded in this round. No test evaluation was run.

## Protocol

Fit only on supplied training data. Use three-fold stratified group CV with seed
7506, keeping all four candidates together and refitting normalization and PCA
within each training fold. Select one regularization setting per new family by
pooled OOF hard-group correct count, breaking ties in listed order. Save selection
before opening validation. Promote only if validation hard accuracy exceeds the
frozen contextual model's 368/419. Settings and optimizer diagnostics are in each
experiment directory's `protocol.json` and `cv_selection.json`.

This round occurs after the earlier frozen model was scored on test. That earlier
aggregate result is known; it was not used as a search objective here. Repeated
development on the same validation bank can overfit it, so these small differences
are development evidence rather than proof of a generalization gain.

## New methods

1. **Answer-aware messages.** Keep the existing context-conditioned candidate
   features. For every candidate, average the group-centered embeddings of its
   other peers with the same parsed answer. Add this vector and its elementwise
   product with the candidate residual. A candidate with no same-answer peer gets
   a zero message. This retains candidate-level information rather than collapsing
   the group into one vector per answer. The new block has 2048 coordinates.
2. **Diagonal residual energy.** Add the 1024 squared coordinates of each
   candidate's group-centered embedding. This learns a discriminative diagonal
   quadratic signal directly, rather than estimating class-conditional covariance
   matrices as in the previous Gaussian experiment.
3. **Joint correctness subsets.** Enumerate the 15 nonempty subsets of four
   candidates. Forbid states that disagree on correctness of equal parsed answers,
   or label an unparseable candidate correct. Assign each state the sum of its
   candidate linear scores plus a learned prior for the number correct. Fit the
   exact correct subset by conditional likelihood, using all candidate labels,
   including all-correct groups. Select the candidate with highest posterior
   correctness marginal. This differs from the prior objective that rewards total
   probability mass on any correct candidate.

The first two families use base L2 0.02 and contextual-block L2 1.0; select new-block
L2 from 0.1, 1, 10. Their matched control uses identical contextual features with
the same L-BFGS optimizer. The joint model uses the original 1070 features and
selects L2 from 0.005, 0.02, 0.08; count-prior L2 is fixed at 0.001. All fits start
at zero. L-BFGS has a 350-iteration cap, ftol 1e-11, gtol 1e-7. All recorded CV fits
reported convergence. The current submission still uses its original Adam fit.

## Results

| Method | Selected new-block/model L2 | Train OOF hard /2513 | Validation hard /419 | Validation all /798 | Rescued / regressed vs frozen |
|---|---:|---:|---:|---:|---:|
| Frozen contextual model (previous record) | — | 2096 | 368 (87.83%) | 747 | — |
| Matched L-BFGS contextual control | — | 2087 | 368 (87.83%) | 747 | 0 / 0 |
| Answer-aware messages | 10 | 2087 | 368 (87.83%) | 747 | 1 / 1 |
| Diagonal residual energy | 10 | 2085 | 368 (87.83%) | 747 | 0 / 0 |
| Joint correctness subsets | 0.08 | 2049 | 354 (84.49%) | 733 | 11 / 25 |

Message and energy features preferred strong shrinkage and did not beat their
matched control. The joint model rescues some errors but creates more than it
fixes. This does not establish an accuracy ceiling or imply that all methods make
identical errors.

As a final development check, select a mixture of the frozen contextual model's
softmax probabilities and the joint model's correctness marginals using training
OOF predictions only. Predeclared joint weights were 0, 0.1, 0.25, 0.5, 0.75, 1.
Their hard correct counts were 2096, 2092, 2094, 2088, 2076, 2049. Weight **zero**
won; the validation evaluation therefore reproduces the unchanged contextual
model. No validation search over mixture weights was performed.

## Reproduction and artifacts

From `code/`, use the pinned `requirements-new-experiments.txt` environment:

```bash
python experiment_relations.py
python experiment_joint_correctness.py
python experiment_joint_fusion.py
```

The fusion script also reads the already tracked original contextual OOF and
validation diagnostics under `runs/bilinear_experiment/`. The first two commands
regenerate their local model/diagnostic NPZ files, which are excluded from Git as
rejected development assets. Protocols, CV selections and validation JSON files
are tracked under `runs/relation_experiment/`, `runs/joint_experiment/` and
`runs/joint_fusion/`. No inference dependency or submitted checkpoint is added.

Local CPU: Intel Core Ultra 9 275HX, Windows, Python 3.12, four numerical threads
per experiment process. Relation/control search and full refits took 91.37 s;
joint search and full refit took 22.99 s. These are training/development timings,
not complete-bank inference benchmarks. Dependencies remain the pinned NumPy,
SciPy, scikit-learn and threadpoolctl experiment requirements.

The selected submission checkpoint SHA-256 remains
`cc8483cb50c6b7287613897b5b78178b2accd0780888b5ac227d46f1fc984537`.
Its teacher entrypoint, English final report and official result remain unchanged.
**The latest verified test result remains 392/467 = 83.94004282655246%; this round
does not claim a higher test result.**

OpenAI Codex designed and implemented these experiments and drafted this record.
No external data or pretrained weights were used. The student remains responsible
for reviewing and understanding the work.
