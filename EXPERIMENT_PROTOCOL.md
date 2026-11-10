# Protocol (working; not yet FINAL_EVAL_LOCK)

## Task and data
Predict recorded Yield>=10 using aryl substrate, amine, catalyst, base, solvent and lawful structure-derived descriptors. Measurements are heterogeneous CAD assay/isolated/MisER/LCAP: this is threshold feasibility, not precise yield or uniformly measured success. Preserve out-of-range reported yields; do not silently clip. Raw data remain immutable. Preserve repeat observations and conflicting labels; do not remove rows by model residual or average/select outcomes. Missing temperature/time/additives restrict applicability.

## Splits
After normalization, leave one source out. A test row survives only when BOTH its aryl entity and its amine entity are absent from the entire outer development set. Sources with >=30 pairs, >=10 positive-containing pairs, >=10 negative-containing pairs and >=20 rows of each class qualify. The outer training sources retain all otherwise valid rows. Inner validation repeats source holdout plus individual-substrate novelty, taking up to two largest valid source folds (>=10 pairs and both labels); ties lexical. No valid inner split: fixed predeclared default, explicit limitation. Pair conditions never randomly straddle inner boundaries. All methods share rows and splits.

## Models and tuning
R0_RF and R0_HGB: standard full-available-input DRFP; both retained to avoid test-based baseline selection. R1_substitute: author's Morgan-OR/XOR substrate fingerprint plus 21 explicit reagent descriptors unless original descriptors and products are proven lawful before freeze. Author RF: 500 trees, entropy, min_samples_split=4, min_samples_leaf=1, max_features=log2, class_weight=balanced, unlimited depth. All random core methods seeds 17/42/2026.

B/B_S, B_bal/B_bal_S, B_A/B_A_S share the same publicly released self-supervised RXNFP bert_pretrained encoder, tokenizer, descriptor input, newly initialized head and basic regularization. No supervised BH weights. AdamW, dropout .1. Initial grid: lr {1e-5,3e-5}, weight_decay {.01,.1}, 3 objectives =12 configurations, tuning seed42. GroupDRO eta=.01. Balanced ERM and DRO use identical group-balanced effective batches (8 examples/source). ERM matches total batch size and updates. Complete all groups before one DRO update; detached group means and log-space q.

Representation amendment before performance: retain original591 vocabulary IDs and append only whole dative-bond tokens ->/<-; new embeddings copy the original single-bond embedding. This avoids dropping720 catalyst rows and does not alter molecular graphs. Original and extended tokenizer artifacts are both hashed. R0 invokes official DRFP molecular shingling/hash/fold directly to avoid arrow/reaction-separator parser ambiguity. See DECISIONS.md and equivalence tests.

Schedule: warmup first10%, reduce LR to .1*base at50%, constant thereafter. Ordinary and SWA variants use same trajectory and candidate checkpoints. SWA starts at50% or75%, chosen internally. Each family has equal selection budget. Fit descriptor imputation/scaling only on training rows. Inner validation chooses hyperparameters and ordinary training fraction; formal final fit has no test validation. Fixed epoch ceiling selected from10,6,3 by measured full-matrix budget before lock. Training budget is matched even if a selected ordinary checkpoint is earlier.

SWA snapshot cadence is every quarter epoch after its start plus final update, so the3-epoch fallback also averages multiple checkpoints. Within each objective, ordinary and SWA share hyperparameters chosen by mean ordinary inner-source AUROC; SWA start is then selected on the same internal folds. This paired design prevents learning-rate or tuning-budget differences from explaining the averaging contrast. Tie-breaking follows fixed grid order; SWA tie chooses75%. Final ordinary fraction is the mean selected inner training fraction, rounded to an outer epoch.

No cross-fold performance-guided global retuning. At most two justified global revisions of <=4 configurations may be made only on designated development sources, within total development <=50 GPU hours; such sources excluded from confirmation.

## Evaluation
Freeze data/feature/weight hashes, groups, folds, model candidates, seeds, tuning/early-stop/SWA/calibration/threshold rules, update budgets and metric comparisons in FINAL_EVAL_LOCK.json before formal outer performance. All external predictions complete before release; only confirmed bugs justify fixes, with retained prior results and reruns of every affected method.

Primary: per-source AUROC per seed, average seeds, equal-source macro. Secondary: source AUROC/AP, worst source, prevalence, Brier, reliability; do not pool sources for primary AUC. Temperature calibration fitted from internal validation only. Default classification threshold .5; no external tuning.

5000 source-stratified paired substrate-pair bootstrap replicates; same row weights for every method/seed. Seeds and condition rows are not independent chemical units. Shared-aryl/amine two-way clustering sensitivity with1000 replicates and cluster diagnostics. Holm adjustment for prespecified confirmation contrasts. Report single-class replicate failures. Intervals conditional on observed source set. At least3 qualifying sources required for broad multi-source interpretation.

Acceptance: macro AUROC gain>=.02 versus strongest prescribed comparator, paired95% interval supports gain, no eligible-source decline>.03. Combined model must beat strongest single module and balanced+SWA. Select within nested development, never select winner from uncorrected external search. Report all results regardless of acceptance.

Top-k only among measured conditions of pairs with multiple conditions; disclose coverage. Analyze substrate types, structural similarity, reagent coverage and prevalence; attention/SHAP not mechanistic proof.

## Resource and correctness gates
GPU <=160h including failure/retry; development<=50h, feasibility<=12h, >=80h formal reserved. CPU<=500 coreh including subprocesses. Shared threads<=4; exclusive<=16. GPU heavy jobs serial and wait for unrelated jobs. Save optimizer/scheduler/q/SWA/RNG state and run ledger; resume after process/cache inspection.

Required checks: labels at boundary; structure/group mapping; train/test aryl and amine disjointness; pair grouping; training-only transforms; tokenizer no omission/UNK/truncation; nonzero encoder gradients; finite forward/backward; small-data overfit; hand-calculated GroupDRO; SWA averaging; resume equivalence. Independently rebuild main table solely from saved predictions.

Final status CONTINUE until complete; READY_FOR_AUTHOR_REVIEW requires complete strong controls, ablation, confirmation and credible increment. Otherwise NOT_READY for completed/budget-ended negative route, BLOCKED only for real obstacles with evidence. No paper or submission.
