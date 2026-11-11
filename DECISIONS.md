# Decisions

## 2026-11-10: scope and isolation
- Implementation authorized by user after planning. New project only; no writes to other projects or their environments.
- Shared-resource mode defaults to four total CPU threads. Workers each use one thread while parallel implementation is active. Only root launches GPU training.
- Python 3.12.3 is locally available and supports chosen modern packages; use a clean project venv rather than modifying global Python. This replaces the provisional Python 3.11 environment without changing experimental methods.

## Inputs and baseline naming
- CSV at bh-hte-ood commit 3a44e398d9fed3f08cf396f1153be2960193f581 is authoritative; archive is separately versioned.
- Primary label is Yield >= 10. Author labels retained; 28 AZ discrepancies and 155 exact-boundary rows require saved audit.
- Current public evidence does not establish every CSV Product SMILES as a predeclared target. Primary models therefore exclude products unless source-level evidence completes before input freeze.
- Original QM has 50 values; role slice 14:35 is verified but seven descriptor identities and origin remain unverified. Primary fallback is an explicit 21-dimensional RDKit reagent descriptor vector, shared by all neural methods and R1_substitute. No new DFT.
- R1_substitute uses the author's Morgan(radius=3,2048)-OR/XOR fingerprint construction on lawful substrate inputs, plus explicit reagent descriptors and documented author RF settings. It is not claimed as an exact production-model reconstruction or evidence of surpassing the original model.
- R0 uses standard DRFP on all lawful input structures with empty product side if product permission remains unverified.

## Confirmation rules
- A priori eligible source: >=30 independent substrate pairs, >=10 pairs containing each label, and >=20 rows of each label. Empty sources not scored; small sources descriptive.
- Practical improvement default: >=0.02 macro AUROC versus strongest prescribed comparator, paired interval supporting improvement, no eligible source decline >0.03. Combination must improve over strongest single module and balanced+SWA. These are engineering acceptance defaults, not universal scientific thresholds.
- Candidates and nested selection rules are fixed globally before testing; never pool outer-fold feedback to change other folds. Global performance-guided revisions require designated development sources excluded from confirmation.
- Calibration is auxiliary and cannot establish ranking improvement. Each seed remains a single model; no undeclared ensemble.

## Access findings
- Nature direct access redirected; legitimate Europe PMC full text was readable. SI retrieval encountered TLS error, captcha HTML, and timeout. No access controls bypassed. Missing descriptor provenance remains explicit.

## Representation corrections before any model performance (2026-11-10)
- 720 rows (Dreher2014 catalysts) contain genuine dative bonds. The original RXNFP vocabulary cannot represent either direction: it contains reaction `>>` but no single `>`. Equivalent SMILES traversals and graph-verified ring closures do not remove this vocabulary limitation. Deleting these rows would nearly remove a training source.
- Therefore retain all chemistry and extend the fixed vocabulary by exactly two whole-bond tokens `->` and `<-`, keeping all original 591 IDs unchanged. Initialize both new token embeddings from the pretrained single-bond `-` embedding, with no extra pretraining. Every neural model uses this identical, documented tokenizer extension; retain original weights and vocabulary hashes plus extended vocabulary manifest. This is an explicit deviation from using the unmodified tokenizer, required by input coverage and chosen before any performance.
- Three `[NaH+]` rows can be represented with explicit bonded H while retaining charge and RDKit roundtrip identity; do not reinterpret these as disconnected sodium hydride ions.
- drfp0.3.7 splits reaction strings on every `>`, so dative arrows can silently corrupt reagent parsing. For the predeclared empty product side, construct standard DRFP directly from official molecular shingles, hash and fold APIs, with tests for exact equivalence on ordinary strings. Dative bonds and all roles remain present; this is a parser adaptation, not a new fingerprint algorithm.
- SWA snapshots every quarter epoch after the selected start (including first eligible step and final step), to ensure actual averaging even if the budget selects three epochs. Ordinary validation remains at epoch boundaries and every paired method has identical optimizer updates.

## Environment bootstrap
- Main packages pinned in requirements.lock.txt. Initial large wheel downloads were interrupted by this agent because throughput was poor; no GPU work was started. The installed PyTorch2.8.0+cu128 and its runtime dependencies were independently copied into this project's venv (not symlinked), with distribution provenance saved. Other pinned wheels installed from the public Tsinghua PyPI mirror. Other environments were not modified.
- Setup outside the central job launcher is charged a conservative one CPU-core-hour allowance, covering environment work, initial downloads, data preparation and CPU unit tests; this is explicitly an upper-bound allowance, not a measured exact cost. Subsequent guarded runs use measured CPU accounting and elapsed allocated GPU time, including failures.

## Post-freeze delivery audit, 2026-11-10T03:23:19.879998+00:00
- Added an independent auditor for actual optimizer updates, training-only IDs, selection configs, all resumable states, sample exposure, normalized group weights/ESS and multiple SWA snapshots. It is invoked only after completed evaluation by the report generator. No frozen model, input, split, metric or selection code changed.
- Six new fault-injection tests and twelve existing report regression tests passed. A completed internal ERM trajectory passed real-state audit (4560 updates; 21/11 SWA snapshots). Final 45-trajectory equality and diagnostics remain pending until those runs exist.

## Real GPU GroupDRO integration check, 2026-11-10T03:27:54.240345+00:00
- Four inner trajectories completed and the first GroupDRO worker was verified live by PID/start time and increasing CPU time. No failed job status observed.
- Independently recomputed q_g proportional to exp(0.01 times cumulative logged group loss) for the first600 complete logged steps of dev_az_eln_2023_dro_g0_i0. All5 sources had8 observations per effective batch; losses were finite, weights summed to1, and maximum absolute recurrence error was5.84049e-7. Evidence: reports/GROUPDRO_RUNTIME_CHECK.json. This validates only the recorded training prefix, not unseen-source performance or final matrix completion.

## First completed GPU GroupDRO trajectory audit, 2026-11-10T03:38:15.837032+00:00
- Recomputed all4560 recorded GroupDRO weight updates independently; maximum error1.89093e-6. Full resumable state, exposure counts, normalized group weights and21/11 SWA snapshots passed the saved-state auditor.
- Corresponding ERM, balanced ERM and GroupDRO inner trajectories each executed4560 optimizer updates. This is controlled-budget implementation evidence only; no outer performance was read or used to modify candidates.
- review_first_completed_groupdro_v1 was CPU-only, measured and charged by the central guard. Report: reports/GROUPDRO_COMPLETED_TRAJECTORY_AUDIT.json.

## First-fold tree controls executed and audited, 2026-11-10T03:46:04.779797+00:00
- All6 precomputed, already-locked tree inner jobs completed successfully: R0_RF/R0_HGB/R1_substitute on the same2 internal source holdouts. Validation probabilities were finite and bounded; IDs/lengths matched their fixed specs. No predictive performance summary was inspected.
- Deserialized estimators were independently checked: R0 representations2048 dimensions; R1 substitute2069 dimensions; expected seed42, RF300/500 trees, entropy/balanced/log2/min_samples_split4, and HGB200 iterations with early_stopping=False. Report: reports/FIRST_FOLD_TREE_RUNTIME_AUDIT.json.
- Their existing run identities and output paths allow the main pipeline to reuse completed jobs. CPU and GPU workers used separate CPU slots; no parallel GPU task was launched.

## Complete main matrix: overlap evaluation CPU with sensitivity GPU, 2026-11-10
- All45 main formal neural trajectories and all45 tree fits completed. The saved-state auditor was additionally run before external evaluation, passed all45 trajectories and preserved its evidence in reports/ALL_FORMAL_NEURAL_RUNTIME_AUDIT.json. This operationally extends the earlier post-release-only audit schedule; no training, selection or metric implementation changed.
- The frozen main release gate is satisfied by all135 prediction files and387666 model/seed/row predictions, with identity/truth/chemistry checks and hashes in artifacts/PREDICTION_COMPLETENESS.json. Both sensitivity locks and inherited choices were fixed before any outer metric access. Their incomplete prediction matrices remain separately sealed.
- Started the controller's exact existing review_primary_evaluation_v1 command early on one guarded CPU slot, while the author-label sensitivity uses the sole GPU and one other CPU slot. This changes task scheduling only: the full main matrix is evaluated uniformly, with the unchanged frozen metrics, bootstrap seeds/counts and acceptance rules. No outer result can change any remaining sensitivity configuration, candidate or selection rule.
- The controller later calls the identical guarded run identity and reuses its completed result. Launch/process details are retained in artifacts/PRIMARY_EVALUATION_PRECOMPUTE.json and the central ledger; do not start a second copy. The monitor reports bootstrap counters only, without opening metric tables or prediction arrays. Final reporting and scientific readiness still require both sensitivities and independent metric reconstruction.

## Common-row reporting correction, 2026-11-11
- The first completed author-label stage report averaged the common-row deltas over all five nonempty sources. This included descriptive HiTEA. The user requires small independent-pair sources to remain outside macro averages. The frozen primary and sensitivity evaluators already used the correct four eligible sources.
- Corrected the reporting aggregation to use the intersection of the two frozen eligible-source lists. It averages seeds within each source, then weights those sources equally. Missing or single-class eligible-source values propagate as NA. The summary now states its source count and source names. The paired CSV retains the per-source descriptive records.
- Saved the affected report, paired table, reporting source and status snapshot under `reports/history/` before the correction. The author-label stage report is regenerated for every affected model and both truth regimes. DD reporting will use the same corrected function after its full matrix completes.
- Added regression cases for descriptive-source exclusion and missing or undefined eligible sources. The independent reporting suite passed all15 tests. Training specifications, prediction files and frozen metric definitions retain their recorded hashes.
