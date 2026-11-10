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
