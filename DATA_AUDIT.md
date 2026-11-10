# Data audit

Frozen author commit: `3a44e398d9fed3f08cf396f1153be2960193f581`. CSV SHA256 `247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333`.

Code: Apache-2.0; Zenodo dataset: CC-BY-4.0. Attribution: Neves, Hao, Aikonen, Diccianni, Wegner, Schwaller and Strambeanu, Robust out-of-distribution prediction of Buchwald–Hartwig reactions, DOI 10.1038/s43588-026-01017-6. Cite each original source when distributing derived results.

Raw rows: 27497. Main usable rows: 23185. Exclusions are enumerated in data/audit/excluded_rows.csv; no model residuals or label conflicts are used for exclusion.

## Outcomes and measurement

Primary label is Yield >= 10 (percentage scale); exactly 10 is positive. 155 boundary rows; 28 disagreements with author labels, enumerated in label_disagreements.csv. Author labels remain available only as metadata/sensitivity targets. Values above 100 are retained, not clipped.

The paper reports heterogeneous CAD assay yields, isolated yields, MisER and LCAP; the JNJ assay uses calibrated UPLC-MS/CAD. These quantities are not interchangeable precise isolated yields. The shared 10% threshold is a harmonized feasibility endpoint. Temperature, time, concentrations and detailed loading are absent from this CSV, limiting causal/condition-specific interpretation.

## Input permissions and normalization

No product enters any prediction feature. Public code demonstrates product-based recommendation but does not establish every CSV product as a prospective target. Product SMILES remains metadata. R0 is therefore a declared input-only DRFP substitute; R1 is a declared substitute for the author feature representation, not an exact reproduction or a claim to outperform the published model.

Model input canonicalization retains salts, formal charges, isotopes, stereo and coordination bonds; it removes atom map indices and standardizes equivalent SMILES. Each role remains explicit in metadata and sequence serialization has fixed aryl/amine/catalyst/base/solvent order. Grouping is deliberately more conservative: largest organic fragment(s), uncharged canonical tautomer, stereo/isotope removal. This grouping only establishes identity and does not change the chemical model input. Both substrate roles are pooled when checking cross-boundary entity overlap.

Token audit found that the original RXNFP vocabulary cannot represent dative bond arrows. Before any model performance was observed, the original vocabulary was extended by exactly two complete bond tokens ('->', '<-'), preserving all original token IDs. This applies equally to every neural comparison and its effective hash is recorded in token_audit.json. Canonical dative SMILES remains intact. Three [NaH+] entries are represented as graph-identical explicit-H SMILES; RemoveHs/canonical-isomeric equality and unchanged charge are verified. No original chemical bond or charge is repaired by assumption. All full reactions must pass strict unknown-token, unmatched-character and 512-token checks; no silent truncation occurs.

All repeated inputs and conflicting outcomes are retained. They remain attached to their substrate entities/pair. Conflicts can reflect repeated experiments or unreported conditions and are not classified as wrong labels. Source names, row IDs, labels and author clusters never enter features.

Dreher Doyle 2018 is excluded from both main training and testing because additive identities are unavailable in the curated CSV. Eight coarse additive flags are not identities; no verified original-row mapping has been recovered. Excluded rows remain in the immutable original data for a separate declared sensitivity experiment. No claim of restored identity is made.

## Descriptors and fingerprints

Both stored DRFP columns are truncated in every row and are never parsed. Original QM is shape (27497, 50), with 0 nonfinite entries. Role slices 0:7 aryl, 7:14 amine, 14:21 catalyst, 21:28 base, 28:35 solvent and 35:42 product are checked against structure identities. QM[42:50] matches the eight additive flags in CSV column order, with 0 mismatching rows. The 7 descriptor identities/provenance are not established by available code, so original QM is excluded from every primary predictor.

Legal replacement: each catalyst, base and solvent receives MolWt, MolLogP, TPSA, HBD, HBA, RotatableBonds, FormalCharge (21 public RDKit values total). All neural variants receive the identical descriptor vector. Any fitted scaling/imputation is deferred to training partitions; none is fitted here. Known empty reagent components receive seven zeros; required substrate parsing failures are explicitly excluded. Author-style substrate fingerprints reproduce Morgan OR construction at radius 3/2048 bits, with an explicitly empty product side. Standard DRFP is recalculated separately. Its upstream reaction-string parser splits on every '>', which would mistake dative arrows for reaction delimiters and silently omit input components. We therefore call the same official molecular shingling/hash/fold functions directly; for an empty product side, the reaction fingerprint is exactly the union of all input shingles. Unit tests verify equality to the original API on ordinary reactions and contributions from all five roles in a dative reaction. This is an input parsing adaptation, not a new fingerprint algorithm.

## Fixed split verification

Outer leave-one-source-out development uses all other main sources. A held-out row is scored only if neither substrate entity appears anywhere in development, including inner validation. Inner source holdouts are filtered by the same rule; at most two valid sources with the greatest remaining pair counts are selected. A source needs at least 30 pairs, positive/negative coverage of at least 10 pairs each, and 20 positive/negative rows each for the primary macro average. Other nonempty sources are descriptive; empty folds do not score.

| Source | Rows | Pairs | Positive rows | Negative rows | Eligible |
|---|---:|---:|---:|---:|---|
| AZ ELN 2023 | 607 | 376 | 467 | 140 | True |
| Dreher 2014 | 0 | 0 | 0 | 0 | False |
| Dreher 2016 | 0 | 0 | 0 | 0 | False |
| HiTEA 2024 | 1387 | 15 | 194 | 1193 | False |
| JNJ HTE 2024 | 8016 | 546 | 2917 | 5099 | True |
| Merck 2025 | 2668 | 2668 | 1080 | 1588 | True |
| Roche 2023 | 1680 | 65 | 754 | 926 | True |

All outer and inner row, entity, pair and input-key intersections are asserted empty. Full mappings, field/missingness tables, conflicts, exclusions and hashes live under data/audit and data/processed. These are data eligibility audits, not model test performance. Same-role versus pooled-role grouping sensitivity is saved separately; the current snapshot gives identical retained rows under these two checks. Merck's retained test pairs share only eight amine entities, so pair counts must not be mistaken for independent coverage of thousands of amines; shared-substrate bootstrap sensitivity remains necessary.

## Provenance

- https://github.com/schwallergroup/bh-hte-ood/tree/3a44e398d9fed3f08cf396f1153be2960193f581
- https://zenodo.org/records/19636649
- https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13593570/fullTextXML
