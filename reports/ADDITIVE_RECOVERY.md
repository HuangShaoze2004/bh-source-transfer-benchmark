# Dreher–Doyle additive identity recovery check

Decision: retain **Dreher Doyle 2018** outside the primary training and test sets. Public original additive identities are available, but a reliable identifier-based mapping to the 4,312 curated rows has not been established. Their presence alone does not justify assigning them to curated experiments.

This was a bounded startup investigation under the 15-minute limit. No model outputs were examined, no wet experiment or quantum calculation was performed, and yields were never used to infer a row mapping. The check stopped when the absence of a common stable identifier was established; the four-hour maximum was not treated as a required expenditure.

## Sources inspected

- The original authors' [rxnpredict repository](https://github.com/doylelab/rxnpredict/tree/57e15fdb7f7483c6bf3a601df69f6ac9e5af6965), frozen at `57e15fdb7f7483c6bf3a601df69f6ac9e5af6965`, MIT license. Its [data_table.csv](https://github.com/doylelab/rxnpredict/blob/57e15fdb7f7483c6bf3a601df69f6ac9e5af6965/data_table.csv) contains `plate`, `row`, `col`, additive names/numbers/SMILES, substrates, ligand, base and outcomes.
- Its [build_table.py](https://github.com/doylelab/rxnpredict/blob/57e15fdb7f7483c6bf3a601df69f6ac9e5af6965/build_table.py) joins measured information to plate layouts by `plate,row,col`, explicitly validating a one-to-one merge. The [plate specification](https://github.com/doylelab/rxnpredict/blob/57e15fdb7f7483c6bf3a601df69f6ac9e5af6965/main.py), additive dictionary and layout tables were also inspected. The original computational-design script excludes controls and additive 7; that exclusion is another reason not to assume interchangeability between common derivative benchmarks.
- The [RXN-Yields repository](https://github.com/rxn4chemistry/rxn_yields/tree/d9e6b87ce1b881978490d68bfc00021e3b48127a), frozen at `d9e6b87ce1b881978490d68bfc00021e3b48127a`, exposes `data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx`. Its availability was confirmed, but no evidence links its reordered benchmark rows to the curated collection's row index.
- The project's [curated BH snapshot](https://github.com/schwallergroup/bh-hte-ood/blob/3a44e398d9fed3f08cf396f1153be2960193f581/data/BH_HTE_Curated_Dataset_v20260219.csv) and accompanying repository files do not supply an original-well-to-curated-row crosswalk.

Downloaded original-source evidence and license are stored read-only under `data/audit/additive_recovery/`. Its `manifest.json` records source URLs, commit, byte counts and SHA256 values. These files are evidence for this audit, not additional training data.

## Identifier and ambiguity findings

| Audit item | Observation |
|---|---|
| Original public table | 4,599 rows, with 4,599 unique `plate,row,col` triples |
| Original nonempty additive structures | 23 distinct SMILES |
| Curated Dreher Doyle subset | 4,312 rows |
| Curated `plate,row,col` | All absent |
| Curated additive number/name/SMILES | All absent |
| Curated unnamed integer index | Exactly the global 0…27,496 export index across all eight sources |
| Curated additive flag patterns | 9 patterns across eight coarse columns |
| Five chemical-role SMILES plus eight flags | 1,620 groups; 897 contain multiple rows; largest contains 7 rows |

The curated global export index is not documented as an original experiment or well identifier. Equal ordering after guessed control removal or yield-based matching would be an unsupported reconstruction. Coarse additive flags also do not uniquely encode the original 23 additive structures. Substrate/reagent condition keys can describe several experiments with different additives; they are not a substitute for a documented row identifier.

The original source demonstrates that original experiments can be mapped to additives **within the original plate coordinate system**. It does not supply the missing bridge from that system to this curated export. No claim is made that recovery is impossible; it is unverified within the authorized bounded check.

## Consequences and minimum condition for reconsideration

Primary data remain unchanged. `sensitivity/dd_original` includes all original 27,497 curated rows with the same legal input roles, recomputed fingerprints and descriptors, and newly recomputed strict LOSO splits. It does **not** fabricate an additive, use the coarse flags as a replacement identity, or claim chemically complete conditions. Its purpose is to show sensitivity to including the incompletely specified source.

Reconsideration requires a public, traceable crosswalk from each curated row to original `plate,row,col` (or another documented stable experimental ID), followed by complete one-to-one mapping checks, chemical consistency checks, explicit unmatched/duplicate reports, and protocol amendment before any confirming evaluation. Yield coincidence, model residuals, or inferred row order do not meet this condition.
