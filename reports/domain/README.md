# Fixed-split domain characterization

No outer prediction files were read. These outputs describe data coverage, not predictive performance.

Similarity is the maximum Morgan radius-2, 2048-bit Tanimoto to unique structures in the corresponding outer development set, separately by substrate role. Chirality is excluded from fingerprints; a similarity of 1 can occur without entity identity because of fingerprint collisions or unrepresented distinctions. Group-based leakage checks remain separate.

The aryl heteroaromatic flag means any fully aromatic ring containing a non-carbon aromatic atom. Halogen classes list F/Cl/Br/I attached directly to aromatic atoms; this does not identify the reacting leaving group. Amine flags count nonaromatic primary/secondary N-H sites, excluding simple amide/thioamide/sulfonamide connections, and aromatic N atoms. Primary or secondary class takes precedence over incidental aromatic N; substrates with both aliphatic classes are 'other' with a mixed flag. These are coarse structural categories, not mechanistic assignments.

Reagent coverage is exact canonical charge/stereo/salt-preserving structure membership in outer training rows. Source class rates distinguish the full source from the retained strict-OOD test subset. The DD common-row files provide fixed shared test IDs for fair paired sensitivity comparisons.

Optional future stratified metrics require explicit --predictions paths and a FROZEN, evaluation_allowed=true lock; the full saved prediction panel must pass the project's independent completeness validation first. No such option was used for characterization.
