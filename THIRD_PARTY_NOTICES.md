# Third-party attribution

The experiment uses the following public artifacts. Original license files,
version identities and modification records accompany their respective files.

| Artifact | Attribution and provenance | License record |
|---|---|---|
| Curated Buchwald–Hartwig data | Neves et al., *Robust out-of-distribution prediction of Buchwald–Hartwig reactions*, DOI 10.1038/s43588-026-01017-6. Zenodo record 19636649. Original source citations are retained in the author's README and the dataset metadata. | CC BY 4.0, documented by `data/raw/zenodo_public_metadata.json`. Standardization, labels, exclusions and feature changes are recorded in `DATA_AUDIT.md`. |
| Author benchmark code and notebooks | schwallergroup/bh-hte-ood, commit 3a44e398d9fed3f08cf396f1153be2960193f581. | `third_party/bh-hte-ood/LICENSE.txt`, Apache-2.0. |
| Generic RXNFP weights and tokenizer reference | Schwaller et al., *Mapping the Space of Chemical Reactions Using Attention-Based Neural Networks*, DOI 10.1038/s42256-020-00284-w. rxn4chemistry/rxnfp, commit 6fd48f4927c2178555cc5d71dbfb225fb178f43c. | `third_party/rxnfp/LICENSE`, MIT. Weight and vocabulary identities are in `weights/rxnfp/manifest.json`. The dative-token extension has a separate manifest. |
| GroupDRO reference implementation | Sagawa, Koh, Hashimoto and Liang, ICLR 2020, arXiv 1911.08731. kohpangwei/group_DRO, commit cbbc1c5b06844e46b87e264326b56056d2a437d1. | `third_party/group_DRO/LICENSE`, MIT. |
| Reaction-yield fine-tuning reference | Schwaller et al., *Prediction of chemical reaction yields using deep learning*, DOI 10.1088/2632-2153/abc81d. rxn4chemistry/rxn_yields, commit d9e6b87ce1b881978490d68bfc00021e3b48127a. | `third_party/rxn_yields/LICENSE`, MIT. |
| Public additive-recovery audit material | Ahneman, Estrada, Lin, Dreher and Doyle. doylelab/rxnpredict, commit 57e15fdb7f7483c6bf3a601df69f6ac9e5af6965. | `data/audit/additive_recovery/LICENSE.txt`, MIT. The attempted mapping and unresolved identities are documented in `reports/ADDITIVE_RECOVERY.md`. |
| SWA method | Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization*, UAI 2018, arXiv 1803.05407. | Method citation and implementation differences are recorded in `reports/LITERATURE_DIFFERENCES.md`. |

The project contains its own controlled experimental implementation and derived
artifacts. Third-party license notices apply to the corresponding material.
