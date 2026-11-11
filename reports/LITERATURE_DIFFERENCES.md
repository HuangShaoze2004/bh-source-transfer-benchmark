# Startup literature audit and controlled experimental differences

Access date: 2026-11-10 UTC. This is an experiment-design record, not a paper draft.
Only public sources were consulted. No subscription bypass, paid API, new quantum
calculation, or external experimental work was used for this audit.

## Direct evidence and boundaries

| Primary source | Established result/design | Difference being tested here |
|---|---|---|
| Neves, P.; Hao, B.; Aikonen, S.; Diccianni, J. B.; Wegner, J. K.; Schwaller, P.; Strambeanu, I. I. (2026), *Robust out-of-distribution prediction of Buchwald–Hartwig reactions*, Nature Computational Science 6, 1015–1023. DOI [10.1038/s43588-026-01017-6](https://doi.org/10.1038/s43588-026-01017-6). | Combines multiple reaction sources; uses a 10% binary feasibility threshold; compares DRFP and physicochemical representations, including strong RF; excludes test rows containing an aryl or amine seen in training in its source benchmark. Uses DRFP clustering and model-guided experimental data generation. | Fixed leave-one-source-out evaluation; strict novelty against the entire outer development set; pretrained RXNFP ERM, balanced ERM, GroupDRO, SWA and strong tree controls under matched budgets. These are controlled experimental increments, not new algorithms or the first multisource/OOD study. |
| Ha, S. K. et al. (2025), *Developing Pharmaceutically Relevant Pd-Catalyzed C–N Coupling Reactivity Models Leveraging High-Throughput Experimentation*, JACS 147, 19602–19613. DOI [10.1021/jacs.5c00933](https://doi.org/10.1021/jacs.5c00933). [Public author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC13047508/). | Reports 4204 unique products at a fixed reaction condition; uses a 20% product-LCAP classification threshold. Explicitly distinguishes both-substrates-out-of-scope from an unseen pair whose two substrates were each seen. | Multisource transfer with changing reagent conditions, the predefined 10% label, and source-based robust training. Both-unseen splitting itself is established prior work. |
| Saebi, M. et al. (2023), *On the use of real-world datasets for reaction yield prediction*, Chemical Science 14, 4997–5005. DOI [10.1039/d2sc06041h](https://doi.org/10.1039/d2sc06041h). [Primary full text](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d2sc06041h). | Reports predictive HTE models but poor performance on heterogeneous ELN yield prediction; transfer/pretraining did not resolve that setting. | Feasibility classification rather than quantitative yield prediction; explicitly retain and explain negative outcomes, measurement heterogeneity and coverage gaps. The study does not guarantee successful transfer. |

The repository/Zenodo bibliography for Saebi et al. has inconsistent author/page
information; the DOI and original journal metadata above are authoritative.

## Input and label audit implications

- The Neves paper combines CAD, isolated yield, MisER and LCAP measurements. The
  JNJ methods describe assay yields. A common binary threshold does not make
  those measurements equivalent; the experiment must report source-specific
  measurement provenance and limitations.
- Some original prose defines experimental success using `yield > 10%`; this
  project fixes `Yield >= 10` as instructed. Preserve author labels and report
  all discrepancies, including boundary observations; do not retrospectively
  select the label definition that performs best.
- The original work already used a substrate–product DRFP plus reagent descriptor
  representation. The public recommender demonstrates the representation but
  does not by itself establish the provenance of every product in the combined
  dataset. In `Reagent Recommender.ipynb`, cell 7 reads a product from an
  experimental-results CSV, cell 8 calls it a target reaction, and cell 10
  encodes it. This is insufficient proof that every merged CSV product is
  independent of experimental outcome. The main protocol must resolve that
  question before admitting product input; otherwise use a documented
  product-free replacement and do not claim exact reconstruction of the
  author's production model.
- Product strings on failed reactions support a target-product interpretation,
  but are not alone a row-level provenance proof. Virtual-reaction scoring in
  the paper likewise does not establish every source's curation procedure.
- Strict novelty means neither test aryl nor test amine occurs anywhere in outer
  development, including validation/calibration. An unseen pair alone is not
  sufficient. Model selection must remain nested within each outer fold;
  cross-fold feedback cannot be used to tune a globally selected method while
  preserving all those sources as independent confirmation.

## Public source and license metadata

| Artifact | Public location / identity | License or access finding |
|---|---|---|
| Neves published paper | [Europe PMC full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13593570/fullTextXML), PMCID PMC13593570; [EPFL published version](https://infoscience.epfl.ch/entities/publication/ab5cd9bb-93d4-46b3-984d-8efa7e6e77a8). | Full XML successfully read; article license explicitly CC BY 4.0. Attribution and change notices required. Nature direct navigation redirected to an identity endpoint; PMC HTML sometimes presented a browser check. Public Europe PMC XML supplied the evidence without bypassing access controls. |
| Original data archive | [Zenodo 19636649](https://zenodo.org/records/19636649), DOI 10.5281/zenodo.19636649, v1, published 2026-04-18; `2026-02-21_df_HTE_BH_QM_10iter_9-3clustertest.pkl`, approximately 1.4 GB, advertised MD5 `c088ab8b56c78ceb381910a05265791c`. | Public JSON-LD identifies CC BY 4.0. A repeat check on 2026-11-11 is saved in `data/raw/zenodo_public_metadata.json`. The earlier human-readable-page uncertainty is resolved by this metadata. The archive pickle was not downloaded. |
| Author code | [schwallergroup/bh-hte-ood](https://github.com/schwallergroup/bh-hte-ood), inspected commit `3a44e398d9fed3f08cf396f1153be2960193f581`. | Repository advertises Apache-2.0. The original license and attribution are preserved. The data archive's separate CC BY 4.0 declaration is recorded above. |
| Ha author manuscript | PMCID PMC13047508, public since 2026-06-11, DOI above. | Public author manuscript access confirmed; do not assume that public access grants unrestricted redistribution. Store bibliographic metadata and original paraphrases, not copied full text without a verified license. |
| Saebi paper | DOI 10.1039/d2sc06041h, PMCID PMC10189898. | Original RSC text publicly accessible; this report uses short factual paraphrases. Full-text reuse license is not inferred here. |

The 2025 preprint, DOI `10.26434/chemrxiv-2025-xcr46`, is an earlier version of
the Neves article. Use the 2026 publication for substantive claims. The GitHub
CSV date and Zenodo pickle date differ; identity cannot be inferred from a
similar row count.

## Module originals and implementation provenance

The following records supplement the startup comparison. The original articles
and public implementation references were checked again on 2026-11-11 during
delivery review. They support the fixed methods already used in training.

| Primary method source | Established method | Implementation in this project |
|---|---|---|
| Sagawa, S.; Koh, P. W.; Hashimoto, T. B.; Liang, P. *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*, ICLR 2020. [Original article](https://arxiv.org/abs/1911.08731). | Minimizes worst-group risk over specified training groups using adversarial group weighting. Regularization and checkpoint selection affect generalization. | Groups are training sources. Log-space exponential updates use eta 0.01. Each effective batch contains all training groups, with replayed dropout randomness for the loss/backward passes. Inner validation selects regularization and checkpoints. The controlled evaluation measures transfer to held-out sources. |
| Izmailov, P.; Podoprikhin, D.; Garipov, T.; Vetrov, D.; Wilson, A. G. *Averaging Weights Leads to Wider Optima and Better Generalization*, UAI 2018. [Original article](https://arxiv.org/abs/1803.05407). | Averages weights along one optimization trajectory, with constant or cyclical learning rates. The averaged parameters define a single model. Batch-normalization statistics require handling where that layer is present. | AdamW uses the frozen warmup/decay schedule followed by a constant tail. Inner validation selects a 50% or 75% averaging start. Quarter-epoch snapshots include the final update. The encoder uses LayerNorm. Ordinary checkpoints receive the same complete training budget. |
| Schwaller, P. et al. *Mapping the Space of Chemical Reactions Using Attention-Based Neural Networks*, Nature Machine Intelligence 3, 144–152 (2021). [Original article](https://doi.org/10.1038/s42256-020-00284-w), [public preprint](https://arxiv.org/abs/2012.06051). | Learns reaction representations through masked-language pretraining and supervised reaction classification. The original Methods describe a 2.6-million-reaction Pistachio version 191118 dataset and a 90/5/5 split with identical products grouped. | The explicitly loaded `bert_pretrained` weights provide the generic masked-language encoder. A binary head and the same public RDKit reagent descriptors are used by all neural controls. Input omits products under the fixed permission rule. Original vocabulary entries are retained, with two audited dative-bond additions. |
| Schwaller, P.; Vaucher, A. C.; Laino, T.; Reymond, J.-L. *Prediction of chemical reaction yields using deep learning*, Machine Learning Science and Technology 2, 015016 (2021). [Original article](https://doi.org/10.1088/2632-2153/abc81d), [author implementation](https://github.com/rxn4chemistry/rxn_yields). | Fine-tunes a pretrained reaction encoder with a regression output layer. The implementation documents initialization from RXNFP's `bert_pretrained` directory. | This project adapts the fine-tuning entry point to the fixed binary feasibility target and nested source holdouts. The task-specific output head is initialized separately for each fixed seed. |

The commercial Pistachio corpus was neither obtained nor queried for this
project. The public generic checkpoint is used under the retained RXNFP MIT
license. Exact pretraining membership cannot be checked from the released
weights and available corpus metadata, so unlabeled exposure to test structures
remains unresolved. The pretraining data description above was verified in
`ms.tex` from the public arXiv v1 source archive. The guarded access record is
`review_rxnfp_original_provenance_v1` in the run ledger. This identifies the
reported training corpus without asserting a row-level audit of that corpus.

Public implementation snapshots preserve author attribution and MIT licenses.
GroupDRO is pinned to `cbbc1c5b06844e46b87e264326b56056d2a437d1` and
`rxn_yields` to `d9e6b87ce1b881978490d68bfc00021e3b48127a`. Their saved README,
license and GroupDRO loss files have individual hashes in
`third_party/METHOD_REFERENCE_MANIFEST.json`. RXNFP weights, tokenizer and
dependencies have their separate frozen manifests. These citations establish
the provenance of the components used in the controlled experiments.

## Statistical interpretation planned before evaluation

The primary estimate is the equally weighted mean of eligible-source AUROCs,
after computing each seed separately and averaging seed metrics. Bootstrap
units are substrate pairs within source, shared between all methods and seeds.
Seeds are optimization repeats, not independent chemical experiments. Two-way
aryl/amine pigeonhole resampling is a sensitivity analysis for shared-substrate
dependence. Intervals condition on the observed sources and fitted runs and do
not quantify all possible future laboratories or all training-data uncertainty.

Macro AUROC gain of 0.02, a supporting paired interval, and at most 0.03 source
regression are project acceptance rules rather than universal significance or
publication thresholds. Any combined module must also beat the strongest
single module and the balanced-ERM-plus-SWA control. Calibration is auxiliary:
monotonic recalibration cannot establish improved ranking. Measured-condition
top-k analysis reports coverage and cannot validate unmeasured recommendations.

No manuscript has been written and no submission has been made. This startup
check is intentionally limited; it is not an exhaustive novelty search.
