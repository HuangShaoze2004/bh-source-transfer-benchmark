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
| Original data archive | [Zenodo 19636649](https://zenodo.org/records/19636649), DOI 10.5281/zenodo.19636649, v1, published 2026-04-18; `2026-02-21_df_HTE_BH_QM_10iter_9-3clustertest.pkl`, approximately 1.4 GB, advertised MD5 `c088ab8b56c78ceb381910a05265791c`. | The human-readable page did not expose an interpretable license string in this audit. Do not infer its license from the paper. Verify record metadata/license separately before redistribution; hash actual downloaded bytes. |
| Author code | [schwallergroup/bh-hte-ood](https://github.com/schwallergroup/bh-hte-ood), inspected commit `3a44e398d9fed3f08cf396f1153be2960193f581`. | Repository advertises Apache-2.0. Preserve LICENSE and author attribution. Data-specific license coverage still needs its own record. |
| Ha author manuscript | PMCID PMC13047508, public since 2026-06-11, DOI above. | Public author manuscript access confirmed; do not assume that public access grants unrestricted redistribution. Store bibliographic metadata and original paraphrases, not copied full text without a verified license. |
| Saebi paper | DOI 10.1039/d2sc06041h, PMCID PMC10189898. | Original RSC text publicly accessible; this report uses short factual paraphrases. Full-text reuse license is not inferred here. |

The 2025 preprint, DOI `10.26434/chemrxiv-2025-xcr46`, is an earlier version of
the Neves article. Use the 2026 publication for substantive claims. The GitHub
CSV date and Zenodo pickle date differ; identity cannot be inferred from a
similar row count.

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
