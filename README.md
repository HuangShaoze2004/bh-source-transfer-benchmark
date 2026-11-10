# BH source robust transfer

Controlled experiments for `Yield >= 10` feasibility prediction on held-out sources whose aryl and amine substrates are each unseen. This repository contains experiments and an experiment report, not a manuscript or submission.

Read `USER_REQUEST.md`, `DATA_AUDIT.md`, `EXPERIMENT_PROTOCOL.md`, `DECISIONS.md`, `STATE.json` and `FINAL_EVAL_LOCK.json` before interpreting results. Source-wise macro AUROC is the primary metric. The source labels are available only for grouping/sampling; they are never predictive features.

## Environment and provenance

Use Python3.12 in a new local `.venv`. Exact distributions are in `requirements.lock.txt`; PyTorch is2.8.0+cu128. For a clean machine, use the official CUDA12.8 PyTorch wheel index in addition to PyPI when installing this lock. The current environment contains independent copies of the already installed PyTorch distribution/runtime and pinned packages installed from the public Tsinghua mirror; it does not inherit another project's site-packages. Bootstrap details are saved in `artifacts/environment_bootstrap.json`.

```bash
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python scripts/fetch_weights.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/prepare_data.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

Original pretrained weights/vocabulary are preserved and hashed. Two explicit dative-bond tokens extend the vocabulary without altering any original embedding; every neural model uses the same extension. Original QM provenance and target-product permission were not established, so primary inputs omit products and use clearly defined RDKit reagent descriptors. R1 is an explicitly named author-inspired substitute, not an exact production-model reconstruction.

## Execution and resumption

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/preflight.py
# Freeze only once after reviewing the checks and cost estimate:
.venv/bin/python scripts/freeze_protocol.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u scripts/pipeline.py --phase all
```

On this existing project, reuse the saved lock and resume the pipeline; do not regenerate splits, rerun preflight under changed code IDs, or overwrite the lock. The launcher checks running process identities, inherited resource locks, budget totals and artifact identities before starting a run. All failures and retries remain in `runs/ledger.jsonl`. A stopped trajectory resumes its optimizer, sampler, RNG, GroupDRO and SWA states. Different code/data/config requires a new, explicitly recorded revision/run identity.

Only after full prediction coverage is verified:

```bash
.venv/bin/python -m bh_transfer.evaluate --predictions artifacts/predictions.parquet --lock FINAL_EVAL_LOCK.json --out reports/final
```

CPU thread constraints apply to all commands, including BLAS and subprocesses. `scripts/run_guard.py` enforces four shared CPU slots and serial GPU access; use it for substantial computation. Do not wrap the coordinating pipeline itself in a resource lease that its child jobs also need.

Inspect live process identities, completed jobs, current epochs, cumulative costs and frozen artifact hashes without opening external predictions or metrics:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python scripts/monitor_progress.py
```

An absent process during an inspection is a reason to recheck the job status and logs, not to start another copy automatically.

## Output interpretation

All seeds are evaluated separately before averaging metrics; no seed ensemble is reported as a single model. Bootstrap samples substrate pairs, not individual condition rows or seeds. Shared-substrate sensitivity is required because even thousands of pairs may share very few amines. Top-k uses measured alternatives only. Negative outcomes, incomplete source coverage, baseline reconstruction limitations and heterogeneous yield assays must remain visible.

Primary data/code attribution: Neves et al., *Robust out-of-distribution prediction of Buchwald–Hartwig reactions*, DOI10.1038/s43588-026-01017-6; repository code Apache-2.0, Zenodo data CC-BY-4.0. RXNFP and GroupDRO are existing methods; original licenses and startup literature differences are retained in `third_party/` and `reports/`.
