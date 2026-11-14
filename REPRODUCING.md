# Reproducing the experiment

The completed frozen experiment has scientific status `NOT_READY`. GroupDRO and
SWA did not meet the predefined increment criteria. The full result, including
limitations of the R1 substitute and the two sensitivities, is in
[the experimental report](reports/EXPERIMENT_REPORT.md).

## Get the code, data and evidence

```bash
git clone https://github.com/HuangShaoze2004/bh-source-transfer-benchmark.git
cd bh-source-transfer-benchmark
```

Git contains the original data, processed features, chemistry mappings, fixed
splits and locks, generic RXNFP weights, all saved predictions, run specifications,
logs, training diagnostics, ledger, statistics, figures and report history.
The complete neural and tree checkpoints are attached to the
[experiment release](https://github.com/HuangShaoze2004/bh-source-transfer-benchmark/releases/tag/experiment-2026-11-11).
The release manifest records each artifact's path, byte size and SHA-256 and
identifies its storage location.

## Create the environment

The original experiment used Linux, Python 3.12.3, one RTX 4090 and the exact
distributions in `requirements.lock.txt`. Metric reconstruction runs on CPU.
Training and neural prediction replay use CUDA. Linux or WSL2 provides the
resource-lock and CPU-affinity APIs used by the launcher.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
.venv/bin/python scripts/reproduce.py verify
```

The verification command restores recorded file permissions, including read-only
data and protocol locks. It checks the Git-resident scientific files against the
release manifest. Installed environments and package/download caches are
recreated from the dependency lock and recorded public sources. Literature
source-download caches are represented by their references and hashes.

## Rebuild all main tables from saved predictions

```bash
.venv/bin/python scripts/reproduce.py metrics
```

This checks complete prediction coverage, the frozen inputs and implementation,
labels, individual substrate novelty, nested split isolation, all per-source
and seed metrics, eligible-source macro averages, bootstrap completion and
registered statistical comparisons. It independently reconstructs all three
evaluation matrices, containing 814,671 method/seed/condition prediction rows.
New tables and a check record are written to `reproduction/`.

The portability entry point resolves the recorded original source-directory
prefix to this checkout when verifying sensitivity code hashes. The original
lock bytes and saved results retain their historical identities. The archived
commands in the run ledger show the original machine's paths.

## Restore every model and recovery state

Install the public `zstd` command-line tool on the new machine, then run

```bash
.venv/bin/python scripts/reproduce.py restore
.venv/bin/python scripts/reproduce.py verify --full
```

The restore command downloads the release archives, checks archive SHA-256,
checks every extracted file, and resumes at already verified archives after an
interruption. It restores all ordinary/SWA weights, final and previous recovery
checkpoints, optimizers, schedulers, group weights, sampler/random states and
tree estimators. The full snapshot needs approximately 92 GiB of model storage
plus 72 GiB for the compressed downloads. The manifest gives exact sizes.

To restore one archive, use its name from `release/MANIFEST.json`

```bash
.venv/bin/python scripts/reproduce.py restore --archive checkpoints-0001.tar.zst
```

These PyTorch and pickle files were produced by this experiment. Load the
verified release artifacts with the pinned environment.

## Replay a saved training or prediction job

```bash
.venv/bin/python scripts/reproduce.py replay --job formal_az_eln_2023_erm_s17
.venv/bin/python scripts/reproduce.py replay --job predict_az_eln_2023_B_s17 --replayed-checkpoint
```

Find every exact job ID in `runs/specs/` and the two sensitivity branches'
`runs/specs/` directories. Training replay uses the saved row IDs, seed, objective,
regularization, optimizer budget and checkpoint/SWA rules. Prediction replay
uses the restored original checkpoint by default. `--replayed-checkpoint` selects
the corresponding new training output.

Replay writes outputs under `reproduction/work/` and uses a separate guarded
replication ledger under `reproduction/runtime/runs/`. It serializes GPU work,
uses one CPU thread, and waits for foreign GPU workloads. The saved original
experiment ledger remains the record of the completed experiment. Multi-seed
results characterize optimization variability. The documented small numerical
divergence in one repeated AZ GroupDRO trajectory limits bitwise reproducibility.

The saved final configurations permit replication of the selected formal
comparisons. The archived inner specifications also permit individual development
job replay. The original selection code and its complete validation records are
included for inspection and reconstruction of the selection procedure.

## Licenses and attribution

Data, generic pretrained weights and third-party code retain their respective
licenses and attribution. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
The raw CSV SHA-256 is
`247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333`.
The main frozen protocol SHA-256 is
`b7005aff7fa71ab452015047b8b3a0d0aa0efa718fcdb937915d1267f6576a84`.

## Publication verification

`scripts/publish_github.py` prepares resumable, individually hashed checkpoint
archives below GitHub's 2 GiB asset limit, pushes the complete Git history and
uploads the assets to the experiment release. It accepts credentials through
the process environment or a private credential file outside this checkout.
Its final receipt records the remote commit and the verified release assets.
