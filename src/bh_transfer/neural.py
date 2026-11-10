"""Controlled RXNFP ERM, source-balanced ERM, GroupDRO and trajectory SWA.

Training and validation IDs must be supplied explicitly. This module neither
discovers nor evaluates outer test partitions. Metadata source is used only for
training sampling/weights and is never passed to the predictive network.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import threading
from contextlib import contextmanager
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel
from transformers import BertConfig, BertModel

from .tokenizer import StrictSmilesTokenizer, TokenizationError, DATIVE_TOKENS

RXNFP_COMMIT = "6fd48f4927c2178555cc5d71dbfb225fb178f43c"
RXNFP_EXPECTED = {
    "pytorch_model.bin": "50a6ed263d33ae759affa82c1e85554cc5ea9f56145f5c7fb39b6c25d4356437",
    "config.json": "024d89cb6056e146e066ebbcd9d565d6968d53e82953570a8b110b26495d7d4b",
    "vocab.txt": "399868a653e85549ce150af4a1f6cd2776240c6a3d951ced2120565f00f027fd",
}


def _verify_initialization(project):
    """Require the exact audited public MLM initialization; no runtime bypass."""
    project = Path(project)
    manifest_path = project / "weights/rxnfp/manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Audited bert_pretrained manifest is required")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("model") != "bert_pretrained" or manifest.get("commit") != RXNFP_COMMIT
            or manifest.get("repository") != "https://github.com/rxn4chemistry/rxnfp"):
        raise ValueError("Only the fixed audited bert_pretrained initialization is permitted")
    entries = {item["path"]: item for item in manifest["files"]}
    if len(entries) != len(manifest["files"]):
        raise ValueError("Initialization manifest has duplicate file entries")
    observed = []
    for name, expected in RXNFP_EXPECTED.items():
        relative = f"weights/rxnfp/{name}"
        item = entries.get(relative)
        if item is None:
            raise ValueError(f"Initialization manifest is missing {relative}")
        path = project / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        source = f"https://raw.githubusercontent.com/rxn4chemistry/rxnfp/{RXNFP_COMMIT}/rxnfp/models/transformers/bert_pretrained/{name}"
        if actual != expected or item.get("sha256") != actual or item.get("source") != source:
            raise ValueError(f"Audited initialization hash/source mismatch: {relative}")
        observed.append({"path": relative, "sha256": actual, "bytes": path.stat().st_size, "source": item["source"]})
    tokenizer = StrictSmilesTokenizer(project / "weights/rxnfp/vocab.txt")
    extension = tokenizer.extension_metadata()
    extension_path = project / "weights/rxnfp/tokenizer_extension_manifest.json"
    extension_bytes = extension_path.read_bytes()
    extended_bytes = (project / "weights/rxnfp/extended_vocab.txt").read_bytes()
    manifest_extension = manifest.get("tokenizer_extension", {})
    if (json.loads(extension_bytes) != extension or extended_bytes != tokenizer.extended_vocab_bytes()
            or manifest_extension.get("manifest_sha256") != hashlib.sha256(extension_bytes).hexdigest()
            or manifest_extension.get("extended_vocab_sha256") != extension["extended_vocab_sha256"]
            or manifest_extension.get("rule_sha256") != extension["rule_sha256"]):
        raise ValueError("Dative tokenizer extension does not match the frozen rule")
    return {"repository": manifest["repository"], "commit": manifest["commit"],
            "model": manifest["model"], "license": manifest.get("license"), "files": observed,
            "tokenizer_extension": extension,
            "tokenizer_extension_manifest_sha256": hashlib.sha256(extension_bytes).hexdigest()}


def _implementation_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("neural.py", "tokenizer.py")}


@contextmanager
def _defer_stop_signals():
    """Receive a stop request without corrupting an in-progress optimizer step."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Training must run in the main thread for safe signal handling")
    request = {"signum": None}
    def handler(signum, frame):
        request["signum"] = int(signum)
    previous = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        for s in previous:
            signal.signal(s, handler)
        yield request
    finally:
        for s, old in previous.items():
            signal.signal(s, old)


@dataclass
class TrainConfig:
    objective: str = "erm"
    seed: int = 1729
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 8
    max_steps: int | None = None
    batch_per_group: int = 8
    microbatch_size: int = 8
    swa_start_fractions: tuple = (0.5, 0.75)
    device: str = "cuda"
    dro_eta: float = 0.01
    dropout: float = 0.1
    grad_clip: float = 1.0
    eval_batch_size: int = 32
    checkpoint_every_steps: int = 100
    amp: bool = True
    ordinary_fraction: float | None = None

    @classmethod
    def parse(cls, config):
        if isinstance(config, cls):
            return config
        return cls(**dict(config))


class DescriptorTransform:
    """Train-only finite median imputation and population-standard scaling."""
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 21 or len(x) == 0:
            raise ValueError("Expected nonempty n x 21 descriptor array")
        self.median = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.
                                for c in x.T])
        clean = np.where(np.isfinite(x), x, self.median)
        self.mean = clean.mean(axis=0)
        self.scale = clean.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=np.float64)
        return ((np.where(np.isfinite(x), x, self.median) - self.mean) / self.scale).astype(np.float32)

    def state_dict(self):
        return {k: getattr(self, k).tolist() for k in ("median", "mean", "scale")}

    @classmethod
    def from_state_dict(cls, state):
        obj = cls()
        for k in ("median", "mean", "scale"):
            setattr(obj, k, np.asarray(state[k], dtype=np.float64))
        return obj


class ReactionClassifier(nn.Module):
    def __init__(self, weights_dir=None, *, encoder_config=None, dropout=0.1, pretrained=True):
        super().__init__()
        if pretrained:
            if weights_dir is None:
                raise ValueError("Explicit bert_pretrained directory required")
            self.encoder = BertModel.from_pretrained(str(weights_dir), local_files_only=True,
                                                    add_pooling_layer=False, use_safetensors=False)
            tokenizer = StrictSmilesTokenizer(Path(weights_dir) / "vocab.txt")
            if self.encoder.config.vocab_size != tokenizer.original_vocab_size:
                raise ValueError("Original encoder/tokenizer vocabulary size mismatch")
            if "-" not in tokenizer.vocab:
                raise ValueError("Original vocabulary lacks the dative initialization token '-'")
            minus_embedding = self.encoder.get_input_embeddings().weight[tokenizer.vocab["-"]].detach().clone()
            self.encoder.resize_token_embeddings(len(tokenizer.tokens), mean_resizing=False)
            with torch.no_grad():
                for token in DATIVE_TOKENS:
                    self.encoder.get_input_embeddings().weight[tokenizer.vocab[token]].copy_(minus_embedding)
        else:
            if encoder_config is None:
                raise ValueError("encoder_config required without pretrained weights")
            self.encoder = BertModel(BertConfig(**encoder_config), add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size + 21, 1)

    def forward(self, input_ids, attention_mask, descriptors):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(self.dropout(torch.cat((hidden, descriptors), dim=1))).squeeze(-1)


class GroupDRO:
    """Exponentiated group-weight ascent, with detached group risk estimates.

    Every effective training batch must contain every group. We reject a missing
    group instead of interpreting its absent risk as zero or renormalizing an
    arbitrary observed subset. Microbatches may omit groups.

    Method: Sagawa et al., ICLR 2020, arXiv:1911.08731. Reference implementation:
    kohpangwei/group_DRO at cbbc1c5b06844e46b87e264326b56056d2a437d1 (MIT).
    """
    def __init__(self, n_groups, eta=0.01, device="cpu"):
        if n_groups < 1 or eta < 0:
            raise ValueError("Invalid group count or eta")
        self.log_q = torch.full((n_groups,), -math.log(n_groups), device=device)
        self.eta = float(eta)

    @property
    def q(self):
        return self.log_q.softmax(0)

    @torch.no_grad()
    def update(self, losses, counts):
        if len(losses) != len(self.log_q) or torch.any(counts <= 0):
            raise ValueError("Every effective GroupDRO batch must contain every group")
        if not torch.isfinite(losses).all():
            raise ValueError("Non-finite group risk")
        self.log_q.add_(self.eta * losses.detach().to(self.log_q))
        self.log_q.sub_(torch.logsumexp(self.log_q, dim=0))
        return self.q.detach().clone()

    def state_dict(self):
        return {"log_q": self.log_q.detach().cpu(), "eta": self.eta}

    def load_state_dict(self, state):
        self.log_q.copy_(state["log_q"].to(self.log_q))
        self.eta = float(state["eta"])


class EffectiveBatchSampler:
    """Stateful source-balanced or row-uniform sampling with replacement."""
    def __init__(self, group_indices, objective, batch_per_group, seed):
        self.groups = [np.asarray(x, dtype=np.int64) for x in group_indices]
        if not self.groups or any(len(x) == 0 for x in self.groups):
            raise ValueError("Empty training source")
        self.objective = objective
        self.per_group = int(batch_per_group)
        self.size = self.per_group * len(self.groups)
        if self.per_group < 1:
            raise ValueError("batch_per_group must be positive")
        self.all_indices = np.concatenate(self.groups)
        self.rng = np.random.default_rng(seed)

    def next(self):
        if self.objective == "erm":
            return self.rng.choice(self.all_indices, self.size, replace=True)
        batch = np.concatenate([self.rng.choice(g, self.per_group, replace=True) for g in self.groups])
        self.rng.shuffle(batch)
        return batch

    def state_dict(self):
        return self.rng.bit_generator.state

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state


def learning_rate_multiplier(step, total_steps):
    """10% warmup, linear decay to 0.1 at 50%, then constant tail."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    progress = (step + 1) / total_steps
    if progress <= 0.1:
        return progress / 0.1
    if progress < 0.5:
        return 1.0 - 0.9 * (progress - 0.1) / 0.4
    return 0.1


def should_average_swa(completed_step, total_steps, steps_per_epoch, start_fraction):
    """Start immediately at ceil(f*T), then each ceil(epoch_steps/4), plus T."""
    start = math.ceil(float(start_fraction) * total_steps)
    interval = max(1, math.ceil(steps_per_epoch / 4))
    return completed_step >= start and ((completed_step - start) % interval == 0 or completed_step == total_steps)


def _load_data(project):
    project = Path(project)
    records = pd.read_parquet(project / "data/processed/records.parquet")
    required = {"row_id", "source", "y", "reaction_smiles"}
    if not required <= set(records):
        raise ValueError(f"Missing record columns: {required - set(records)}")
    if records.row_id.duplicated().any():
        raise ValueError("row_id must be unique")
    with np.load(project / "data/processed/features.npz", allow_pickle=False) as features:
        ids = features["row_ids"].astype(np.int64)
        descriptors = features["descriptors"].copy()
    if len(np.unique(ids)) != len(ids) or descriptors.shape != (len(ids), 21):
        raise ValueError("Feature IDs or descriptor dimensions invalid")
    positions = {int(x): i for i, x in enumerate(ids)}
    records = records.set_index("row_id", drop=False)
    try:
        descriptors = descriptors[[positions[int(i)] for i in records.index]]
    except KeyError as exc:
        raise ValueError(f"Missing features for row {exc}") from exc
    return records, descriptors


def _positions(records, ids, allow_empty=False):
    ids = np.asarray(ids, dtype=np.int64)
    if (len(ids) == 0 and not allow_empty) or len(np.unique(ids)) != len(ids):
        raise ValueError("IDs must be nonempty and unique")
    pos = records.index.get_indexer(ids)
    if (pos < 0).any():
        raise ValueError(f"Unknown row IDs: {ids[pos < 0].tolist()}")
    return pos


def _rng_state(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if str(device).startswith("cuda"):
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    if path.name == "last.pt" and path.exists():
        previous = path.with_name("last.previous.pt")
        previous_tmp = path.with_name("last.previous.pt.tmp")
        shutil.copy2(path, previous_tmp)
        os.replace(previous_tmp, previous)
    os.replace(tmp, path)


def _base_checkpoint(model, transform, config, train_ids, val_ids, identity, initialization, implementation):
    measured = {Path(item["path"]).name: item["sha256"] for item in initialization["files"]}
    return {"format_version": 1, "model_state": model.state_dict(),
            "encoder_config": model.encoder.config.to_dict(), "descriptor_transform": transform.state_dict(),
            "config": asdict(config), "train_ids": list(map(int, train_ids)),
            "val_ids": list(map(int, val_ids)), "identity": identity,
            "initialization": initialization, "implementation_sha256": implementation,
            "tokenizer_config": {"type": "StrictSmilesTokenizer", "max_length": 512,
                                 "do_lower_case": False, "truncation": False,
                                 "extension": initialization["tokenizer_extension"],
                                 "vocab_sha256": measured["vocab.txt"],
                                 "rxnfp_commit": initialization["commit"],
                                 "base_weight_sha256": measured.get("pytorch_model.bin", measured.get("model.safetensors"))}}


def _metrics(y, logits):
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
    if not len(y):
        return {"n": 0, "log_loss": None, "brier": None, "auroc": None, "auprc": None}
    probabilities = 1. / (1. + np.exp(-np.clip(np.asarray(logits, dtype=np.float64), -80, 80)))
    result = {"n": len(y), "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
              "brier": float(brier_score_loss(y, probabilities))}
    if len(np.unique(y)) == 2:
        result.update(auroc=float(roc_auc_score(y, probabilities)),
                      auprc=float(average_precision_score(y, probabilities)))
    else:
        result.update(auroc=None, auprc=None)
    return result


def _validation_metrics(y, logits, sources):
    result = _metrics(y, logits)
    sources = np.asarray(sources)
    group_results = {str(g): _metrics(np.asarray(y)[sources == g], np.asarray(logits)[sources == g])
                     for g in np.unique(sources)}
    valid = [x["auroc"] for x in group_results.values() if x["auroc"] is not None]
    result["pooled_auroc"] = result["auroc"]
    result["auroc"] = float(np.mean(valid)) if valid else None
    result["source_macro_auroc"] = result["auroc"]
    result["source_metrics"] = group_results
    return result


def _validation_score(metrics):
    return metrics["auroc"] if metrics["auroc"] is not None else -metrics["log_loss"]


@torch.no_grad()
def _predict(model, encoded, descriptors, tokenizer, device, batch_size):
    model.eval()
    output = []
    for start in range(0, len(encoded), batch_size):
        batch = tokenizer.batch_tensors(encoded[start:start + batch_size], device)
        desc = torch.as_tensor(descriptors[start:start + batch_size], dtype=torch.float32, device=device)
        output.append(model(**batch, descriptors=desc).float().cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float32)


def train_trajectory(project, train_ids, val_ids, config, output_dir):
    """Train/resume one objective and return internal-validation candidates.

    The caller owns resource scheduling and cost accounting. Checkpoints contain
    optimizer, group weights, both SWA states, sampler state and all RNG states.
    An existing trajectory resumes only if data/config identity is identical.
    SIGTERM/SIGINT are deferred until a complete optimizer/checkpoint boundary;
    the function then raises SystemExit(75), indicating a recoverable stop.
    """
    with _defer_stop_signals() as stop_request:
        return _train_trajectory_impl(project, train_ids, val_ids, config, output_dir, stop_request)


def _train_trajectory_impl(project, train_ids, val_ids, config, output_dir, stop_request):
    cfg = TrainConfig.parse(config)
    if cfg.objective not in {"erm", "bal", "dro"}:
        raise ValueError("objective must be erm, bal or dro")
    if cfg.epochs < 1 or cfg.microbatch_size < 1 or cfg.eval_batch_size < 1:
        raise ValueError("Invalid epoch or batch size")
    if any(not 0 < float(f) < 1 for f in cfg.swa_start_fractions):
        raise ValueError("SWA fractions must be strictly between zero and one")
    if set(map(int, train_ids)) & set(map(int, val_ids)):
        raise ValueError("Training/validation row overlap")
    project, output_dir = Path(project), Path(output_dir)
    initialization = _verify_initialization(project)
    implementation = _implementation_hashes()
    output_dir.mkdir(parents=True, exist_ok=True)
    records, raw_descriptors = _load_data(project)
    if not len(val_ids) and (cfg.ordinary_fraction is None or not 0 < cfg.ordinary_fraction <= 1):
        raise ValueError("No-validation final fit requires preselected ordinary_fraction in (0, 1]")
    train_pos, val_pos = _positions(records, train_ids), _positions(records, val_ids, allow_empty=True)
    train, val = records.iloc[train_pos], records.iloc[val_pos]
    y = train.y.to_numpy(dtype=np.float32)
    val_y = val.y.to_numpy(dtype=np.int64)
    if not set(np.unique(y)) <= {0, 1} or not set(np.unique(val_y)) <= {0, 1}:
        raise ValueError("Nonbinary labels")
    groups = sorted(train.source.astype(str).unique())
    group_lookup = {name: i for i, name in enumerate(groups)}
    group_ids = np.array([group_lookup[x] for x in train.source.astype(str)], dtype=np.int64)
    group_indices = [np.flatnonzero(group_ids == i) for i in range(len(groups))]
    tokenizer = StrictSmilesTokenizer(project / "weights/rxnfp/vocab.txt")
    def encode_rows(frame):
        result = []
        for row in frame.itertuples(index=False):
            try:
                result.append(tokenizer.encode(row.reaction_smiles))
            except TokenizationError as exc:
                raise TokenizationError(f"row_id={row.row_id}: {exc}") from exc
        return result
    train_encoded, val_encoded = encode_rows(train), encode_rows(val)
    transform = DescriptorTransform().fit(raw_descriptors[train_pos])
    train_desc, val_desc = transform.transform(raw_descriptors[train_pos]), transform.transform(raw_descriptors[val_pos])
    identity_data = {"config": asdict(cfg), "train_ids": list(map(int, train_ids)),
                     "val_ids": list(map(int, val_ids)), "groups": groups,
                     "train_text": train.reaction_smiles.tolist(), "val_text": val.reaction_smiles.tolist(),
                     "train_y": y.tolist(), "val_y": val_y.tolist(),
                     "descriptor_sha256": hashlib.sha256(raw_descriptors[np.r_[train_pos, val_pos]].tobytes()).hexdigest(),
                     "vocab_sha256": hashlib.sha256(Path(tokenizer.vocab_file).read_bytes()).hexdigest(),
                     "initialization": initialization, "implementation_sha256": implementation}
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
    model = ReactionClassifier(project / "weights/rxnfp", dropout=cfg.dropout).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = EffectiveBatchSampler(group_indices, cfg.objective, cfg.batch_per_group, cfg.seed)
    steps_per_epoch = math.ceil(len(train) / sampler.size)
    total_steps = cfg.epochs * steps_per_epoch
    if cfg.max_steps is not None:
        total_steps = min(total_steps, int(cfg.max_steps))
    if total_steps < 1:
        raise ValueError("No training steps requested")
    selected_step = (min(total_steps, max(1, round(cfg.ordinary_fraction * total_steps / steps_per_epoch)) * steps_per_epoch)
                     if cfg.ordinary_fraction is not None else None)
    dro = GroupDRO(len(groups), cfg.dro_eta, cfg.device)
    swa = {f"swa_{round(float(f) * 100):03d}": AveragedModel(model) for f in cfg.swa_start_fractions}
    swa_fractions = {f"swa_{round(float(f) * 100):03d}": float(f) for f in cfg.swa_start_fractions}
    swa_snapshot_steps = {name: [] for name in swa}
    if len(swa) != len(cfg.swa_start_fractions):
        raise ValueError("SWA start fractions produce duplicate names")
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.device.startswith("cuda"))
    history, group_history = [], []
    exposure = np.zeros(len(groups), dtype=np.int64)
    sample_exposure = np.zeros(len(train), dtype=np.int64)
    best_score, best_step, start_step, elapsed_prior = -float("inf"), 0, 0, 0.
    resume_path = output_dir / "last.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["identity"] != identity or state["total_steps"] != total_steps:
            raise ValueError("Existing trajectory has different data/config; use another output directory")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer"])
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if torch.is_tensor(value):
                    opt_state[key] = value.to(cfg.device)
        dro.load_state_dict(state["dro"])
        for name in swa:
            swa[name].load_state_dict(state["swa"][name])
        swa_snapshot_steps = state["swa_snapshot_steps"]
        scaler.load_state_dict(state["scaler"])
        sampler.load_state_dict(state["sampler"])
        exposure, sample_exposure = state["exposure"], state["sample_exposure"]
        start_step, best_score, best_step = state["completed_steps"], state["best_score"], state["best_step"]
        history, group_history = state["history"], state["group_history"]
        elapsed_prior = state["elapsed_seconds"]
        _restore_rng(state["rng"])
    t0 = time.monotonic()
    encoder_grad_verified = start_step > 0
    def save_resume(completed):
        checkpoint = _base_checkpoint(model, transform, cfg, train_ids, val_ids, identity, initialization, implementation)
        checkpoint.update(optimizer=optimizer.state_dict(), dro=dro.state_dict(),
                          swa={name: m.state_dict() for name, m in swa.items()},
                          swa_snapshot_steps=swa_snapshot_steps,
                          swa_interval_steps=max(1, math.ceil(steps_per_epoch / 4)),
                          scheduler={"kind": "warmup10_decay_to10_at50_constant_tail",
                                     "completed_steps": completed, "total_steps": total_steps,
                                     "base_lr": cfg.lr, "next_lr": cfg.lr * learning_rate_multiplier(completed, total_steps)},
                          scaler=scaler.state_dict(), sampler=sampler.state_dict(), rng=_rng_state(cfg.device),
                          completed_steps=completed, total_steps=total_steps, groups=groups,
                          exposure=exposure, sample_exposure=sample_exposure,
                          best_score=best_score, best_step=best_step, history=history, group_history=group_history,
                          stop_signal=stop_request["signum"],
                          elapsed_seconds=elapsed_prior + time.monotonic() - t0)
        _atomic_save(checkpoint, resume_path)
        pd.DataFrame(history).to_csv(output_dir / "validation_history.csv", index=False)
        pd.DataFrame(group_history).to_csv(output_dir / "group_history.csv", index=False)
    for step in range(start_step, total_steps):
        model.train()
        batch_idx = sampler.next()
        batch_groups = group_ids[batch_idx]
        counts_np = np.bincount(batch_groups, minlength=len(groups))
        if cfg.objective in {"bal", "dro"} and (counts_np == 0).any():
            raise RuntimeError("Balanced effective batch unexpectedly misses a source")
        lr = cfg.lr * learning_rate_multiplier(step, total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch_group_tensor = torch.as_tensor(batch_groups, device=cfg.device)
        counts = torch.as_tensor(counts_np, dtype=torch.float32, device=cfg.device)
        def micro_losses(idx):
            inputs = tokenizer.batch_tensors([train_encoded[i] for i in idx], cfg.device)
            desc = torch.as_tensor(train_desc[idx], device=cfg.device)
            labels = torch.as_tensor(y[idx], device=cfg.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=cfg.amp and cfg.device.startswith("cuda")):
                logits = model(**inputs, descriptors=desc)
                return F.binary_cross_entropy_with_logits(logits.float(), labels, reduction="none")
        # DRO needs current group risks before the weighted backward. A no-grad
        # pass followed by exact RNG replay preserves the same dropout masks and
        # allows genuine microbatch gradient accumulation (bounded activation RAM).
        if cfg.objective == "dro":
            before_pass_rng = _rng_state(cfg.device)
            with torch.no_grad():
                observed = torch.cat([micro_losses(batch_idx[lo:lo + cfg.microbatch_size])
                                      for lo in range(0, len(batch_idx), cfg.microbatch_size)])
                sums = torch.zeros(len(groups), device=cfg.device).scatter_add(0, batch_group_tensor, observed)
                group_losses = sums / counts
            weights = dro.update(group_losses, counts)
            _restore_rng(before_pass_rng)
        elif cfg.objective == "bal":
            weights = torch.full_like(counts, 1. / len(groups))
        else:
            weights = counts / counts.sum()
        sums = torch.zeros(len(groups), device=cfg.device)
        for lo in range(0, len(batch_idx), cfg.microbatch_size):
            idx = batch_idx[lo:lo + cfg.microbatch_size]
            micro_groups = batch_group_tensor[lo:lo + cfg.microbatch_size]
            losses = micro_losses(idx)
            loss = (losses * weights[micro_groups] / counts[micro_groups]).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}")
            scaler.scale(loss).backward()
            sums.scatter_add_(0, micro_groups, losses.detach())
        group_losses = sums / counts.clamp_min(1)
        scaler.unscale_(optimizer)
        if not encoder_grad_verified:
            norm = sum(float(p.grad.detach().float().norm()) for p in model.encoder.parameters() if p.grad is not None)
            if not math.isfinite(norm) or norm <= 0:
                raise RuntimeError("Encoder has no finite nonzero gradient; fine-tuning check failed")
            encoder_grad_verified = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        exposure += counts_np
        np.add.at(sample_exposure, batch_idx, 1)
        group_losses_np = group_losses.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        for gi, name in enumerate(groups):
            ex = sample_exposure[group_indices[gi]].astype(float)
            ess = float(ex.sum() ** 2 / np.square(ex).sum()) if ex.sum() else 0.
            group_history.append({"step": step + 1, "source": name,
                                  "group_loss": float(group_losses_np[gi]) if counts_np[gi] else None,
                                  "weight": float(weights_np[gi]), "batch_count": int(counts_np[gi]),
                                  "cumulative_exposure": int(exposure[gi]), "unique_exposed": int((ex > 0).sum()),
                                  "exposure_ess": ess, "lr": lr,
                                  "group_weight_ess": float(1. / np.square(weights_np).sum()),
                                  "effective_batch_ess": float(1. / np.sum(np.square(weights_np) / np.maximum(counts_np, 1)))})
        for name, avg in swa.items():
            if should_average_swa(step + 1, total_steps, steps_per_epoch, swa_fractions[name]):
                avg.update_parameters(model)
                swa_snapshot_steps[name].append(step + 1)
        boundary = (step + 1) % steps_per_epoch == 0 or step + 1 == total_steps
        if boundary:
            logits = _predict(model, val_encoded, val_desc, tokenizer, cfg.device, cfg.eval_batch_size)
            metrics = _validation_metrics(val_y, logits, val.source.to_numpy())
            score = _validation_score(metrics) if len(val) else None
            history.append({"step": step + 1, "candidate": "ordinary", **metrics})
            chosen = (step + 1 == selected_step) if selected_step is not None else (score > best_score)
            if chosen:
                best_score, best_step = (score if score is not None else 0.), step + 1
                checkpoint = _base_checkpoint(model, transform, cfg, train_ids, val_ids, identity, initialization, implementation)
                checkpoint.update(step=step + 1, val_metrics=metrics, candidate="ordinary_best")
                _atomic_save(checkpoint, output_dir / "best.pt")
            save_resume(step + 1)
        elif cfg.checkpoint_every_steps and (step + 1) % cfg.checkpoint_every_steps == 0:
            save_resume(step + 1)
        if stop_request["signum"] is not None:
            save_resume(step + 1)
            raise SystemExit(75)
    candidates = []
    best_path = output_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError("No ordinary checkpoint was produced")
    best_state = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model_state"])
    logits = _predict(model, val_encoded, val_desc, tokenizer, cfg.device, cfg.eval_batch_size)
    candidates.append({"name": "ordinary_best", "checkpoint": str(best_path), "step": best_step,
                       "val_ids": np.asarray(val_ids), "val_logits": logits,
                       "val_metrics": _validation_metrics(val_y, logits, val.source.to_numpy())})
    for name, avg in swa.items():
        if int(avg.n_averaged) == 0:
            raise RuntimeError(f"No snapshots for {name}")
        logits = _predict(avg.module, val_encoded, val_desc, tokenizer, cfg.device, cfg.eval_batch_size)
        metrics = _validation_metrics(val_y, logits, val.source.to_numpy())
        path = output_dir / f"{name}.pt"
        checkpoint = _base_checkpoint(avg.module, transform, cfg, train_ids, val_ids, identity, initialization, implementation)
        checkpoint.update(step=total_steps, candidate=name, val_metrics=metrics,
                          n_averaged=int(avg.n_averaged), swa_start_fraction=swa_fractions[name],
                          swa_snapshot_steps=swa_snapshot_steps[name],
                          swa_interval_steps=max(1, math.ceil(steps_per_epoch / 4)))
        _atomic_save(checkpoint, path)
        candidates.append({"name": name, "checkpoint": str(path), "step": total_steps,
                           "n_averaged": int(avg.n_averaged), "val_ids": np.asarray(val_ids),
                           "swa_snapshot_steps": swa_snapshot_steps[name],
                           "val_logits": logits, "val_metrics": metrics})
    return {"status": "complete", "total_steps": total_steps, "completed_steps": total_steps,
            "elapsed_seconds": elapsed_prior + time.monotonic() - t0,
            "encoder_grad_verified": encoder_grad_verified, "candidates": candidates,
            "ordinary_best_step": best_step,
            "history_path": str(output_dir / "validation_history.csv"),
            "group_history_path": str(output_dir / "group_history.csv"),
            "resume_checkpoint": str(resume_path)}


def predict_checkpoint(project, ids, checkpoint, batch_size=32, device=None):
    """Return logits in caller ID order; inference does not consume labels."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocab_path = Path(project) / "weights/rxnfp/vocab.txt"
    if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != state["tokenizer_config"]["vocab_sha256"]:
        raise ValueError("Checkpoint tokenizer vocabulary differs from project vocabulary")
    if state["implementation_sha256"] != _implementation_hashes():
        raise ValueError("Checkpoint was produced by different implementation source")
    cfg = TrainConfig.parse(state["config"])
    device = cfg.device if device is None else str(device)
    records, descriptors = _load_data(project)
    positions = _positions(records, ids)
    tokenizer = StrictSmilesTokenizer(Path(project) / "weights/rxnfp/vocab.txt")
    if tokenizer.extension_metadata() != state["tokenizer_config"]["extension"]:
        raise ValueError("Checkpoint dative tokenizer extension differs from current tokenizer")
    encoded = [tokenizer.encode(s) for s in records.iloc[positions].reaction_smiles]
    transform = DescriptorTransform.from_state_dict(state["descriptor_transform"])
    model = ReactionClassifier(encoder_config=state["encoder_config"], dropout=cfg.dropout, pretrained=False)
    model.load_state_dict(state["model_state"])
    model.to(device)
    return _predict(model, encoded, transform.transform(descriptors[positions]), tokenizer, device, batch_size)
