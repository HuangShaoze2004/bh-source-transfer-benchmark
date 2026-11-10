"""CPU-only behavioral checks for strict inputs, DRO and restart semantics."""
import json
import hashlib
import os
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel

from bh_transfer.neural import (
    DescriptorTransform, EffectiveBatchSampler, GroupDRO, ReactionClassifier,
    learning_rate_multiplier, predict_checkpoint, train_trajectory, should_average_swa,
)
import bh_transfer.neural as neural
from bh_transfer.tokenizer import StrictSmilesTokenizer, TokenizationError

torch.set_num_threads(1)


@pytest.fixture
def synthetic_project(tmp_path):
    """Random tiny BERT is solely a test fixture, never an experiment model."""
    weights = tmp_path / "weights/rxnfp"
    weights.mkdir(parents=True)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "C", "N", "O", ".", ">>", "-"]
    (weights / "vocab.txt").write_text("\n".join(vocab) + "\n")
    config = BertConfig(vocab_size=len(vocab), hidden_size=8, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=16, max_position_embeddings=512)
    torch.manual_seed(77)
    BertModel(config).save_pretrained(weights, safe_serialization=False)
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    pd.DataFrame({"row_id": np.arange(12), "source": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                  "y": [0, 1] * 6, "reaction_smiles": ["CC.N>>CCN", "C.O>>CO"] * 6}).to_parquet(processed / "records.parquet")
    descriptors = np.arange(12 * 21, dtype=float).reshape(12, 21)
    descriptors[0, 0] = np.nan
    np.savez(processed / "features.npz", row_ids=np.arange(12), descriptors=descriptors)
    return tmp_path


@pytest.fixture
def mocked_test_initialization(synthetic_project, monkeypatch):
    """Explicit private-function mock; production has no bypass option or flag."""
    def measured_fixture(project):
        root = Path(project)
        weights = root / "weights/rxnfp"
        return {"model": "TEST_ONLY_RANDOM_TINY_BERT", "commit": "TEST_FIXTURE",
                "repository": "TEST_FIXTURE", "license": None, "test_only_fixture": True,
                "tokenizer_extension": StrictSmilesTokenizer(weights / "vocab.txt").extension_metadata(),
                "files": [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                           "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "source": "pytest_fixture"}
                          for p in sorted(weights.iterdir()) if p.is_file()]}
    monkeypatch.setattr(neural, "_verify_initialization", measured_fixture)
    return synthetic_project


def test_real_training_requires_exact_audited_manifest(tmp_path):
    with pytest.raises(ValueError, match="manifest is required"):
        neural._verify_initialization(tmp_path)
    project = Path(__file__).resolve().parents[1]
    observed = neural._verify_initialization(project)
    assert observed["model"] == "bert_pretrained"
    for item in observed["files"]:
        assert item["sha256"] == neural.RXNFP_EXPECTED[Path(item["path"]).name]
    weights = tmp_path / "weights/rxnfp"
    weights.mkdir(parents=True)
    manifest = json.loads((project / "weights/rxnfp/manifest.json").read_text())
    (weights / "manifest.json").write_text(json.dumps(manifest))
    (weights / "pytorch_model.bin").write_bytes(b"unapproved weights")
    with pytest.raises(ValueError, match="hash/source mismatch"):
        neural._verify_initialization(tmp_path)


def test_tokenizer_never_silently_drops_information(synthetic_project):
    tok = StrictSmilesTokenizer(synthetic_project / "weights/rxnfp/vocab.txt", max_length=10)
    assert tok.encode("C.N>>CN") == [2, 5, 8, 6, 9, 5, 6, 3]
    with pytest.raises(TokenizationError, match="unmatched"):
        tok.encode("C N")
    with pytest.raises(TokenizationError, match="Unknown"):
        tok.encode("[Pd]")
    with pytest.raises(TokenizationError, match="truncation"):
        tok.encode("C" * 9)
    assert tok.audit(["C", "[Pd]"], [31, 32])["failures"][0]["row_id"] == 32


def test_dative_tokenizer_preserves_original_ids_and_roundtrip():
    project = Path(__file__).resolve().parents[1]
    vocab_path = project / "weights/rxnfp/vocab.txt"
    original = vocab_path.read_text().splitlines()
    tok = StrictSmilesTokenizer(vocab_path)
    assert len(original) == 591 and len(tok.tokens) == 593
    assert tok.version == "rxnfp_dative_v1"
    assert all(tok.vocab[token] == i for i, token in enumerate(original))
    assert tok.vocab["->"] == 591 and tok.vocab["<-"] == 592
    text = "N->[Pd]<-N.C>>"
    assert "".join(tok.tokenize(text)) == text
    assert tok.tokenize(text) == ["N", "->", "[Pd]", "<-", "N", ".", "C", ">>"]
    assert tok.extended_vocab_bytes() == (project / "weights/rxnfp/extended_vocab.txt").read_bytes()
    assert tok.extension_metadata() == json.loads((project / "weights/rxnfp/tokenizer_extension_manifest.json").read_text())


def test_audited_pretrained_extension_preserves_591_original_embeddings():
    project = Path(__file__).resolve().parents[1]
    weights = project / "weights/rxnfp"
    original = BertModel.from_pretrained(weights, local_files_only=True, add_pooling_layer=False,
                                        use_safetensors=False).get_input_embeddings().weight.detach().clone()
    model = ReactionClassifier(weights)
    extended = model.encoder.get_input_embeddings().weight.detach()
    tok = StrictSmilesTokenizer(weights / "vocab.txt")
    assert model.encoder.config.vocab_size == 593
    assert extended.shape == (593, 256)
    assert torch.equal(extended[:591], original)
    for arrow in ("->", "<-"):
        assert torch.equal(extended[tok.vocab[arrow]], original[tok.vocab["-"]])


def test_imputer_fits_training_only_and_preserves_finite_values():
    x = np.tile(np.array([[1.], [3.], [np.nan]]), (1, 21))
    x[:, 1] = np.nan
    transformer = DescriptorTransform().fit(x)
    assert transformer.median[0] == 2
    assert transformer.median[1] == 0
    transformed = transformer.transform(np.full((1, 21), 1000.))
    assert transformed[0, 0] > 100
    restored = DescriptorTransform.from_state_dict(transformer.state_dict())
    np.testing.assert_equal(restored.transform(x), transformer.transform(x))
    assert np.isfinite(transformer.transform(x)).all()


def test_dro_hand_calculation_detach_and_missing_groups():
    dro = GroupDRO(2, eta=1.)
    losses = torch.tensor([0.2, 0.8], requires_grad=True)
    weights = dro.update(losses, torch.tensor([2., 2.]))
    np.testing.assert_allclose(weights.numpy(), [0.35434369, 0.64565631], rtol=1e-6)
    robust = (weights * losses).sum()
    robust.backward()
    np.testing.assert_allclose(losses.grad.numpy(), weights.numpy())
    assert not weights.requires_grad
    assert float(robust.detach()) == pytest.approx(0.58739379)
    before = dro.q.clone()
    with pytest.raises(ValueError, match="every group"):
        dro.update(torch.tensor([100., 0.]), torch.tensor([1., 0.]))
    torch.testing.assert_close(dro.q, before)
    dro.update(torch.tensor([100000., 100001.]), torch.ones(2))
    assert torch.isfinite(dro.q).all()
    assert float(dro.q.sum()) == pytest.approx(1.)


def test_balanced_sampler_all_sources_and_exact_restart():
    sampler = EffectiveBatchSampler([np.arange(2), np.arange(2, 5)], "bal", 3, 9)
    batch = sampler.next()
    assert sum(batch < 2) == sum(batch >= 2) == 3
    state = sampler.state_dict()
    expected = sampler.next()
    restored = EffectiveBatchSampler([np.arange(2), np.arange(2, 5)], "dro", 3, 0)
    restored.load_state_dict(state)
    np.testing.assert_equal(expected, restored.next())


def test_lr_warmup_decay_and_matched_tail():
    assert learning_rate_multiplier(0, 100) == pytest.approx(.1)
    assert learning_rate_multiplier(9, 100) == pytest.approx(1.)
    assert learning_rate_multiplier(49, 100) == pytest.approx(.1)
    assert learning_rate_multiplier(99, 100) == pytest.approx(.1)


def test_swa_quarter_epoch_cadence_averages_multiple_snapshots_numerically():
    model = torch.nn.Linear(1, 1, bias=False)
    averaged = torch.optim.swa_utils.AveragedModel(model)
    chosen = []
    for step in range(1, 13):  # Three epochs, four updates per epoch.
        with torch.no_grad():
            model.weight.fill_(step)
        if should_average_swa(step, total_steps=12, steps_per_epoch=4, start_fraction=.75):
            chosen.append(step)
            averaged.update_parameters(model)
    assert chosen == [9, 10, 11, 12]
    assert int(averaged.n_averaged) == 4
    assert averaged.module.weight.item() == pytest.approx(np.mean(chosen))
    assert should_average_swa(15, 15, 5, .75)  # Final step included off cadence.


def test_encoder_receives_gradient(synthetic_project):
    model = ReactionClassifier(synthetic_project / "weights/rxnfp")
    model.train()
    logits = model(torch.tensor([[2, 5, 3]]), torch.ones(1, 3, dtype=torch.long), torch.zeros(1, 21))
    logits.sum().backward()
    assert sum(p.grad.abs().sum().item() for p in model.encoder.parameters() if p.grad is not None) > 0


def _config(**updates):
    return {"objective": "dro", "device": "cpu", "seed": 10, "epochs": 2,
            "batch_per_group": 2, "microbatch_size": 2, "eval_batch_size": 2,
            "checkpoint_every_steps": 1, "amp": False, **updates}


def test_training_resume_reproduces_trajectory(mocked_test_initialization):
    project = mocked_test_initialization
    train_ids, val_ids = list(range(8)), list(range(8, 12))
    output = project / "run"
    full = train_trajectory(project, train_ids, val_ids, _config(), output)
    assert full["encoder_grad_verified"]
    assert full["completed_steps"] == 4
    assert {x["name"] for x in full["candidates"]} == {"ordinary_best", "swa_050", "swa_075"}
    assert next(x for x in full["candidates"] if x["name"] == "swa_050")["n_averaged"] == 3
    assert next(x for x in full["candidates"] if x["name"] == "swa_075")["n_averaged"] == 2
    expected = {x["name"]: x["val_logits"].copy() for x in full["candidates"]}
    previous = torch.load(output / "last.previous.pt", weights_only=False)
    assert previous["completed_steps"] == 3
    torch.save(previous, output / "last.pt")
    resumed = train_trajectory(project, train_ids, val_ids, _config(), output)
    for candidate in resumed["candidates"]:
        np.testing.assert_array_equal(candidate["val_logits"], expected[candidate["name"]])
        pred = predict_checkpoint(project, val_ids[::-1], candidate["checkpoint"], batch_size=2)
        np.testing.assert_array_equal(pred, candidate["val_logits"][::-1])
    with pytest.raises(ValueError, match="different data/config"):
        train_trajectory(project, train_ids, val_ids, _config(lr=1e-4), output)


def test_final_refit_uses_preselected_step_without_validation(mocked_test_initialization):
    synthetic_project = mocked_test_initialization
    with pytest.raises(ValueError, match="preselected"):
        train_trajectory(synthetic_project, list(range(8)), [], _config(), synthetic_project / "invalid")
    result = train_trajectory(synthetic_project, list(range(8)), [], _config(ordinary_fraction=.5),
                              synthetic_project / "final")
    assert result["ordinary_best_step"] == 2
    assert result["total_steps"] == 4
    assert all(len(c["val_logits"]) == 0 for c in result["candidates"])
    assert all(c["val_metrics"]["n"] == 0 for c in result["candidates"])


def test_source_change_rejects_resume_and_inference(mocked_test_initialization, monkeypatch):
    project = mocked_test_initialization
    result = train_trajectory(project, list(range(8)), list(range(8, 12)), _config(), project / "run")
    state = torch.load(project / "run/last.pt", weights_only=False)
    assert state["initialization"]["test_only_fixture"]
    assert state["tokenizer_config"]["rxnfp_commit"] == "TEST_FIXTURE"
    assert state["implementation_sha256"] == neural._implementation_hashes()
    changed = dict(neural._implementation_hashes(), **{"neural.py": "changed_source"})
    monkeypatch.setattr(neural, "_implementation_hashes", lambda: changed)
    with pytest.raises(ValueError, match="different data/config"):
        train_trajectory(project, list(range(8)), list(range(8, 12)), _config(), project / "run")
    with pytest.raises(ValueError, match="different implementation"):
        predict_checkpoint(project, [8, 9], result["candidates"][0]["checkpoint"])


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_signal_stops_at_complete_step_and_resumes_exactly(mocked_test_initialization, monkeypatch, stop_signal):
    project = mocked_test_initialization
    train_ids, val_ids = list(range(8)), list(range(8, 12))
    uninterrupted = train_trajectory(project, train_ids, val_ids, _config(), project / "reference")
    old_handler = signal.getsignal(stop_signal)
    old_step = torch.optim.AdamW.step
    sent = False
    def interrupt_after_update(self, *args, **kwargs):
        nonlocal sent
        value = old_step(self, *args, **kwargs)
        if not sent:
            sent = True
            os.kill(os.getpid(), stop_signal)
        return value
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", interrupt_after_update)
        with pytest.raises(SystemExit) as exc:
            train_trajectory(project, train_ids, val_ids, _config(), project / "interrupted")
        assert exc.value.code == 75
    assert signal.getsignal(stop_signal) == old_handler
    state = torch.load(project / "interrupted/last.pt", weights_only=False)
    assert state["completed_steps"] == 1
    assert state["stop_signal"] == stop_signal
    resumed = train_trajectory(project, train_ids, val_ids, _config(), project / "interrupted")
    for expected, actual in zip(uninterrupted["candidates"], resumed["candidates"]):
        assert expected["name"] == actual["name"]
        np.testing.assert_array_equal(expected["val_logits"], actual["val_logits"])
