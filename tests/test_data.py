import numpy as np
import pandas as pd
from bh_transfer.data import canonical_smiles, entity_key, descriptor_vector, double_unseen, author_morgan_input_fingerprint, assert_split_integrity, make_splits, prepare_records, tokenizer_equivalent_smiles, standard_drfp_input_shingles, build_features, ROLES


def test_group_identity_keeps_input_chemistry():
    charged = canonical_smiles("C[NH3+].[Cl-]")
    assert "+" in charged and "[Cl-]" in charged
    assert entity_key(charged) == entity_key("CN")
    assert entity_key("C[C@H](N)C(=O)O") == entity_key("C[C@@H](N)C(=O)O")
    assert canonical_smiles("C[C@H](N)C(=O)O") != canonical_smiles("C[C@@H](N)C(=O)O")


def test_two_individually_unseen_entities():
    train = pd.DataFrame({"aryl_id": ["a"], "amine_id": ["b"]})
    candidate = pd.DataFrame({"aryl_id": ["a", "c", "c", "b"], "amine_id": ["d", "b", "d", "e"]})
    assert double_unseen(candidate, train).index.tolist() == [2]


def test_descriptor_and_fingerprint_invariance():
    np.testing.assert_allclose(descriptor_vector("CCO"), descriptor_vector("OCC"))
    np.testing.assert_array_equal(author_morgan_input_fingerprint("CCO.N"), author_morgan_input_fingerprint("N.OCC"))
    assert descriptor_vector("[Na+].[OH-]")[-1] == 0
    assert np.isfinite(descriptor_vector("CCO")).all()


def test_split_partition_leakage_assertion():
    rows = []
    for source in range(3):
        for i in range(40):
            rows.append({"row_id": len(rows), "source": str(source), "aryl_id": f"a{source}_{i}", "amine_id": f"b{source}_{i}",
                         "pair_id": f"p{source}_{i}", "input_key": f"x{source}_{i}", "y": i % 2})
    records = pd.DataFrame(rows)
    splits = make_splits(records)
    assert all(f["eligible"] for f in splits["folds"].values())
    assert all(len(f["inner"]) == 2 for f in splits["folds"].values())
    assert_split_integrity(records, splits)


def test_threshold_author_label_and_product_permission():
    base = {"Source": "test", "Aryl SMILES": "Brc1ccccc1", "Amine SMILES": "CN",
            "Catalyst SMILES": "", "Base SMILES": "[Na+].[OH-]", "Solvent SMILES": "CCO"}
    raw = pd.DataFrame([{**base, "Yield": 10, "labels": 0, "Product SMILES": "CNc1ccccc1"},
                        {**base, "Yield": 9.9, "labels": 1, "Product SMILES": "unverified product text"}])
    records, _, failures = prepare_records(raw)
    assert failures.empty
    assert records.y.tolist() == [1, 0]
    assert records.author_label.tolist() == [0, 1]
    assert records.reaction_smiles.nunique() == 1
    assert records.input_key.nunique() == 1
    assert records.reaction_smiles.iloc[0].endswith(">>")


def test_explicit_hydrogen_tokenization_does_not_change_charge():
    from pathlib import Path
    from rdkit import Chem
    from bh_transfer.tokenizer import StrictSmilesTokenizer
    tokenizer = StrictSmilesTokenizer(Path(__file__).resolve().parents[1] / "weights/rxnfp/vocab.txt")
    encoded, rule = tokenizer_equivalent_smiles("[NaH+]", tokenizer)
    tokenizer.encode(encoded)
    assert rule == "equivalent_explicit_h_removehs_verified"
    assert Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromSmiles(encoded))) == "[NaH+]"


def test_standard_drfp_graph_route_matches_package_and_keeps_dative():
    from drfp import DrfpEncoder
    smiles = "Brc1ccccc1.CN.[Na+].[OH-].CCO"
    direct = DrfpEncoder.fold(DrfpEncoder.hash(list(standard_drfp_input_shingles(smiles))), length=2048)[0]
    np.testing.assert_array_equal(direct, DrfpEncoder.encode(smiles + ">>")[0])
    # This contains a single '>' inside a molecule, not a reaction delimiter.
    assert len(standard_drfp_input_shingles("N->[Pd]")) > 0
    assert standard_drfp_input_shingles("N->[Pd]") != standard_drfp_input_shingles("N.[Pd]")


def test_dative_reaction_fingerprint_includes_every_input_role():
    from drfp import DrfpEncoder
    inputs = {"aryl": "Brc1ccccc1", "amine": "CN", "catalyst": "N->[Pd]", "base": "[Na+].[OH-]", "solvent": "CCO"}
    row = {role + "_smiles": smi for role, smi in inputs.items()}
    row.update(row_id=0, substrates_smiles=inputs["aryl"] + "." + inputs["amine"], reaction_smiles=".not-used.>>")
    fingerprint = build_features(pd.DataFrame([row]))["drfp"][0]
    full = set().union(*(standard_drfp_input_shingles(smi) for smi in inputs.values()))
    np.testing.assert_array_equal(fingerprint, DrfpEncoder.fold(DrfpEncoder.hash(list(full)), length=2048)[0])
    for omitted in ROLES:
        reduced = set().union(*(standard_drfp_input_shingles(smi) for role, smi in inputs.items() if role != omitted))
        assert not np.array_equal(fingerprint, DrfpEncoder.fold(DrfpEncoder.hash(list(reduced)), length=2048)[0]), omitted
