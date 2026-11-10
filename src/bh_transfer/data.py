"""Auditable, outcome-independent feature preparation and source/entity splits."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

COMMIT = "3a44e398d9fed3f08cf396f1153be2960193f581"
CSV_NAME = "BH_HTE_Curated_Dataset_v20260219.csv"
CSV_SHA256 = "247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333"
EXCLUDED_SOURCE = "Dreher Doyle 2018"
ROLES = ("aryl", "amine", "catalyst", "base", "solvent")
RAW_COLUMNS = {role: role.capitalize() + " SMILES" for role in ROLES}
DESCRIPTOR_NAMES = ("MolWt", "MolLogP", "TPSA", "HBD", "HBA", "RotatableBonds", "FormalCharge")
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
_TAUTOMERS = rdMolStandardize.TautomerEnumerator()
_UNCHARGER = rdMolStandardize.Uncharger()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(*values: str) -> str:
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=20000)
def canonical_smiles(value: str, *, allow_empty: bool = False) -> str:
    """Keep charge, salts, isotopes, stereo and coordination in model inputs."""
    value = str(value).strip()
    if value.lower() in {"", "empty", "none", "nan"}:
        if allow_empty:
            return ""
        raise ValueError("missing required substrate")
    mol = Chem.MolFromSmiles(value)
    if mol is None:
        raise ValueError("SMILES cannot be parsed")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=20000)
def entity_key(smiles: str) -> str:
    """Conservative leakage key only: parent/tautomer/charge/stereo equivalence.

    This never replaces the molecular input. Keeping all largest organic parents
    in a tie avoids arbitrarily selecting a counterion with identical size.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid substrate for entity key")
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    organic = [m for m in fragments if any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
    candidates = organic or list(fragments)
    largest = max(m.GetNumHeavyAtoms() for m in candidates)
    parents = []
    for fragment in candidates:
        if fragment.GetNumHeavyAtoms() != largest:
            continue
        fragment = _UNCHARGER.uncharge(fragment)
        fragment = _TAUTOMERS.Canonicalize(fragment)
        Chem.RemoveStereochemistry(fragment)
        for atom in fragment.GetAtoms():
            atom.SetAtomMapNum(0)
            atom.SetIsotope(0)
        parents.append(Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=False))
    return stable_key(*sorted(set(parents)))


@lru_cache(maxsize=10000)
def descriptor_vector(smiles: str) -> np.ndarray:
    if not smiles:
        return np.zeros(7, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("invalid reagent")
    return np.array([Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
                     rdMolDescriptors.CalcTPSA(mol), Lipinski.NumHDonors(mol),
                     Lipinski.NumHAcceptors(mol), Lipinski.NumRotatableBonds(mol),
                     sum(a.GetFormalCharge() for a in mol.GetAtoms())], dtype=np.float32)


@lru_cache(maxsize=10000)
def author_morgan_input_fingerprint(smiles: str) -> np.ndarray:
    """Author OR/XOR representation with empty product side (declared substitute)."""
    result = np.zeros(2048, dtype=np.uint8)
    for component in smiles.split("."):
        if component:
            mol = Chem.MolFromSmiles(component)
            if mol is None:
                raise ValueError("invalid Morgan component")
            result |= np.asarray(_MORGAN.GetFingerprint(mol), dtype=np.uint8)
    return result


def tokenizer_equivalent_smiles(smiles: str, tokenizer) -> tuple[str, str]:
    """Choose a lossless, vocab-compatible rendering; never alter a molecule.

    Canonical isomeric graph equality includes stereo, formal charge and bond
    type/direction. Explicit H alternatives must also survive RemoveHs equality.
    """
    if not smiles:
        return "", "known_empty"
    try:
        tokenizer.encode(smiles)
        return smiles, "canonical"
    except ValueError:
        pass
    original = Chem.MolFromSmiles(smiles)
    signature = Chem.MolToSmiles(original, canonical=True, isomericSmiles=True)

    def allowed(candidate: str, explicit_h: bool = False) -> bool:
        try:
            tokenizer.encode(candidate)
            parsed = Chem.MolFromSmiles(candidate)
            if parsed is None:
                return False
            if explicit_h:
                parsed = Chem.RemoveHs(parsed)
            return Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True) == signature
        except (ValueError, RuntimeError):
            return False

    for atom_index in range(original.GetNumAtoms()):
        candidate = Chem.MolToSmiles(original, rootedAtAtom=atom_index, canonical=False, isomericSmiles=True)
        if allowed(candidate):
            return candidate, f"equivalent_rooted_at_{atom_index}"
    # The fixed seed/candidate limit are part of the data version, not tuned on outcomes.
    for candidate in Chem.MolToRandomSmilesVect(original, 256, randomSeed=1729, isomericSmiles=True):
        if allowed(candidate):
            return candidate, "equivalent_random_seed1729"
    expanded = Chem.AddHs(original)
    candidate = Chem.MolToSmiles(expanded, canonical=True, isomericSmiles=True)
    if allowed(candidate, explicit_h=True):
        return candidate, "equivalent_explicit_h_removehs_verified"
    raise ValueError("no graph-identical tokenizable SMILES found using fixed rules")


def apply_tokenizer_serialization(records: pd.DataFrame, tokenizer) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache, transforms = {}, []
    result = records.copy()
    for role in ROLES:
        for smiles in records[role + "_smiles"].unique():
            if smiles not in cache:
                try:
                    serialized, rule = tokenizer_equivalent_smiles(smiles, tokenizer)
                    cache[smiles] = (serialized, rule, "")
                except Exception as exc:
                    # Keep the failure visible for the caller's common-row disposition.
                    cache[smiles] = (smiles, "unresolved", str(exc))
        for row in records[["row_id", role + "_smiles"]].itertuples(index=False, name=None):
            serialized, rule, error = cache[row[1]]
            transforms.append({"row_id": row[0], "role": role, "canonical_smiles": row[1],
                               "serialized_smiles": serialized, "rule": rule, "error": error,
                               "graph_equivalence_verified": not bool(error)})
    result["reaction_smiles"] = [".".join(cache[getattr(row, role + "_smiles")][0] for role in ROLES
                                          if getattr(row, role + "_smiles")) + ">>" for row in records.itertuples()]
    return result, pd.DataFrame(transforms)


def summarize_rows(frame: pd.DataFrame) -> dict:
    return {"rows": len(frame), "pairs": int(frame.pair_id.nunique()),
            "positive_rows": int(frame.y.sum()), "negative_rows": int((frame.y == 0).sum()),
            "positive_pairs": int(frame.loc[frame.y == 1, "pair_id"].nunique()),
            "negative_pairs": int(frame.loc[frame.y == 0, "pair_id"].nunique()),
            "aryls": int(frame.aryl_id.nunique()), "amines": int(frame.amine_id.nunique()),
            "positive_rate": float(frame.y.mean()) if len(frame) else None}


def double_unseen(candidate: pd.DataFrame, development: pd.DataFrame) -> pd.DataFrame:
    # Union across roles additionally prevents a dual-role compound crossing a boundary.
    seen = set(development.aryl_id) | set(development.amine_id)
    return candidate.loc[~candidate.aryl_id.isin(seen) & ~candidate.amine_id.isin(seen)].copy()


def make_splits(records: pd.DataFrame) -> dict:
    folds = {}
    for source in sorted(records.source.unique()):
        development = records.loc[records.source != source]
        test = double_unseen(records.loc[records.source == source], development)
        summary = summarize_rows(test)
        eligible = (summary["pairs"] >= 30 and summary["positive_pairs"] >= 10
                    and summary["negative_pairs"] >= 10 and summary["positive_rows"] >= 20
                    and summary["negative_rows"] >= 20)
        inner = []
        for validation_source in sorted(development.source.unique()):
            train = development.loc[development.source != validation_source]
            val = double_unseen(development.loc[development.source == validation_source], train)
            details = summarize_rows(val)
            if details["pairs"] >= 10 and details["positive_pairs"] > 0 and details["negative_pairs"] > 0:
                inner.append({"source": validation_source, "train_ids": train.row_id.tolist(),
                              "val_ids": val.row_id.tolist(), "summary": details})
        inner.sort(key=lambda part: (-part["summary"]["pairs"], part["source"]))
        slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
        folds[slug] = {"source": source, "train_ids": development.row_id.tolist(),
                       "test_ids": test.row_id.tolist(), "eligible": eligible,
                       "test_summary": summary, "inner": inner[:2],
                       "inner_valid": bool(inner), "inner_failure_reason": None if inner else "no source with >=10 double-unseen pairs and both labels"}
    return {"schema_version": 1, "entity_rule": "role-union, largest organic fragment, uncharged canonical tautomer, no stereo/isotope; inputs retain original chemical forms",
            "eligibility": {"min_pairs": 30, "min_pairs_each_class": 10, "min_rows_each_class": 20},
            "inner_rule": "at most two valid held-out sources ranked by double-unseen pair count; >=10 pairs and both labels",
            "folds": folds}


def assert_split_integrity(records: pd.DataFrame, splits: dict) -> None:
    indexed = records.set_index("row_id", drop=False)
    for fold in splits["folds"].values():
        partitions = [(fold["train_ids"], fold["test_ids"])]
        partitions.extend((p["train_ids"], p["val_ids"]) for p in fold["inner"])
        for left, right in partitions:
            assert not set(left) & set(right)
            a, b = indexed.loc[left], indexed.loc[right]
            assert not (set(a.aryl_id) | set(a.amine_id)) & (set(b.aryl_id) | set(b.amine_id))
            assert not set(a.pair_id) & set(b.pair_id)
            assert not set(a.input_key) & set(b.input_key)


def prepare_records(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records, mapping, failures = [], [], []
    for row_id, row in raw.iterrows():
        data = {"row_id": int(row_id), "source": row["Source"], "y": int(float(row["Yield"]) >= 10),
                "author_label": int(row["labels"]), "yield_raw": float(row["Yield"]),
                "product_smiles": row["Product SMILES"]}
        if data["source"] == EXCLUDED_SOURCE:
            failures.append({"row_id": row_id, "reason": "unresolved_additive_identity", "detail": EXCLUDED_SOURCE})
            continue
        try:
            for role in ROLES:
                data[role + "_smiles"] = canonical_smiles(row[RAW_COLUMNS[role]], allow_empty=role not in {"aryl", "amine"})
            data["aryl_id"] = entity_key(data["aryl_smiles"])
            data["amine_id"] = entity_key(data["amine_smiles"])
            data["pair_id"] = stable_key(data["aryl_id"], data["amine_id"])
            data["input_key"] = stable_key(*(data[r + "_smiles"] for r in ROLES))
            data["substrates_smiles"] = data["aryl_smiles"] + "." + data["amine_smiles"]
            data["reaction_smiles"] = ".".join(data[r + "_smiles"] for r in ROLES if data[r + "_smiles"]) + ">>"
            for role in ROLES:
                mapping.append({"row_id": row_id, "role": role, "raw_smiles": row[RAW_COLUMNS[role]],
                                "canonical_smiles": data[role + "_smiles"], "entity_id": data.get(role + "_id", "")})
            records.append(data)
        except Exception as exc:
            failures.append({"row_id": row_id, "reason": "structure_parse_or_standardization_failed", "detail": str(exc)})
    return pd.DataFrame(records).sort_values("row_id"), pd.DataFrame(mapping), pd.DataFrame(failures)


def build_features(records: pd.DataFrame) -> dict:
    from drfp import DrfpEncoder
    # Never read product_smiles, author_label, source or yield to construct inputs.
    # The package reaction-string parser splits at every '>', including dative
    # arrows. Use its exact molecular shingling/hash/fold APIs to avoid silently
    # discarding catalyst atoms. Empty-product DRFP is the reactant shingle union.
    drfp_cache = {s: standard_drfp_input_shingles(s) for role in ROLES for s in records[role + "_smiles"].unique()}
    drfp = np.stack([DrfpEncoder.fold(DrfpEncoder.hash(list(set().union(
        *(drfp_cache[getattr(row, role + "_smiles")] for role in ROLES)))), length=2048)[0]
        for row in records.itertuples()]).astype(np.uint8)
    author = np.stack([author_morgan_input_fingerprint(s) for s in records.substrates_smiles])
    cache = {s: descriptor_vector(s) for role in ("catalyst", "base", "solvent") for s in records[role + "_smiles"].unique()}
    descriptors = np.stack([np.concatenate([cache[getattr(row, role + "_smiles")] for role in ("catalyst", "base", "solvent")])
                            for row in records.itertuples()])
    assert drfp.shape == author.shape == (len(records), 2048)
    assert descriptors.shape == (len(records), 21) and np.isfinite(descriptors).all()
    return {"row_ids": records.row_id.to_numpy(), "drfp": drfp, "author_substr": author, "descriptors": descriptors}


@lru_cache(maxsize=20000)
def standard_drfp_input_shingles(smiles: str) -> frozenset:
    from drfp import DrfpEncoder
    if not smiles:
        return frozenset()
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("invalid DRFP molecule")
    shingles = set()
    for fragment in Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True):
        shingles.update(DrfpEncoder.shingling_from_mol(fragment, radius=3, rings=True,
                         min_radius=0, root_central_atom=True, include_hydrogens=False))
    return frozenset(shingles)
