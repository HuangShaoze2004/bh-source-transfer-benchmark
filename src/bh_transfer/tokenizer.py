"""Strict RXNFP tokenization, adapted from the MIT-licensed RXNFP tokenizer.

Original: rxn4chemistry/rxnfp, commit 6fd48f4927c2178555cc5d71dbfb225fb178f43c.
The unchanged source and license are retained in third_party/rxnfp/.
Unlike the upstream convenience wrapper, no unmatched text, unknown tokens,
or overlength reaction is silently accepted.
"""
from pathlib import Path
import hashlib
import json
import re

TOKENIZER_VERSION = "rxnfp_dative_v1"
DATIVE_TOKENS = ("->", "<-")
SMI_REGEX_PATTERN = r"(->|<-|\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"


class TokenizationError(ValueError):
    """The input cannot be represented losslessly by the fixed tokenizer."""


class StrictSmilesTokenizer:
    def __init__(self, vocab_file, max_length=512):
        self.version = TOKENIZER_VERSION
        self.vocab_file = str(vocab_file)
        self.original_tokens = Path(vocab_file).read_text().splitlines()
        if any(t in self.original_tokens for t in DATIVE_TOKENS):
            raise ValueError("Pass the unchanged original RXNFP vocab.txt; extension is automatic")
        self.original_vocab_size = len(self.original_tokens)
        self.tokens = self.original_tokens + list(DATIVE_TOKENS)
        if len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Duplicate tokens in vocabulary")
        self.vocab = {token: i for i, token in enumerate(self.tokens)}
        self.regex = re.compile(SMI_REGEX_PATTERN)
        self.max_length = int(max_length)
        for special in ("[PAD]", "[CLS]", "[SEP]", "[UNK]"):
            if special not in self.vocab:
                raise ValueError(f"Missing special token {special}")
        self.pad_token_id = self.vocab["[PAD]"]

    def extension_metadata(self):
        recipe = {"version": self.version, "added_tokens": list(DATIVE_TOKENS),
                  "added_token_ids": [self.vocab[t] for t in DATIVE_TOKENS],
                  "original_vocab_size": self.original_vocab_size, "extended_vocab_size": len(self.tokens),
                  "embedding_initialization": "copy_original_minus_embedding",
                  "regex_sha256": hashlib.sha256(SMI_REGEX_PATTERN.encode()).hexdigest()}
        return {**recipe,
                "rule_sha256": hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest(),
                "original_vocab_sha256": hashlib.sha256(Path(self.vocab_file).read_bytes()).hexdigest(),
                "extended_vocab_sha256": hashlib.sha256(self.extended_vocab_bytes()).hexdigest()}

    def extended_vocab_bytes(self):
        return ("\n".join(self.tokens) + "\n").encode()

    def tokenize(self, text):
        if not isinstance(text, str) or not text:
            raise TokenizationError("Empty or non-string reaction SMILES")
        tokens = self.regex.findall(text)
        if "".join(tokens) != text:
            raise TokenizationError("Regex leaves unmatched characters in reaction SMILES")
        unknown = sorted(set(tokens).difference(self.vocab))
        if unknown:
            raise TokenizationError(f"Unknown tokens: {unknown}")
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        n = len(tokens) + 2
        if n > self.max_length:
            raise TokenizationError(f"{n} tokens exceeds limit {self.max_length}; truncation is forbidden")
        return [self.vocab["[CLS]"]] + [self.vocab[t] for t in tokens] + [self.vocab["[SEP]"]]

    def batch_tensors(self, encoded, device="cpu"):
        import torch
        if not encoded:
            raise ValueError("Empty batch")
        width = max(map(len, encoded))
        ids = torch.full((len(encoded), width), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(encoded):
            ids[i, :len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, :len(row)] = 1
        return {"input_ids": ids.to(device), "attention_mask": mask.to(device)}

    def audit(self, texts, row_ids=None):
        texts = list(texts)
        row_ids = list(range(len(texts))) if row_ids is None else list(row_ids)
        if len(row_ids) != len(texts):
            raise ValueError("row_ids/texts length mismatch")
        lengths, failures = [], []
        for row_id, text in zip(row_ids, texts):
            try:
                lengths.append(len(self.encode(text)))
            except TokenizationError as exc:
                failures.append({"row_id": int(row_id), "reason": str(exc)})
        return {"tokenizer_version": self.version, "extension": self.extension_metadata(),
                "n_rows": len(texts), "n_valid": len(lengths), "n_invalid": len(failures),
                "max_tokens": max(lengths, default=0), "failures": failures}
