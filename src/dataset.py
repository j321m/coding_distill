"""The pre-tokenized dataset: where it lives, what it is, and loading it."""


import hashlib
import json
import os
import re

from datasets import load_from_disk

from src.paths import expand_path

META_FILE = "meta.json"
FINGERPRINT_FIELD = "tokenizer_fingerprint"

# what defines *which* artifact this is
IDENTITY_FIELDS = (
    "dataset",
    "config",
    "split",
    "tokenizer",
    "max_length",
    "test_size",
    "split_seed",
)


def _slug(value):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()


def tokenizer_fingerprint(tokenizer):
    parts = (
        tokenizer.backend_tokenizer.to_str(),  # vocab, merges, normalizer
        tokenizer.chat_template or "",
        json.dumps(tokenizer.special_tokens_map, sort_keys=True),
    )
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def artifact_name(params):
    """The single naming rule, used by the writer and the reader alike."""
    return "__".join(
        (
            _slug(params.config),
            _slug(params.tokenizer),
            str(params.max_length),
            *(("debug",) if params.get("filter_n") else ()),
        )
    )


def artifact_path(params):
    return os.path.join(expand_path(params.data_dir, "data_dir"), artifact_name(params))


def write_meta(path, params, rows, tokenizer=None):
    meta = {field: params.get(field) for field in IDENTITY_FIELDS}
    if tokenizer is not None:
        meta[FINGERPRINT_FIELD] = tokenizer_fingerprint(tokenizer)
    meta["rows"] = rows
    meta["filters"] = {
        "drop_finish_reason": params.get("drop_finish_reason"),
        "filter_max_tokens": params.get("filter_max_tokens"),
        "filter_n": params.get("filter_n"),
    }
    with open(os.path.join(path, META_FILE), "w") as file:
        json.dump(meta, file, indent=2)
    return meta


def assert_meta_matches(path, params):
    with open(os.path.join(path, META_FILE)) as file:
        meta = json.load(file)

    mismatched = {
        field: (meta.get(field), params.get(field))
        for field in IDENTITY_FIELDS
        if meta.get(field) != params.get(field)
    }
    if mismatched:
        raise SystemExit(
            f"tokenized dataset at {path} does not match this config: "
            + ", ".join(
                f"{k}: artifact={a!r} config={c!r}" for k, (a, c) in mismatched.items()
            )
        )
    return meta


def assert_tokenizer_matches(params, tokenizer):
    """Refuse to train on data built by a different tokenizer.
    """
    path = artifact_path(params)
    with open(os.path.join(path, META_FILE)) as file:
        recorded = json.load(file).get(FINGERPRINT_FIELD)
    if recorded is None:
        print(
            f"{path}/{META_FILE} predates tokenizer fingerprints; falling back to "
            f"comparing names ({params.get('tokenizer')}). Re-run the tokenize job "
            "to get the real check."
        )
        return

    actual = tokenizer_fingerprint(tokenizer)
    if actual != recorded:
        raise SystemExit(
            f"tokenizer does not match the data at {path}: artifact={recorded} "
            f"model={actual}. Vocabulary, chat template or special tokens differ, "
            "so the stored input_ids mean something else than this model reads."
        )


def build_dataset(params):
    """Load a pre-tokenized artifact -- see scripts/tokenize_dataset.py."""
    path = artifact_path(params)
    assert_meta_matches(path, params)
    splits = load_from_disk(path)
    return splits["train"], splits["test"]
