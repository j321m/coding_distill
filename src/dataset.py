"""The pre-tokenized dataset: where it lives, what it is, and loading it."""


import json
import os
import re

from datasets import load_from_disk

from src.paths import expand_path

META_FILE = "meta.json"

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


def write_meta(path, params, rows):
    meta = {field: params.get(field) for field in IDENTITY_FIELDS}
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


def build_dataset(params):
    """Load a pre-tokenized artifact -- see scripts/tokenize_dataset.py."""
    path = artifact_path(params)
    assert_meta_matches(path, params)
    splits = load_from_disk(path)
    return splits["train"], splits["test"]
