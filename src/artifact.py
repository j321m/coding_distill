"""Identity of a pre-tokenized dataset: where it lives, and what it is.

Tokens from the wrong tokenizer train happily and give a plausible loss curve, so
the directory name encodes the identity and `meta.json` is asserted before use.
"""

import json
import os
import re

META_FILE = "meta.json"

# what defines *which* artifact this is; process knobs (num_proc, filters) do not
IDENTITY_FIELDS = ("dataset", "config", "split", "tokenizer", "max_length")


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
    data_dir = os.path.expandvars(params.data_dir)
    if "$" in data_dir:
        raise SystemExit(
            f"unresolved variable in data_dir: {params.data_dir!r} -> {data_dir!r}; "
            "it must be exported by prolog_cmd in clusters.yaml"
        )
    return os.path.join(data_dir, artifact_name(params))


def write_meta(path, params, num_rows):
    meta = {field: params.get(field) for field in IDENTITY_FIELDS}
    meta["num_rows"] = num_rows
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
