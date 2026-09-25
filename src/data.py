from datasets import load_from_disk

from src.artifact import artifact_path, assert_meta_matches


def build_dataset(params):
    """Load a pre-tokenized artifact -- see scripts/tokenize_dataset.py."""
    path = artifact_path(params)
    assert_meta_matches(path, params)
    splits = load_from_disk(path)
    return splits["train"], splits["test"]
