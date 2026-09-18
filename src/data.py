from datasets import load_from_disk

from src.artifact import artifact_path, assert_meta_matches


def build_dataset(params, tokenizer):
    """Load a pre-tokenized artifact -- see scripts/tokenize_dataset.py."""
    path = artifact_path(params)
    assert_meta_matches(path, params)
    dataset = load_from_disk(path)

    def add_labels(batch):
        # Built per batch rather than stored: labels are input_ids with the
        # prompt masked, so storing them would duplicate the token array and
        # freeze a masking policy that is likely to change.
        batch["labels"] = [
            [-100] * prompt_len + ids[prompt_len:]
            for ids, prompt_len in zip(batch["input_ids"], batch["prompt_len"])
        ]
        # exactly the columns the model accepts: nothing prunes them now that
        # remove_unused_columns is off
        return {"input_ids": batch["input_ids"], "labels": batch["labels"]}

    dataset.set_transform(add_labels)
    return dataset
