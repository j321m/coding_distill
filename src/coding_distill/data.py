"""codeforces-cots -> chat-templated, prompt-masked SFT examples.

This is the part that differs most from pretraining-style packing: each sample is
a single (problem -> reasoning trace + solution) pair, and the loss must not cover
the problem statement.
"""

import logging

from datasets import load_dataset

logger = logging.getLogger(__name__)


def _to_messages(example: dict) -> list:
    """Normalise one row into a chat `messages` list.

    NOTE: verify against the subset you actually load -- codeforces-cots subsets do
    not all share a schema. Inspect with scripts/inspect_dataset.py before trusting this.
    """
    if example.get("messages"):
        return example["messages"]
    if example.get("prompt") and example.get("generation"):
        return [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["generation"]},
        ]
    raise KeyError(f"cannot build messages from columns: {sorted(example)}")


def build_dataset(cfg, tokenizer):
    """Return a tokenized dataset with `input_ids`, `attention_mask`, `labels`."""
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    logger.info(f"loaded {len(ds)} rows from {cfg.dataset_name}:{cfg.dataset_config}")

    if cfg.n_train_samples is not None:
        ds = ds.select(range(min(cfg.n_train_samples, len(ds))))

    def encode(example):
        messages = _to_messages(example)
        prompt_messages = messages[:-1]

        full = tokenizer.apply_chat_template(messages, tokenize=True)
        labels = list(full)

        if cfg.mask_prompt:
            prompt = tokenizer.apply_chat_template(
                prompt_messages, tokenize=True, add_generation_prompt=True
            )
            labels[: len(prompt)] = [-100] * len(prompt)

        # truncate from the left of nothing -- drop the tail, keeping the prompt intact
        full, labels = full[: cfg.max_seq_length], labels[: cfg.max_seq_length]
        return {
            "input_ids": full,
            "attention_mask": [1] * len(full),
            "labels": labels,
        }

    tokenized = ds.map(
        encode, remove_columns=ds.column_names, num_proc=cfg.num_proc, desc="tokenizing"
    )

    # a sample whose labels are entirely masked contributes no gradient
    n_before = len(tokenized)
    tokenized = tokenized.filter(lambda ex: any(t != -100 for t in ex["labels"]))
    if len(tokenized) < n_before:
        logger.warning(f"dropped {n_before - len(tokenized)} fully-masked samples")

    return tokenized
