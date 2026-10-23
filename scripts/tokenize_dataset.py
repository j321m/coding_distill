"""Pre-tokenize a CoT dataset into a training-ready artifact.

pixi run mrun --context helios run configs/data/tokenize.py
pixi run python -m scripts.tokenize_dataset --ex configs/data/tokenize_debug.py
"""

import os
import random
import shutil

from datasets import DatasetDict, load_dataset
from mrunner.helpers.client_helper import get_configuration
from transformers import AutoTokenizer

from src.dataset import artifact_path, write_meta

params = get_configuration()

tokenizer = AutoTokenizer.from_pretrained(params.tokenizer)
dataset = load_dataset(params.dataset, params.config, split=params.split)

if params.get("drop_finish_reason"):
    dataset = dataset.filter(
        lambda reason: reason != params.drop_finish_reason,
        input_columns="finish_reason",
        num_proc=params.num_proc,
    )

if params.get("filter_max_tokens"):
    # R1's token counts -- a proxy, used only to bound the debug subset
    dataset = dataset.filter(
        lambda meta: meta["total_tokens"] <= params.filter_max_tokens,
        input_columns="api_metadata",
        num_proc=params.num_proc,
    )
if params.get("filter_n"):
    dataset = dataset.select(range(min(params.filter_n, len(dataset))))


def tokenize(messages):
    # add_generation_prompt=True puts `<|im_start|>assistant\n` in the prompt,
    # where it belongs: at inference the serving stack emits it, so supervising
    # it would teach the model to write its own header.
    prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    # ends with `<|im_end|>`, so stopping behaviour is supervised -- appending an
    # eos by hand on top would be a bug
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # the template already inserted every control token
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

    assert len(input_ids) <= params.max_length, (
        f"sample of {len(input_ids)} tokens exceeds max_length={params.max_length}; "
        "the structural bound (max prompt + generation cap) no longer holds"
    )
    assert prompt_len < len(input_ids), "prompt covers the whole sample"
    return {
        "input_ids": input_ids,
        "labels": [-100] * prompt_len + input_ids[prompt_len:],
        "length": len(input_ids),
    }


def split_by_problem(dataset):
    """Hold out whole problems, not rows.

    The C++ configs carry five traces per problem, so a row-wise split would put
    the same problem on both sides and the eval loss would measure memorisation.
    """
    ids = sorted(set(dataset["id"]))
    random.Random(params.split_seed).shuffle(ids)
    held_out = set(ids[: max(1, round(len(ids) * params.test_size))])
    is_test = [problem in held_out for problem in dataset["id"]]
    return DatasetDict(
        train=dataset.select([i for i, t in enumerate(is_test) if not t]),
        test=dataset.select([i for i, t in enumerate(is_test) if t]),
    )


# split before tokenizing: `id` is among the columns the map drops
splits = split_by_problem(dataset)
splits = splits.map(
    tokenize,
    input_columns="messages",
    remove_columns=dataset.column_names,
    num_proc=params.num_proc,
)
# Arrow infers int64 for labels, doubling that column against int32 input_ids
dataset = splits.cast_column("labels", splits["train"].features["input_ids"])

path = artifact_path(params)
staging = f"{path}.staging"
shutil.rmtree(staging, ignore_errors=True)
dataset.save_to_disk(staging)
# rename is atomic: an interrupted job leaves no artifact rather than a plausible
# truncated one, which the loss curve would never reveal
shutil.rmtree(path, ignore_errors=True)
os.rename(staging, path)
rows = {split: len(subset) for split, subset in dataset.items()}
write_meta(path, params, rows, tokenizer)

print(f"wrote {rows} to {path}")
