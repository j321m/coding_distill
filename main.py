import dataclasses
import os

import tyro

from src.config import MainArgs
from src.dataset import assert_tokenizer_matches, build_dataset
from src.model import build_model
from src.paths import resolve_output_dir
from src.provenance import load_git, write_run_metadata
from src.tracking import init_wandb
from src.trainer import train

params = tyro.cli(MainArgs)

# Before the model and the dataset: both can fail, and a run that dies while
# loading is the one whose config you most want to read afterwards.
output_dir = resolve_output_dir(params.train.output_dir)
os.makedirs(output_dir, exist_ok=True)
metadata = write_run_metadata(output_dir, {**dataclasses.asdict(params), **load_git()})
print(f"output_dir: {output_dir}")

if params.train.report_to == "wandb":
    init_wandb(metadata)

model, tokenizer = build_model(params.model)
# the model may legitimately borrow another's artifact -- Qwen2.5-Coder
# sizes share a tokenizer -- so compare what the tokenizer is, not its name
assert_tokenizer_matches(params.data, tokenizer)
train_dataset, eval_dataset = build_dataset(params.data)
train(model, tokenizer, train_dataset, eval_dataset, params.train, output_dir)
