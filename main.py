import os

from mrunner.helpers.client_helper import get_configuration

from src.data import build_dataset
from src.model import build_model
from src.run_dir import resolve_output_dir, write_run_metadata
from src.tracking import init_wandb
from src.trainer import train

# model_name -> params.model.name
params = get_configuration(nesting_prefixes=("model_", "data_", "train_"))

# Before the model and the dataset: both can fail, and a run that dies while
# loading is the one whose config you most want to read afterwards.
output_dir = resolve_output_dir(params.train.output_dir)
os.makedirs(output_dir, exist_ok=True)
metadata = write_run_metadata(output_dir, params)
print(f"output_dir: {output_dir}")

if "wandb" in params.train.get("report_to", "none"):
    init_wandb(metadata)

model, tokenizer = build_model(params.model)
dataset = build_dataset(params.data, tokenizer)
train(model, tokenizer, dataset, params.train, output_dir)
