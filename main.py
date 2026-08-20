from mrunner.helpers.client_helper import get_configuration

from src.data import build_dataset
from src.model import build_model
from src.trainer import train

# model_name -> params.model.name
params = get_configuration(nesting_prefixes=("model_", "data_", "train_"))

model, tokenizer = build_model(params.model)
dataset = build_dataset(params.data, tokenizer)
train(model, tokenizer, dataset, params.train)
