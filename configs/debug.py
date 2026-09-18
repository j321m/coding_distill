"""Tiny CPU run to check the pipeline end to end.

pixi run debug
"""

from configs.common import data_params, make_experiment
from configs.tokenize import TOKENIZE

experiment = make_experiment(
    "debug",
    dict(
        model_name=TOKENIZE["tokenizer"],
        model_pad_token="<|endoftext|>",
        # the artifact configs/tokenize_debug.py produced
        **data_params(TOKENIZE, data_dir="out/data", filter_n=64),
        train_output_dir="out/debug",
        train_learning_rate=1e-4,
        train_batch_size=1,
        train_num_epochs=1,
        train_max_steps=5,
        train_logging_steps=1,
        train_save_steps=1000,
        train_bf16=False,
    ),
)

experiments_list = [experiment]
