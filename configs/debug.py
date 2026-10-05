"""Tiny CPU run to check the pipeline end to end.

pixi run debug
"""

from mrunner.helpers.specification_helper import create_experiments_helper

from configs._helper_functions import HELPER_KWARGS, stamped

experiments_list = create_experiments_helper(
    experiment_name=stamped("debug"),
    **HELPER_KWARGS,
    # flat so mrunner can sweep them; dotted keys -> tyro nesting (src/config.py)
    base_config={
        "model.name": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "model.pad_token": "<unk>",
        "data.name": "roneneldan/TinyStories",
        "data.config": None,
        "data.split": "train[:64]",
        "data.text_field": "text",
        "data.max_length": 64,
        "data.padding": "max_length",
        "data.truncation": True,
        "train.output_dir": "out/debug",
        "train.learning_rate": 1e-4,
        "train.batch_size": 4,
        "train.num_epochs": 1,
        "train.max_steps": 5,
        "train.logging_steps": 1,
        "train.save_steps": 1000,
        "train.bf16": False,
    },
    params_grid={},  # local run takes only the first combination
)
