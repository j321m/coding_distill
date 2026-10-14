"""Tiny CPU run to check the pipeline end to end.

pixi run debug
"""

from mrunner.helpers.specification_helper import create_experiments_helper

from configs._helper_functions import HELPER_KWARGS

experiments_list = create_experiments_helper(
    experiment_name="debug",
    **HELPER_KWARGS,
    # flat so mrunner can sweep them; dotted keys -> tyro nesting (src/config.py)
    base_config={
        # random weights, same tokenizer + chat template as Qwen2.5-Coder
        "model.name": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "data.name": "open-r1/codeforces-cots",
        "data.config": "solutions_py_decontaminated",
        "data.split": "train[:8]",  # the first rows are short (< 2.5k tokens)
        "data.max_length": 32768,
        "train.output_dir": "out/debug",
        "train.learning_rate": 1e-4,
        "train.batch_size": 2,
        "train.num_epochs": 1,
        "train.max_steps": 5,
        "train.logging_steps": 1,
        "train.save_steps": 1000,
        "train.bf16": False,
        "train.loss": "chunked",  # liger needs a GPU
        "train.loss_chunk_size": 1024,
    },
    params_grid={},  # local run takes only the first combination
)
