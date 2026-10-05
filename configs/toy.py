from mrunner.helpers.specification_helper import create_experiments_helper

from configs._helper_functions import HELPER_KWARGS, stamped

experiments_list = create_experiments_helper(
    experiment_name=stamped("toy_exp"),
    **HELPER_KWARGS,
    # flat so mrunner can sweep them; dotted keys -> tyro nesting (src/config.py)
    base_config={
        "model.name": "HuggingFaceTB/SmolLM2-135M",
        "model.pad_token": "<|endoftext|>",
        "data.name": "roneneldan/TinyStories",
        "data.config": None,
        "data.split": "train[:1%]",
        "data.text_field": "text",
        "data.max_length": 512,
        "data.padding": "max_length",
        "data.truncation": True,
        # Checkpoints go to the NVMe, not to the mrunner-copied repo (which is
        # on $HOME). Only reachable from a compute node, so it is resolved at
        # runtime -- $USER is expanded and the job id appended, see trainer.py.
        "train.output_dir": "/storage_nvme_4/coding_distill/$USER/runs",
        "train.learning_rate": 3e-4,
        "train.batch_size": 8,
        "train.num_epochs": 1,
        "train.max_steps": 100,
        "train.logging_steps": 10,
        "train.save_steps": 500,
        "train.bf16": True,
    },
    # key -> list of values; one array task per combination (cartesian product)
    params_grid={},
)
