from mrunner.helpers.specification_helper import create_experiments_helper

from configs._helper_functions import HELPER_KWARGS

experiments_list = create_experiments_helper(
    experiment_name="toy_exp",
    **HELPER_KWARGS,
    # flat so mrunner can sweep them; dotted keys -> tyro nesting (src/config.py)
    base_config={
        "model.name": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "data.name": "open-r1/codeforces-cots",
        "data.config": "solutions_py_decontaminated",
        "data.split": "train[:10%]",
        "data.max_length": 32768,
        # Checkpoints go to the NVMe, not to the mrunner-copied repo (which is
        # on $HOME). Only reachable from a compute node, so it is resolved at
        # runtime -- $USER is expanded and the job id appended, see trainer.py.
        "train.output_dir": "/storage_nvme_4/coding_distill/$USER/runs",
        "train.learning_rate": 3e-4,
        "train.batch_size": 1,  # samples up to 26k tokens
        "train.num_epochs": 1,
        "train.max_steps": 100,
        "train.logging_steps": 10,
        "train.save_steps": 500,
        "train.bf16": True,
    },
    # key -> list of values; one array task per combination (cartesian product)
    params_grid={},
)
