"""Baseline: SFT-distill a small coder on codeforces-cots reasoning traces.

Submit:
    mrunner --config ~/.mrunner.yaml --context <your_context> run mrunner_specs/sft_baseline.py

Run the first grid point locally instead:
    pixi run python main.py --ex mrunner_specs/sft_baseline.py

Params are flat `section.field` paths consumed by config_from_params; the attrs
sections in coding_distill.config are the source of truth. validate_experiments
parses every grid point here, so a typo fails now rather than on a compute node.
"""

from mrunner.helpers.specification_helper import create_experiments_helper
from mrunner.logging.wandb import WandbLogger

from coding_distill.presets import BASE, label_experiments, validate_experiments

base_config = BASE

# One axis for the first sweep -- the LR is what most often decides whether trace
# SFT helps or wrecks the base model. Values must be a list/tuple; mrunner asserts.
# (Suffix a key with `___` to zip it against other `___` keys instead of taking the
# cartesian product.)
params_grid = {
    "optim.learning_rate": [5e-6, 1e-5, 2e-5],
}

experiments_list = create_experiments_helper(
    experiment_name="coding-distill-sft-baseline",
    base_config=base_config,
    params_grid=params_grid,
    script="python3 main.py",
    python_path=".:src",
    # Pass a logger explicitly: with neither `logger` nor `with_neptune`, mrunner
    # quietly falls back to Neptune. This also forwards WANDB_API_KEY to the job.
    logger=WandbLogger(
        project_name=BASE["logging.project"],
        tags=["coding_distill", "sft", "baseline"],
    ),
    # mrunner copies the working dir to the cluster; keep the shipped tree small.
    # `.git`/`.gitignore`/`.gitmodules` are already excluded by exclude_git_files.
    exclude=[
        ".venv",
        ".pixi",
        "out",
        "outputs",
        "data",
        "wandb",
        "__pycache__",
        "*.pyc",
    ],
)

label_experiments(experiments_list, params_grid, output_root="./out")
validate_experiments(experiments_list)
