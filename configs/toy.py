from munch import Munch
from mrunner.cli.mrunner_cli import register_after_run_callback
from mrunner.experiment import Experiment


def _print_log_location(sweep, experiments):
    """Called by mrunner right after the job is submitted."""
    log = f"{sweep.grid_logs_dir}/slurm_"
    log += "0.log" if len(experiments) == 1 else "<array_task_id>.log"
    print(f"\nlog: ssh {sweep.slurm_url} tail -f {log}")


register_after_run_callback(_print_log_location)

# mrunner tars up everything in the cwd that is not listed here. Note that
# providing `exclude` REPLACES mrunner's default of [.git, .gitignore,
# .gitmodules], so those have to be repeated.
#
# The pixi bits are excluded on purpose: `.pixi` is a local, platform-specific
# env with a huge file count, and the manifest is kept out so nothing on the
# cluster can accidentally materialize an env inside the copied repo (home has a
# file-count quota). The remote env lives at $PIXI_HOME and is activated by
# `prolog_cmd` in clusters.yaml -- see update_pixi.py.
EXCLUDE = [
    ".git",
    ".gitignore",
    ".gitmodules",
    ".pixi",
    "pixi.toml",
    "pixi.lock",
    ".vscode",
    "__pycache__",
    "out",
]

experiment = Experiment(
    name="toy_exp",
    # plain `python`, not `pixi run` -- prolog_cmd already activated $PIXI_HOME
    script="python mrunner_run.py",
    exclude=EXCLUDE,
    # flat so mrunner can sweep them; dotted keys -> tyro nesting (src/config.py)
    parameters=Munch(
        {
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
        }
    ),
)

experiments_list = [experiment]
