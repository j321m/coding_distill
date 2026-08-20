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
    script="python main.py",
    exclude=EXCLUDE,
    # flat so mrunner can sweep them; main.py nests by prefix
    parameters=Munch(
        model_name="HuggingFaceTB/SmolLM2-135M",
        model_pad_token="<|endoftext|>",
        data_name="roneneldan/TinyStories",
        data_config=None,
        data_split="train[:1%]",
        data_text_field="text",
        data_max_length=512,
        data_padding="max_length",
        data_truncation=True,
        train_output_dir="out",
        train_learning_rate=3e-4,
        train_batch_size=8,
        train_num_epochs=1,
        train_max_steps=100,
        train_logging_steps=10,
        train_save_steps=500,
        train_bf16=True,
    ),
)

experiments_list = [experiment]
