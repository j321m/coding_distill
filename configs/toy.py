from configs.common import data_params, make_experiment
from configs.tokenize import TOKENIZE

experiment = make_experiment(
    "toy_exp",
    dict(
        model_name=TOKENIZE["tokenizer"],
        model_pad_token="<|endoftext|>",
        # the artifact configs/tokenize.py produced
        **data_params(TOKENIZE, data_dir="$DATA_DIR"),
        # Checkpoints go to cluster storage, not to the mrunner-copied repo. The
        # path is context-specific, so it comes from $RUNS_DIR in prolog_cmd and
        # is expanded on the compute node -- see src/run_dir.py.
        train_output_dir="$RUNS_DIR",
        train_learning_rate=3e-4,
        train_batch_size=1,
        train_num_epochs=1,
        train_max_steps=100,
        train_logging_steps=10,
        train_save_steps=500,
        train_bf16=True,
        train_report_to="wandb",
    ),
)

experiments_list = [experiment]
