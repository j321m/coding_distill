from configs.common import make_experiment

experiment = make_experiment(
    "toy_exp",
    dict(
        model_name="HuggingFaceTB/SmolLM2-135M",
        model_pad_token="<|endoftext|>",
        data_name="roneneldan/TinyStories",
        data_config=None,
        data_split="train[:1%]",
        data_text_field="text",
        data_max_length=512,
        data_padding="max_length",
        data_truncation=True,
        # Checkpoints go to cluster storage, not to the mrunner-copied repo. The
        # path is context-specific, so it comes from $RUNS_DIR in prolog_cmd and
        # is expanded on the compute node -- see src/run_dir.py.
        train_output_dir="$RUNS_DIR",
        train_learning_rate=3e-4,
        train_batch_size=8,
        train_num_epochs=1,
        train_max_steps=100,
        train_logging_steps=10,
        train_save_steps=500,
        train_bf16=True,
        train_report_to="wandb",
    ),
)

experiments_list = [experiment]
