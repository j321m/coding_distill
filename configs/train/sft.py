"""The real run: 0.5B on the full Python split.

Hyperparameters follow open-r1's own SFT recipe for this dataset
(recipes/OpenR1-Distill-7B/sft/config_distill.yaml), so a bad result is
attributable to something other than a guessed learning rate.

Probe before committing to the full run -- Liger is untestable outside CUDA and
no-ops silently on an unsupported architecture:

    pixi run mrun --context helios run configs/train/sft.py     # after a max_steps probe
"""

from configs.common import make_experiment

# effective batch 128, the open-r1 recipe value: 2 nodes x 4 GPUs x 1 x 16.
GPUS = 8  # across the helios context's `nodes`
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

experiment = make_experiment(
    "sft",
    {
        "model.name": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "model.tokenizer": None,
        "model.pad_token": "<|endoftext|>",
        "model.dtype": None,
        # the artifact configs/data/tokenize.py built; checked against its meta.json
        "data.dataset": "open-r1/codeforces-cots",
        "data.config": "solutions_py_decontaminated",
        "data.split": "train",
        "data.tokenizer": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "data.max_length": 32768,
        "data.drop_finish_reason": "length",
        "data.test_size": 0.05,
        "data.split_seed": 0,
        "data.data_dir": "$DATA_DIR",
        "data.filter_max_tokens": None,
        "data.filter_n": None,
        "train.output_dir": "$RUNS_DIR",
        "train.learning_rate": 4e-5,
        "train.lr_scheduler_type": "cosine_with_min_lr",
        "train.lr_scheduler_kwargs": {"min_lr_rate": 0.1},
        "train.warmup_steps": 0.03,  # below 1 = fraction of total steps
        "train.batch_size": BATCH_SIZE,
        "train.gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "train.num_epochs": 1,
        "train.max_steps": -1,  # epochs decide the length
        "train.max_grad_norm": 1.0,
        "train.gradient_checkpointing": True,
        "train.sampling_strategy": "group_by_length",
        "train.use_liger_kernel": True,
        "train.fsdp": None,
        "train.fsdp_config": None,
        "train.bf16": True,
        "train.logging_steps": 1,
        "train.logging_first_step": True,
        "train.eval_strategy": "steps",
        "train.eval_steps": 30,
        "train.eval_batch_size": 1,
        "train.save_strategy": "steps",
        "train.save_steps": 60,
        "train.save_total_limit": 2,
        "train.resume_job_id": None,
        "train.resume_from_checkpoint": None,
        "train.report_to": "wandb",
    },
    gpus=GPUS,
)

experiments_list = [experiment]
