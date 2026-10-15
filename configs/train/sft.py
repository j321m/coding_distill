"""The real run: 0.5B on the full Python split.

Hyperparameters follow open-r1's own SFT recipe for this dataset
(recipes/OpenR1-Distill-7B/sft/config_distill.yaml), so a bad result is
attributable to something other than a guessed learning rate. The exception is
the effective batch: 512, where the recipe has 128.

Probe before committing to the full run -- Liger is untestable outside CUDA and
no-ops silently on an unsupported architecture:

    pixi run mrun --context helios run configs/train/sft.py     # after a max_steps probe
"""

from configs.common import data_params, make_experiment
from configs.data.tokenize import TOKENIZE

# effective batch 512: 4 GPUs x 1 x 128, what past `sft` runs trained with
GPUS = 4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 128

experiment = make_experiment(
    "sft",
    {
        "model.name": TOKENIZE["tokenizer"],
        "model.tokenizer": None,
        "model.pad_token": "<|endoftext|>",
        **data_params(
            TOKENIZE, data_dir="$DATA_DIR", filter_max_tokens=None, filter_n=None
        ),
        "train.output_dir": "$RUNS_DIR",
        "train.learning_rate": 4e-5,
        "train.lr_scheduler_type": "cosine_with_min_lr",
        "train.lr_scheduler_kwargs": {"min_lr_rate": 0.1},
        "train.warmup_steps": 0.03,  # below 1 = fraction of total steps
        "train.batch_size": BATCH_SIZE,
        "train.gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "train.num_epochs": 5,
        "train.max_steps": -1,  # epochs decide the length
        "train.gradient_checkpointing": True,
        "train.sampling_strategy": "group_by_length",
        "train.use_liger_kernel": True,
        # `fused_linear_cross_entropy` produces NaN gradients; the
        # remaining rope/rms_norm/swiglu kernels are worth ~22% speedups
        "train.liger_kernel_config": {"fused_linear_cross_entropy": False},
        "train.bf16": True,
        "train.logging_steps": 1,
        "train.logging_first_step": True,
        "train.eval_strategy": "steps",
        "train.eval_steps": 30,
        "train.eval_batch_size": 1,
        "train.save_steps": 60,
        "train.save_total_limit": 2,
        "train.report_to": "wandb",
    },
    gpus=GPUS,
)

experiments_list = [experiment]
