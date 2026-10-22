"""The 7B run: Qwen2.5-Coder-7B-Instruct on the full Python split, 4 GPUs.

Same recipe as configs/train/sft.py -- open-r1's own hyperparameters for this dataset
-- so a difference in outcome is attributable to scale rather than to tuning.

What the probes settled:
  - fused_linear_cross_entropy is back to finite gradients, and takes the peak
    to ~60GB, so the model fits per-GPU and DDP replaces the FSDP sharding that
    PLAN.md section 10 assumed. mrunner_run.py starts one process per GPU in
    `GPUS` (checked against the context's --gres at submit); that is the whole
    distributed story.
  - ~6s per forward/backward at 7B, so a 32-pass step is ~3.2min and 290 steps
    is ~16h. The context needs a matching --time.

The tokenizer is byte-identical across Qwen2.5-Coder sizes, so the artifact
tokenized for 0.5B is reused unchanged: `data.tokenizer` keeps naming it, which
is what assert_meta_matches compares, while the model is 7B.

    pixi run mrun --context helios run configs/train/sft_7b.py
"""

from configs.common import data_params, make_experiment
from configs.data.tokenize import TOKENIZE

# effective batch 128, matching configs/train/sft.py: 4 GPUs x 1 x 32.
GPUS = 4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32

experiment = make_experiment(
    "sft_7b",
    {
        "model.name": "Qwen/Qwen2.5-Coder-7B-Instruct",
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
        # on at 7B: without its fused cross-entropy the fp32 logits are 19.9GB
        # (32k x 152064) plus as much again for their gradient, which is the
        # difference between fitting on one card and not
        "train.use_liger_kernel": True,
        "train.bf16": True,
        "train.logging_steps": 1,
        "train.logging_first_step": True,
        "train.eval_strategy": "steps",
        # 60, not sft.py's 30: the 389-row eval set costs ~13min at this scale,
        # so every-30 would add ~2h to a ~16h run
        "train.eval_steps": 60,
        "train.eval_batch_size": 1,
        "train.save_steps": 60,
        "train.save_total_limit": 2,
        "train.report_to": "wandb",
    },
    gpus=GPUS,
)

experiments_list = [experiment]
