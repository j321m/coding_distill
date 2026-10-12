"""The 7B run: Qwen2.5-Coder-7B-Instruct on the full Python split, 4 GPUs.

Same recipe as configs/train/sft.py -- open-r1's own hyperparameters for this dataset
-- so a difference in outcome is attributable to scale rather than to tuning.

What the probes settled:
  - fused_linear_cross_entropy is back to finite gradients, and takes the peak
    to ~60GB, so the model fits per-GPU and DDP replaces the FSDP sharding that
    PLAN.md section 10 assumed. torchrun below is the whole distributed story.
  - ~6s per forward/backward at 7B, so a 32-pass step is ~3.2min and 290 steps
    is ~16h. The context needs a matching --time.

The tokenizer is byte-identical across Qwen2.5-Coder sizes, so the artifact
tokenized for 0.5B is reused unchanged: `data_tokenizer` keeps naming it, which
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
    dict(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        model_pad_token="<|endoftext|>",
        **data_params(TOKENIZE, data_dir="$DATA_DIR"),
        train_output_dir="$RUNS_DIR",
        train_learning_rate=4e-5,
        train_lr_scheduler_type="cosine_with_min_lr",
        train_lr_scheduler_kwargs={"min_lr_rate": 0.1},
        train_warmup_steps=0.03,  # below 1 = fraction of total steps
        train_batch_size=BATCH_SIZE,
        train_gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        train_num_epochs=5,
        train_max_steps=-1,  # epochs decide the length
        train_gradient_checkpointing=True,
        train_sampling_strategy="group_by_length",
        train_use_liger_kernel=True,
        # on at 7B: without it the fp32 logits are 19.9GB (32k x 152064) plus
        # as much again for their gradient, which is the difference between
        # fitting on one card and not
        train_liger_kernel_config={"fused_linear_cross_entropy": True},
        train_bf16=True,
        train_logging_steps=1,
        train_logging_first_step=True,
        train_eval_strategy="steps",
        # 60, not sft.py's 30: the 389-row eval set costs ~13min at this scale,
        # so every-30 would add ~2h to a ~16h run
        train_eval_steps=60,
        train_eval_batch_size=1,
        train_save_steps=60,
        train_save_total_limit=2,
        train_report_to="wandb",
    ),
    script=f"torchrun --nproc_per_node={GPUS} main.py",
)

experiments_list = [experiment]
