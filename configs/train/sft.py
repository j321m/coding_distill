"""The real run: 0.5B on the full Python split.

Hyperparameters follow open-r1's own SFT recipe for this dataset
(recipes/OpenR1-Distill-7B/sft/config_distill.yaml), so a bad result is
attributable to something other than a guessed learning rate.

Probe before committing to the full run -- Liger is untestable outside CUDA and
no-ops silently on an unsupported architecture:

    pixi run mrun --context helios run configs/train/sft.py     # after a max_steps probe
"""

from configs.common import data_params, make_experiment
from configs.data.tokenize import TOKENIZE

# effective batch 128, matching the recipe's 2 x 8 accum x 8 GPUs. 
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 128

experiment = make_experiment(
    "sft",
    dict(
        model_name=TOKENIZE["tokenizer"],
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
        # `fused_linear_cross_entropy` produces NaN gradients; the
        # remaining rope/rms_norm/swiglu kernels are worth ~22% speedups
        train_liger_kernel_config={"fused_linear_cross_entropy": False},
        train_bf16=True,
        train_logging_steps=1,
        train_logging_first_step=True,
        train_eval_strategy="steps",
        train_eval_steps=30,
        train_eval_batch_size=1,
        train_save_steps=60,
        train_save_total_limit=2,
        train_report_to="wandb",
    ),
)

experiments_list = [experiment]
