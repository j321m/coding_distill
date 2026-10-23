"""Reproduce open-r1's OlympicCoder-7B recipe.

recipes/OlympicCoder-7B/sft/config_v00.00.yaml.

The recipe says lr 1e-5, the model card says 4e-5. A first run at 4e-5 trained
cleanly by every teacher-forced measure -- loss falling, grad_norm stable, eval
loss moving 0.5747 -> 0.5820 over ten epochs -- and yet sampled incoherently:
2-3 of 27 generations closed <think>, and fenced code blocks per generation grew
80 -> 144 -> 261 across epochs 8, 9, 10 as it drafted and abandoned attempts
until the token budget ran out. Re-prompting in C++ steered the language but not
the termination, so the prompt was not the cause. 1e-5 is the recipe's value.

    pixi run mrun --context helios run configs/train/olympic_7b.py

Deviation: sdpa instead of flash_attention_2, which has no aarch64 wheels. Device
count matches theirs (2 nodes x 4 GPUs), batch split 1 x 16 against their 2 x 8.
"""

from configs.common import data_params, make_experiment, torchrun
from configs.data.tokenize_cpp import TOKENIZE_CPP

GPUS_PER_NODE = 4  # node count comes from --nodes in clusters.yaml
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

experiment = make_experiment(
    "olympic_7b",
    dict(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        model_pad_token="<|endoftext|>",
        **data_params(TOKENIZE_CPP, data_dir="$DATA_DIR"),
        train_output_dir="$RUNS_DIR",
        # job 24931873 reached 6.5 epochs in 28h and was cut off at the walltime;
        # picks up its newest checkpoint, optimizer and lr schedule included
        train_resume_job_id="24931873",
        train_learning_rate=1e-5,
        train_lr_scheduler_type="cosine_with_min_lr",
        train_lr_scheduler_kwargs={"min_lr_rate": 0.1},
        train_warmup_steps=0.03,  # below 1 = fraction of total steps
        train_batch_size=BATCH_SIZE,
        train_gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        train_num_epochs=10,
        train_max_steps=-1,
        # five times tighter than the default, and theirs
        train_max_grad_norm=0.2,
        train_gradient_checkpointing=True,
        # theirs, and at batch_size=1 there is no padding for grouping to save:
        # it would only correlate each step's samples by length
        train_sampling_strategy="random",
        train_use_liger_kernel=True,
        train_liger_kernel_config={"fused_linear_cross_entropy": True},
        train_bf16=True,
        train_logging_steps=1,
        train_logging_first_step=True,
        # ~315 steps per epoch, so every-60 would be 52 evaluations
        train_eval_strategy="epoch",
        train_eval_batch_size=1,
        train_save_strategy="steps",
        train_save_steps=604,  # ~2 epochs at 302 steps each
        # keeps all five: at 43GB each that is ~215GB on $RUNS_DIR
        train_save_total_limit=6,
        train_report_to="wandb",
    ),
    script=torchrun(GPUS_PER_NODE),
)

experiments_list = [experiment]
