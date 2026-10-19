"""The 7B run from open-r1's own starting point, instead of Coder-Instruct.

Qwen2.5-Coder-7B-Instruct was the wrong base: its post-training already fixed a
short, direct answer style, and teaching it to emit <think> first fights that.
open-r1 (and DeepSeek's own R1 distills) start from Qwen2.5-Math-7B -- a base
model with nothing to overwrite. `-RoPE-300k` is that model with
max_position_embeddings 4096->32768 and rope_theta 10k->300k; weight_map and
total_size are identical to stock Qwen2.5-Math-7B, so the config is the whole
difference.

Identical to configs/train/sft_7b.py apart from the base model and the tokenizer, so
the comparison isolates the starting point.

`model.tokenizer`: the Math base shares a byte-identical vocabulary with the
Coder tokenizer that built the artifact, but declares eos as <|endoftext|> where
the data closes turns with <|im_end|>. The tokenizer that produced the data is
the authority on its special tokens, and src/model.py syncs the model's
generation config to it -- otherwise the checkpoint would stop on a token that
never appears and every generation would run to the eval's token cap. Note
OpenR1-Distill-7B ships eos=<|im_end|>, i.e. open-r1 made the same change.

    pixi run mrun --context helios run configs/train/sft_math7b.py
"""

from configs.common import data_params, make_experiment
from configs.data.tokenize import TOKENIZE

# effective batch 128, matching configs/train/sft.py: 4 GPUs x 1 x 32.
GPUS = 4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32

experiment = make_experiment(
    "sft_math7b",
    {
        "model.name": "open-r1/Qwen2.5-Math-7B-RoPE-300k",
        "model.tokenizer": TOKENIZE["tokenizer"],
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
        "train.bf16": True,
        "train.logging_steps": 1,
        "train.logging_first_step": True,
        "train.eval_strategy": "steps",
        "train.eval_steps": 60,
        "train.eval_batch_size": 1,
        "train.save_steps": 60,
        "train.save_total_limit": 2,
        "train.report_to": "wandb",
    },
    gpus=GPUS,
)

experiments_list = [experiment]
