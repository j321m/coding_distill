"""Reproduce open-r1's OlympicCoder-7B recipe.

What the published run actually used, from its training_args.bin and
trainer_state.json (the recipe YAML disagrees and is wrong): lr 4e-5,
max_grad_norm 1.0, 10 epochs, effective batch 128, cosine to 10%, 3% warmup,
Liger off, DeepSpeed ZeRO-3 with bf16 compute over fp32 master weights.

Two earlier runs here (4e-5 and 1e-5, both pure bf16 under DDP) sampled
incoherently -- 2-5 of 27 generations closed <think> -- while teacher-forced
loss looked fine. scripts/compare_weights.py showed why: 66% of the weights were
bit-identical to the base after 3,020 steps (96% of those with |w| in
[1e-2, 1e-1), 99% of RMSNorm), against 8% for OlympicCoder-7B. Without an fp32
master, Adam steps below half a bf16 ulp are simply lost. This run restores the
master copy through FSDP, which shards the 16 bytes/param across all 8 GPUs.

    pixi run mrun --context helios run configs/train/olympic_7b.py

Deviations: sdpa instead of flash_attention_2, which has no aarch64 wheels, and
FSDP full-shard instead of ZeRO-3 for the fp32 master weights. Device count
matches theirs (2 nodes x 4 GPUs); batch split 1 x 16 against their 2 x 8.
"""

from configs.common import make_experiment

# effective batch 128: 2 nodes x 4 GPUs x 1 x 16.
GPUS = 8  # across the helios context's `nodes`
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16

experiment = make_experiment(
    "olympic_7b",
    {
        "model.name": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "model.tokenizer": None,
        "model.pad_token": "<|endoftext|>",
        # fp32 master weights, sharded by FSDP below; bf16 does the compute
        "model.dtype": "float32",
        # the artifact configs/data/tokenize_cpp.py built; checked against its meta.json
        "data.dataset": "open-r1/codeforces-cots",
        "data.config": "solutions_decontaminated",
        "data.split": "train",
        "data.tokenizer": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "data.max_length": 32768,
        "data.drop_finish_reason": "length",
        "data.test_size": 0.01,
        "data.split_seed": 0,
        "data.data_dir": "$DATA_DIR",
        "data.filter_max_tokens": None,
        "data.filter_n": None,
        "train.output_dir": "$RUNS_DIR",
        "train.resume_job_id": None,
        "train.learning_rate": 4e-5,
        "train.lr_scheduler_type": "cosine_with_min_lr",
        "train.lr_scheduler_kwargs": {"min_lr_rate": 0.1},
        "train.warmup_steps": 0.03,  # below 1 = fraction of total steps
        "train.batch_size": BATCH_SIZE,
        "train.gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "train.num_epochs": 10,
        "train.max_steps": -1,  # epochs decide the length
        "train.max_grad_norm": 1.0,
        "train.gradient_checkpointing": True,
        # theirs, and at batch_size=1 there is no padding for grouping to save:
        # it would only correlate each step's samples by length
        "train.sampling_strategy": "random",
        "train.use_liger_kernel": True,
        # full_shard: weights, grads and Adam state split over every GPU, so the
        # fp32 master copy (16 bytes/param) is paid once, not once per card
        "train.fsdp": "full_shard auto_wrap",
        "train.fsdp_config": {"transformer_layer_cls_to_wrap": "Qwen2DecoderLayer"},
        "train.bf16": True,
        "train.logging_steps": 1,
        "train.logging_first_step": True,
        # ~315 steps per epoch, so every-60 would be 52 evaluations
        "train.eval_strategy": "epoch",
        "train.eval_steps": None,
        "train.eval_batch_size": 1,
        "train.save_strategy": "steps",
        "train.save_steps": 604,  # ~2 epochs at 302 steps each
        # keeps all five: at 43GB each that is ~215GB on $RUNS_DIR
        "train.save_total_limit": 6,
        "train.resume_from_checkpoint": None,
        "train.report_to": "wandb",
    },
    gpus=GPUS,
)

experiments_list = [experiment]
