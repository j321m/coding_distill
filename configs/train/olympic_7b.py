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

Deviation: sdpa instead of flash_attention_2, which has no aarch64 wheels.
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
        # fp32 weights, bf16 compute (train_bf16 below): their precision regime
        model_dtype="float32",
        **data_params(TOKENIZE_CPP, data_dir="$DATA_DIR"),
        train_output_dir="$RUNS_DIR",
        train_learning_rate=4e-5,
        train_lr_scheduler_type="cosine_with_min_lr",
        train_lr_scheduler_kwargs={"min_lr_rate": 0.1},
        train_warmup_steps=0.03,  # below 1 = fraction of total steps
        train_batch_size=BATCH_SIZE,
        train_gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        train_num_epochs=10,
        # SMOKE TEST -- set back to -1 for the real run. 8 steps x 16 x 8 = 1024
        # forward/backward passes: enough for the lr schedule, the optimizer and
        # FLCE-under-FSDP2 to all have run, and to read sec_per_step and memory.
        train_max_steps=8,
        train_max_grad_norm=1.0,
        train_fsdp=True,
        train_fsdp_config={
            "version": 2,
            "cpu_ram_efficient_loading": False,
            # in place of gradient_checkpointing, which under FSDP re-gathers
            # parameters a second time in the backward pass
            "activation_checkpointing": True,
            # a single HF-format checkpoint, which eval_lcb.py and vllm read as-is
            "state_dict_type": "FULL_STATE_DICT",
            # keep the gathered bf16 params between forward and backward: halves
            # the all-gathers, which cross the inter-node link, for ~15GB of memory
            "reshard_after_forward": False,
        },
        # FSDP all-gathers inside every micro-batch, so every micro-batch waits
        # for the rank holding the longest sequence. Grouping by length keeps the
        # 8 ranks' sequences similar -- without it the random-order DDP run was
        # already 1.5x slower, and per-micro-batch sync would make that worse.
        train_sampling_strategy="group_by_length",
        train_use_liger_kernel=True,
        train_liger_kernel_config={"fused_linear_cross_entropy": True},
        train_bf16=True,
        train_logging_steps=1,
        train_logging_first_step=True,
        train_eval_strategy="no",  # SMOKE TEST -- "epoch" for the real run
        train_eval_batch_size=1,
        train_save_strategy="steps",
        train_save_steps=604,  # ~2 epochs at 302 steps each
        # fp32 weights (30GB) plus optimizer state (61GB) per checkpoint
        train_save_total_limit=6,
        train_report_to="wandb",
    ),
    script=torchrun(GPUS_PER_NODE),
)

experiments_list = [experiment]
