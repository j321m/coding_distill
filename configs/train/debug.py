"""Tiny CPU run to check the pipeline end to end.

pixi run debug
"""

from configs.common import make_experiment

experiment = make_experiment(
    "debug",
    {
        # random weights at Qwen2.5's vocab size: tests the pipeline, not the
        # model, in seconds on a CPU
        "model.name": "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
        "model.tokenizer": "Qwen/Qwen2.5-Coder-0.5B-Instruct",  # built the data
        "model.pad_token": "<|endoftext|>",
        # the artifact configs/data/tokenize_debug.py produced
        "data.dataset": "open-r1/codeforces-cots",
        "data.config": "solutions_py_decontaminated",
        "data.split": "train",
        "data.tokenizer": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "data.max_length": 32768,
        "data.drop_finish_reason": "length",
        "data.test_size": 0.05,
        "data.split_seed": 0,
        "data.data_dir": "out/data",
        "data.filter_max_tokens": 1024,
        "data.filter_n": 64,
        "train.output_dir": "out/debug",
        "train.learning_rate": 1e-4,
        "train.lr_scheduler_type": "linear",
        "train.lr_scheduler_kwargs": None,
        "train.warmup_steps": 0,
        "train.batch_size": 1,
        "train.gradient_accumulation_steps": 1,
        "train.num_epochs": 1,
        "train.max_steps": 5,
        "train.gradient_checkpointing": False,
        "train.sampling_strategy": "random",
        "train.use_liger_kernel": False,  # CUDA only
        "train.bf16": False,
        "train.logging_steps": 1,
        "train.logging_first_step": False,
        "train.eval_strategy": "no",
        "train.eval_steps": None,
        "train.eval_batch_size": 1,
        "train.save_steps": 1000,
        "train.save_total_limit": None,
        "train.report_to": "none",
    },
    gpus=0,  # CPU
)

experiments_list = [experiment]
