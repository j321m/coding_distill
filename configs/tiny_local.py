"""Smoke-test config: 135M model, 32 samples, runnable on CPU.

    pixi run python main.py --local-config configs/tiny_local.py

Overrides only what has to shrink; everything else comes from BASE, so this config
tracks the baseline instead of drifting from a copy of it.
"""

from coding_distill.presets import BASE, override

params = override(
    BASE,
    {
        "data.max_seq_length": 1024,
        "data.n_train_samples": 32,
        "data.n_eval_samples": 8,
        "data.num_proc": 2,
        "model.student_name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "model.attn_implementation": "eager",
        "model.torch_dtype": "float32",
        "model.gradient_checkpointing": False,
        "optim.warmup_ratio": 0.0,
        "optim.n_epochs": 1,
        "optim.grad_accumulation_steps": 1,
        "eval.benchmarks": ["humanevalplus"],
        "eval.max_new_tokens": 512,
        "logging.run_name": "tiny_local",
        "logging.tags": ["smoke"],
        "logging.output_dir": "/tmp/coding_distill/tiny_local",
        "logging.save_steps": 1000,
        "logging.logging_steps": 1,
        "logging.use_wandb": False,
    },
)
