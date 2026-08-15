"""Baseline: SFT-distill a small coder on codeforces-cots reasoning traces.

Run:
    mrunner --config ~/.mrunner.yaml --context <your_context> run mrunner_specs/sft_baseline.py

Every key in base_config / params_grid is a flat `section.field` path consumed by
coding_distill.config.config_from_params -- the sections there are the source of truth,
and a missing or misspelled key fails loudly at startup rather than silently defaulting.
"""

from mrunner.helpers.specification_helper import create_experiments_helper

base_config = {
    "data.dataset_name": "open-r1/codeforces-cots",
    "data.dataset_config": "solutions_py",
    "data.split": "train",
    "data.max_seq_length": 16384,
    "data.n_train_samples": None,
    "data.n_eval_samples": 200,
    "data.mask_prompt": True,
    "data.num_proc": 8,
    "model.student_name": "Qwen/Qwen2.5-Coder-1.5B",
    "model.attn_implementation": "flash_attention_2",
    "model.torch_dtype": "bfloat16",
    "model.gradient_checkpointing": True,
    "optim.learning_rate": 1e-5,
    "optim.lr_scheduler_type": "cosine",
    "optim.warmup_ratio": 0.03,
    "optim.weight_decay": 0.0,
    "optim.max_grad_norm": 1.0,
    "optim.n_epochs": 3,
    "optim.per_device_batch_size": 1,
    "optim.grad_accumulation_steps": 16,
    "optim.seed": 42,
    "eval.benchmarks": ["humanevalplus", "livecodebench"],
    "eval.n_samples_per_problem": 1,
    "eval.temperature": 0.0,
    "eval.top_p": 1.0,
    "eval.max_new_tokens": 4096,
    "eval.run_at_end": False,  # flip on once evaluation.py is implemented
    "logging.project": "coding_distill",
    "logging.run_name": "sft_baseline",
    "logging.tags": ["baseline", "codeforces-cots"],
    "logging.output_dir": "./out",
    "logging.save_steps": 500,
    "logging.logging_steps": 10,
    "logging.use_wandb": True,
}

# one axis for the first sweep -- the LR is what most often decides whether trace SFT
# helps or wrecks the base model
params_grid = {
    "optim.learning_rate": [5e-6, 1e-5, 2e-5],
}

experiments_list = create_experiments_helper(
    experiment_name="coding-distill-sft-baseline",
    project="coding_distill",
    script="python3 main.py",
    python_path=".:src",
    tags=["coding_distill", "sft", "baseline"],
    base_config=base_config,
    params_grid=params_grid,
    with_neptune=False,
    # keep the shipped tree small: mrunner copies the working dir to $HOME on the cluster
    exclude=[
        ".git",
        ".venv",
        ".pixi",
        "out",
        "outputs",
        "data",
        "wandb",
        "__pycache__",
        "*.pyc",
    ],
)
