"""Experiment configuration.

Plain attrs classes, no defaults: every field must be set explicitly by a
config file or an mrunner spec, so a run is always fully described by its params.
"""

from typing import Optional

from attr import define, fields


@define
class DataConfig:
    dataset_name: str
    dataset_config: str  # codeforces-cots subset, e.g. "solutions_py"
    split: str
    max_seq_length: int
    n_train_samples: Optional[int]  # None = all; small int for smoke tests
    n_eval_samples: Optional[int]
    mask_prompt: bool  # loss on completion tokens only
    num_proc: int


@define
class ModelConfig:
    student_name: str  # HF id of the base model being distilled into
    attn_implementation: str
    torch_dtype: str
    gradient_checkpointing: bool


@define
class OptimConfig:
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    n_epochs: float
    per_device_batch_size: int
    grad_accumulation_steps: int
    seed: int


@define
class EvalConfig:
    benchmarks: list  # e.g. ["humanevalplus", "livecodebench"]
    n_samples_per_problem: int  # >1 for pass@k
    temperature: float
    top_p: float
    max_new_tokens: int
    run_at_end: bool


@define
class LoggingConfig:
    project: str
    run_name: str
    tags: list
    output_dir: str
    save_steps: int
    logging_steps: int
    use_wandb: bool


@define
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    eval: EvalConfig
    logging: LoggingConfig


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "optim": OptimConfig,
    "eval": EvalConfig,
    "logging": LoggingConfig,
}


def config_from_params(params: dict) -> ExperimentConfig:
    """Build a config from a flat dict of `section.field` keys.

    mrunner hands the script a flat params dict, so the grid can vary
    `optim.learning_rate` directly without nesting.
    """
    nested = {section: {} for section in _SECTIONS}
    unknown = []
    for key, value in params.items():
        section, _, field = key.partition(".")
        if section in nested and field:
            nested[section][field] = value
        else:
            unknown.append(key)

    sections = {}
    for name, cls in _SECTIONS.items():
        expected = {f.name for f in fields(cls)}
        got = set(nested[name])
        missing, extra = expected - got, got - expected
        if missing or extra:
            raise ValueError(
                f"section '{name}': missing={sorted(missing)} unexpected={sorted(extra)}"
            )
        sections[name] = cls(**nested[name])

    if unknown:
        # mrunner injects its own bookkeeping keys; surface them rather than hide them
        print(f"[config] ignoring non-config params: {sorted(unknown)}")
    return ExperimentConfig(**sections)
