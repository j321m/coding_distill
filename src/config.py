from dataclasses import dataclass
from typing import Literal, Optional

# One top-level dataclass per entry point (`MainArgs`, `TokenizeArgs`,
# `EvalArgs`), each parsed with `tyro.cli` from the flags mrunner_run.py builds.
#
# No defaults on purpose: every value comes from the spec in configs/, so a
# missing key is a startup error instead of a silent fallback.
#
# The string under each field is its `--help` text (tyro reads field docstrings).


@dataclass
class ModelArgs:
    name: str
    """HF hub id or local path of the model."""
    tokenizer: Optional[str]
    """Tokenizer to load instead of the model's own: the one that built the
    dataset. None = the model's."""
    pad_token: str
    """Set on the tokenizer; many LMs ship without one."""
    dtype: Optional[str]
    """torch dtype to load the weights in, e.g. `float32` for fp32 master
    weights under FSDP; None = the checkpoint's own."""


@dataclass
class DataArgs:
    """A pre-tokenized artifact: what scripts/tokenize_dataset.py builds and
    main.py trains on. The identity fields must match the artifact's meta.json
    (src/dataset.py)."""

    dataset: str
    """HF hub dataset id the artifact was built from."""
    config: str
    """Dataset config name."""
    split: str
    """Dataset split that was tokenized."""
    tokenizer: str
    """Tokenizer that built the artifact."""
    max_length: int
    """Longest allowed sample, in tokens."""
    drop_finish_reason: Optional[str]
    """`finish_reason` dropped at tokenize time; recorded, not checked."""
    test_size: float
    """Fraction held out as the eval split."""
    split_seed: int
    """Seed of the train/eval split."""
    data_dir: str
    """Base dir of the artifacts; $VARS expanded at runtime."""
    filter_max_tokens: Optional[int]
    """Drop rows whose R1 token count exceeds this; recorded, not checked."""
    filter_n: Optional[int]
    """Row cap of a debug artifact; None for the full one."""


@dataclass
class TrainArgs:
    output_dir: str
    """Run base dir; $VARS expanded and $SLURM_JOB_ID appended at runtime."""
    learning_rate: float
    """Peak LR."""
    lr_scheduler_type: str
    """HF scheduler name, e.g. `linear`, `cosine_with_min_lr`."""
    lr_scheduler_kwargs: Optional[dict[str, float]]
    """Extra scheduler args, e.g. `min_lr_rate 0.1`."""
    warmup_steps: float
    """Warmup steps; below 1 = fraction of total steps."""
    batch_size: int
    """Per-device train batch size."""
    gradient_accumulation_steps: int
    """Micro-batches per optimizer step."""
    num_epochs: int
    """Epochs; ignored when max_steps > 0."""
    max_steps: int
    """Total optimizer steps; -1 to train for num_epochs."""
    max_grad_norm: float
    """Gradient clipping threshold; HF's default is 1.0."""
    gradient_checkpointing: bool
    """Recompute activations in backward."""
    sampling_strategy: Literal["random", "group_by_length"]
    """Train sampler; `group_by_length` reads the dataset's `length` column."""
    use_liger_kernel: bool
    """Patch the model with Liger's default kernels (CUDA only)."""
    fsdp: Optional[str]
    """FSDP options, e.g. `full_shard auto_wrap`: shards weights, grads and
    optimizer state over every process on every node. None = DDP."""
    fsdp_config: Optional[dict[str, str]]
    """FSDP settings, e.g. `transformer_layer_cls_to_wrap Qwen2DecoderLayer`."""
    bf16: bool
    """bf16 mixed precision; off on CPU."""
    logging_steps: int
    """Log every N steps."""
    logging_first_step: bool
    """Also log step 1."""
    eval_strategy: Literal["no", "steps", "epoch"]
    """When to run the eval split."""
    eval_steps: Optional[int]
    """Eval every N steps, for `eval_strategy=steps`."""
    eval_batch_size: int
    """Per-device eval batch size."""
    save_strategy: Literal["no", "steps", "epoch"]
    """When to checkpoint."""
    save_steps: int
    """Checkpoint every N steps, for `save_strategy=steps`."""
    save_total_limit: Optional[int]
    """Checkpoints kept; None = all."""
    resume_job_id: Optional[str]
    """SLURM job id of a killed run: continue from its newest checkpoint under
    `output_dir`'s parent (optimizer and lr schedule included)."""
    resume_from_checkpoint: Optional[str]
    """Explicit checkpoint path to continue from; overrides `resume_job_id`."""
    report_to: Literal["wandb", "none"]
    """Metrics sink."""


# entry module -> one process per GPU. Only training: helios grants every job
# 4 GPUs, but tokenize runs on CPU and eval shards through vLLM itself.
ENTRY_POINTS = {
    "main": True,
    "scripts.tokenize_dataset": False,
    "scripts.eval_lcb": False,
}


@dataclass
class MainArgs:
    """main.py: fine-tune a model on a pre-tokenized artifact."""

    experiment_name: str
    """The spec's name in configs/, set by make_experiment; names the wandb run."""
    model: ModelArgs
    data: DataArgs
    train: TrainArgs


@dataclass
class TokenizeArgs:
    """scripts/tokenize_dataset.py: build the artifact main.py reads."""

    experiment_name: str
    """The spec's name in configs/, set by make_experiment; unused here."""
    data: DataArgs
    num_proc: int
    """Worker processes for the filter/map passes."""


@dataclass
class EvalArgs:
    """scripts/eval_lcb.py: score a checkpoint on LiveCodeBench."""

    experiment_name: str
    """The spec's name in configs/, set by make_experiment; names the wandb run."""
    train_job_id: str
    """Training run under $RUNS_DIR; also where results land."""
    checkpoint: Optional[str]
    """Checkpoint dir inside the run; None when `model_path` is set."""
    model_path: Optional[str]
    """HF hub id to score instead of a local checkpoint."""
    model_name: Optional[str]
    """Name of the results dir; None = coding-distill-<train_job_id>."""
    scenario: str
    """LiveCodeBench scenario, e.g. `codegeneration`."""
    release_version: str
    """LiveCodeBench problem release, e.g. `release_v6`."""
    start_date: str
    """First contest date of the problem window, YYYY-MM-DD."""
    end_date: str
    """Last contest date of the problem window, YYYY-MM-DD."""
    n: int
    """Samples per problem."""
    temperature: float
    """Sampling temperature."""
    top_p: float
    """Nucleus sampling threshold."""
    max_tokens: int
    """Generation cap per sample."""
    dtype: str
    """vLLM weight dtype, e.g. `bfloat16`."""
    tensor_parallel_size: int
    """GPUs vLLM shards the model over."""
    report_to: Literal["wandb", "none"]
    """Metrics sink."""
