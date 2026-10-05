from dataclasses import dataclass
from typing import Optional

# No defaults on purpose: every value comes from the spec in configs/, so a
# missing key is a startup error instead of a silent fallback.
#
# The string under each field is its `--help` text (tyro reads field docstrings).


@dataclass
class ModelArgs:
    name: str
    """HF hub id or local path, used for both model and tokenizer."""
    pad_token: str
    """Set on the tokenizer; many LMs ship without one."""


@dataclass
class DataArgs:
    name: str
    """HF hub dataset id."""
    config: Optional[str]
    """Dataset config name; None for datasets that have only one."""
    split: str
    """Split with HF slicing, e.g. `train[:1%]`."""
    text_field: str
    """Column holding the raw text to tokenize."""
    max_length: int
    """Tokenizer max length, in tokens."""
    padding: str
    """Tokenizer padding mode, e.g. `max_length`."""
    truncation: bool
    """Cut samples longer than max_length."""


@dataclass
class TrainArgs:
    output_dir: str
    """Checkpoint base dir; $VARS expanded and $SLURM_JOB_ID appended at runtime."""
    learning_rate: float
    """Peak LR."""
    batch_size: int
    """Per-device batch size."""
    num_epochs: int
    """Epochs; ignored when max_steps > 0."""
    max_steps: int
    """Total optimizer steps; -1 to train for num_epochs."""
    logging_steps: int
    """Log every N steps."""
    save_steps: int
    """Checkpoint every N steps."""
    bf16: bool
    """bf16 mixed precision; off on CPU."""


@dataclass
class Args:
    model: ModelArgs
    data: DataArgs
    train: TrainArgs
