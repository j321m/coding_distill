from dataclasses import dataclass
from typing import Literal, Optional

# No defaults on purpose: every value comes from the spec in configs/, so a
# missing key is a startup error instead of a silent fallback.
#
# The string under each field is its `--help` text (tyro reads field docstrings).


@dataclass
class ModelArgs:
    name: str
    """HF hub id or local path, used for both model and tokenizer; needs a chat
    template and a pad token."""


@dataclass
class DataArgs:
    name: str
    """HF hub dataset id; rows have `messages` (user, assistant) and
    `finish_reason`."""
    config: Optional[str]
    """Dataset config name; None for datasets that have only one."""
    split: str
    """Split with HF slicing, e.g. `train[:1%]`."""
    max_length: int
    """Longest allowed sample, in tokens; a longer one is an error."""


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
    loss: Literal["full", "liger", "chunked"]
    """Next-token CE: `full` materializes `seq x vocab` logits (HF default);
    `liger` uses Liger's fused linear+CE kernel (GPU only); `chunked` is our
    own chunked CE (`src/trainer.py`)."""
    loss_chunk_size: int
    """Tokens per logits chunk for `loss=chunked`."""


@dataclass
class Args:
    model: ModelArgs
    data: DataArgs
    train: TrainArgs
