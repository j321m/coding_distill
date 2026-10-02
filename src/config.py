from dataclasses import dataclass
from typing import Optional

# No defaults on purpose: every value comes from the spec in configs/, so a
# missing key is a startup error instead of a silent fallback.


@dataclass
class ModelArgs:
    name: str
    pad_token: str


@dataclass
class DataArgs:
    name: str
    config: Optional[str]
    split: str
    text_field: str
    max_length: int
    padding: str
    truncation: bool


@dataclass
class TrainArgs:
    output_dir: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    max_steps: int
    logging_steps: int
    save_steps: int
    bf16: bool


@dataclass
class Args:
    model: ModelArgs
    data: DataArgs
    train: TrainArgs
