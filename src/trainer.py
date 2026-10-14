import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Trainer, TrainingArguments

from src.data import make_collator

logger = logging.getLogger(__name__)


def resolve_output_dir(output_dir):
    """`output_dir` is a base shared by every run, so give each job its own
    subdirectory. mrunner gives runs a unique cwd for free, but checkpoints live
    outside the copied repo, so uniqueness has to come from the job id."""
    run_id = os.environ.get("SLURM_JOB_ID", "local")  # unique per array task
    return os.path.join(os.path.expandvars(output_dir), run_id)


def _chunk_ce(hidden, labels, weight):
    logits = F.linear(hidden, weight.to(hidden.dtype)).float()
    return F.cross_entropy(logits, labels, reduction="sum")


def chunked_cross_entropy(hidden, labels, weight, chunk_size):
    """Summed next-token CE without the full `seq x vocab` logits: logits are
    built `chunk_size` tokens at a time and recomputed in backward."""
    hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    labels = labels[:, 1:].reshape(-1)
    keep = labels != -100  # prompt and padding cost nothing
    hidden, labels = hidden[keep], labels[keep]

    total = hidden.new_zeros((), dtype=torch.float32)
    for h, y in zip(hidden.split(chunk_size), labels.split(chunk_size)):
        total = total + checkpoint(_chunk_ce, h, y, weight, use_reentrant=False)
    return total, len(labels)


class ChunkedLossTrainer(Trainer):
    """Trainer whose loss is `chunked_cross_entropy` on the last hidden state."""

    def __init__(self, *args, loss_chunk_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_chunk_size = loss_chunk_size

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        # through the DDP wrapper; logits only for the last token (unused),
        # `hidden_states[-1]` is the final normed hidden state
        outputs = model(**inputs, logits_to_keep=1, output_hidden_states=True)
        weight = self.accelerator.unwrap_model(model).get_output_embeddings().weight
        loss, n_tokens = chunked_cross_entropy(
            outputs.hidden_states[-1], labels, weight, self.loss_chunk_size
        )

        # same normalization as HF's default loss: by the label count of the
        # whole (accumulated, cross-device) batch, scaled back up by world size
        if num_items_in_batch is None:
            return loss / n_tokens
        loss = loss / num_items_in_batch
        if self.args.average_tokens_across_devices:
            loss = loss * self.accelerator.num_processes
        return loss


def train(model, tokenizer, dataset, params):
    output_dir = resolve_output_dir(params.output_dir)
    logger.info(f"output_dir: {output_dir}")

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=params.learning_rate,
        per_device_train_batch_size=params.batch_size,
        num_train_epochs=params.num_epochs,
        max_steps=params.max_steps,
        logging_steps=params.logging_steps,
        save_steps=params.save_steps,
        bf16=params.bf16,
        use_liger_kernel=params.loss == "liger",
    )

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=make_collator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    if params.loss == "chunked":
        trainer = ChunkedLossTrainer(
            **trainer_kwargs, loss_chunk_size=params.loss_chunk_size
        )
    else:
        trainer = Trainer(**trainer_kwargs)
    trainer.train()
    return trainer
