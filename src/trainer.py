import os

from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from src.metrics import ThroughputCallback


def _resume_from(output_dir, params):
    """Pick up where a killed run left off.

    `resume_job_id` is the usual case: output_dir is keyed on SLURM_JOB_ID, so a
    resubmitted job gets a fresh directory and has to be told which previous
    one to continue. `resume_from_checkpoint` takes an explicit path instead.
    """
    if params.get("resume_from_checkpoint"):
        return params.resume_from_checkpoint

    job_id = params.get("resume_job_id")
    if not job_id:
        return None

    previous = os.path.join(os.path.dirname(output_dir), str(job_id))
    found = get_last_checkpoint(previous) if os.path.isdir(previous) else None
    if not found:
        raise SystemExit(f"resume_job_id={job_id}: no checkpoint under {previous}")
    print(f"resuming from {found}")
    return found


def train(model, tokenizer, train_dataset, eval_dataset, params, output_dir):
    """`output_dir` is resolved and created by main.py -- see src/paths.py."""
    report_to = params.get("report_to", "none")
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=params.learning_rate,
        per_device_train_batch_size=params.batch_size,
        gradient_accumulation_steps=params.get("gradient_accumulation_steps", 1),
        num_train_epochs=params.num_epochs,
        max_steps=params.max_steps,
        lr_scheduler_type=params.get("lr_scheduler_type", "linear"),
        lr_scheduler_kwargs=params.get("lr_scheduler_kwargs", None),
        warmup_steps=params.get("warmup_steps", 0),
        gradient_checkpointing=params.get("gradient_checkpointing", False),
        max_grad_norm=params.get("max_grad_norm", 1.0),
        use_liger_kernel=params.get("use_liger_kernel", False),
        # per-op flags: rope, rms_norm, swiglu, fused_linear_cross_entropy
        liger_kernel_config=params.get("liger_kernel_config", None),
        logging_steps=params.logging_steps,
        logging_first_step=params.get("logging_first_step", False),
        save_strategy=params.get("save_strategy", "steps"),
        save_steps=params.save_steps,
        save_total_limit=params.get("save_total_limit", None),
        eval_strategy=params.get("eval_strategy", "no"),
        eval_steps=params.get("eval_steps", None),
        per_device_eval_batch_size=params.get("eval_batch_size", params.batch_size),
        train_sampling_strategy=params.get("sampling_strategy", "random"),
        length_column_name="length",
        bf16=params.bf16,
        report_to=report_to,
        include_num_input_tokens_seen="non_padding",
        logging_nan_inf_filter=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # dynamic padding to the batch maximum; pads labels with -100
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        callbacks=[ThroughputCallback()] if "wandb" in report_to else [],
    )
    trainer.train(resume_from_checkpoint=_resume_from(output_dir, params))
    return trainer
