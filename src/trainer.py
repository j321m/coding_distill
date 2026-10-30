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
    if params.resume_from_checkpoint:
        return params.resume_from_checkpoint

    job_id = params.resume_job_id
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
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=params.learning_rate,
        per_device_train_batch_size=params.batch_size,
        gradient_accumulation_steps=params.gradient_accumulation_steps,
        num_train_epochs=params.num_epochs,
        max_steps=params.max_steps,
        lr_scheduler_type=params.lr_scheduler_type,
        lr_scheduler_kwargs=params.lr_scheduler_kwargs,
        warmup_steps=params.warmup_steps,
        gradient_checkpointing=params.gradient_checkpointing,
        use_liger_kernel=params.use_liger_kernel,
        fsdp=params.fsdp,
        fsdp_config=params.fsdp_config,
        logging_steps=params.logging_steps,
        logging_first_step=params.logging_first_step,
        max_grad_norm=params.max_grad_norm,
        save_strategy=params.save_strategy,
        save_steps=params.save_steps,
        save_total_limit=params.save_total_limit,
        eval_strategy=params.eval_strategy,
        eval_steps=params.eval_steps,
        per_device_eval_batch_size=params.eval_batch_size,
        train_sampling_strategy=params.sampling_strategy,
        length_column_name="length",
        bf16=params.bf16,
        report_to=params.report_to,
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
        callbacks=[ThroughputCallback()] if params.report_to == "wandb" else [],
    )
    trainer.train(resume_from_checkpoint=_resume_from(output_dir, params))
    return trainer
