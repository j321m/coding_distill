from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from src.metrics import ThroughputCallback


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
        logging_steps=params.logging_steps,
        logging_first_step=params.logging_first_step,
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
    trainer.train()
    return trainer
