from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

from src.metrics import ThroughputCallback


def train(model, tokenizer, train_dataset, eval_dataset, params, output_dir):
    """`output_dir` is resolved and created by main.py -- see src/run_dir.py."""
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
        use_liger_kernel=params.get("use_liger_kernel", False),
        # per-op flags: rope, rms_norm, swiglu, fused_linear_cross_entropy
        liger_kernel_config=params.get("liger_kernel_config", None),
        logging_steps=params.logging_steps,
        logging_first_step=params.get("logging_first_step", False),
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
    trainer.train()
    return trainer
