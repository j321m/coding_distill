from transformers import Trainer, TrainingArguments

from src.metrics import ThroughputCallback


def train(model, tokenizer, dataset, params, output_dir):
    """`output_dir` is resolved and created by main.py -- see src/run_dir.py."""
    report_to = params.get("report_to", "none")
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=params.learning_rate,
        per_device_train_batch_size=params.batch_size,
        num_train_epochs=params.num_epochs,
        max_steps=params.max_steps,
        logging_steps=params.logging_steps,
        save_steps=params.save_steps,
        bf16=params.bf16,
        report_to=report_to,
        include_num_input_tokens_seen="non_padding",
        logging_nan_inf_filter=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[ThroughputCallback()] if "wandb" in report_to else [],
    )
    trainer.train()
    return trainer
