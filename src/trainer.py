from transformers import Trainer, TrainingArguments


def train(model, tokenizer, dataset, params):
    args = TrainingArguments(
        output_dir=params.output_dir,
        learning_rate=params.learning_rate,
        per_device_train_batch_size=params.batch_size,
        num_train_epochs=params.num_epochs,
        max_steps=params.max_steps,
        logging_steps=params.logging_steps,
        save_steps=params.save_steps,
        bf16=params.bf16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    return trainer
