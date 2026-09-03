import os

from transformers import Trainer, TrainingArguments


def resolve_output_dir(output_dir):
    """`output_dir` is a base shared by every run, so give each job its own
    subdirectory. mrunner gives runs a unique cwd for free, but checkpoints live
    outside the copied repo, so uniqueness has to come from the job id."""
    run_id = os.environ.get("SLURM_JOB_ID", "local")  # unique per array task
    return os.path.join(os.path.expandvars(output_dir), run_id)


def train(model, tokenizer, dataset, params):
    output_dir = resolve_output_dir(params.output_dir)
    print(f"output_dir: {output_dir}")

    args = TrainingArguments(
        output_dir=output_dir,
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
