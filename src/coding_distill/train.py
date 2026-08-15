"""SFT distillation on teacher reasoning traces."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from coding_distill.data import build_dataset

logger = logging.getLogger(__name__)


def _sft_config(cfg) -> SFTConfig:
    return SFTConfig(
        output_dir=cfg.logging.output_dir,
        run_name=cfg.logging.run_name,
        report_to=["wandb"] if cfg.logging.use_wandb else [],
        learning_rate=cfg.optim.learning_rate,
        lr_scheduler_type=cfg.optim.lr_scheduler_type,
        warmup_ratio=cfg.optim.warmup_ratio,
        weight_decay=cfg.optim.weight_decay,
        max_grad_norm=cfg.optim.max_grad_norm,
        num_train_epochs=cfg.optim.n_epochs,
        per_device_train_batch_size=cfg.optim.per_device_batch_size,
        gradient_accumulation_steps=cfg.optim.grad_accumulation_steps,
        gradient_checkpointing=cfg.model.gradient_checkpointing,
        max_length=cfg.data.max_seq_length,
        seed=cfg.optim.seed,
        save_steps=cfg.logging.save_steps,
        logging_steps=cfg.logging.logging_steps,
        bf16=torch.cuda.is_available(),
    )


def run(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.student_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(cfg.data, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.student_name,
        attn_implementation=cfg.model.attn_implementation,
        dtype=getattr(torch, cfg.model.torch_dtype),
    )

    trainer = SFTTrainer(
        model=model,
        args=_sft_config(cfg),
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(cfg.logging.output_dir)
    tokenizer.save_pretrained(cfg.logging.output_dir)
    logger.info(f"saved student to {cfg.logging.output_dir}")

    if cfg.eval.run_at_end:
        from coding_distill.evaluation import evaluate

        results = evaluate(cfg.logging.output_dir, cfg.eval)
        logger.info(f"eval results: {results}")
        if cfg.logging.use_wandb:
            import wandb

            if wandb.run is not None:
                wandb.run.log(results)

    return cfg.logging.output_dir
