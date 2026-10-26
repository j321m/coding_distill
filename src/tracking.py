"""wandb run setup."""

import os


def init_wandb(metadata):
    """Start the run before training, so its config survives a crash in loading."""
    if os.environ.get("RANK", "0") != "0":  # torchrun runs this on every rank
        return None

    import wandb

    run = wandb.init(
        # `sft_7b-29047`: the spec it ran, and the SLURM job -- the run dir for
        # training, the job's log for an eval
        name=f"{metadata['params']['experiment_name']}-"
        f"{metadata['job']['SLURM_JOB_ID'] or 'local'}",
        config={key: metadata[key] for key in ("params", "job", "git", "versions")},
    )
    wandb.define_metric("train/num_input_tokens_seen")
    wandb.define_metric("train/*", step_metric="train/num_input_tokens_seen")

    output_dir = metadata["output_dir"]
    for name in ("args.json", metadata["git"]["patch"]):
        if name and os.path.exists(os.path.join(output_dir, name)):
            run.save(os.path.join(output_dir, name), base_path=output_dir, policy="now")
    return run
