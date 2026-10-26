import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(params):
    # the tokenizer that built the dataset, which is not always the model's own
    tokenizer = AutoTokenizer.from_pretrained(params.get("tokenizer") or params.name)
    tokenizer.pad_token = params.pad_token  # explicit: many LMs ship without one
    # unset: the checkpoint's own dtype (bf16 for Qwen). "float32" gives the
    # optimizer fp32 master weights -- without them, an Adam step below half a bf16
    # ulp rounds to nothing, and 66% of a 7B run's weights never moved.
    dtype = getattr(torch, params.dtype) if params.get("dtype") else None
    model = AutoModelForCausalLM.from_pretrained(params.name, dtype=dtype)
    # without this the checkpoint can ignore the token it was trained to end with
    configured = model.generation_config.eos_token_id
    configured = configured if isinstance(configured, list) else [configured]
    if tokenizer.eos_token_id not in configured:
        model.generation_config.eos_token_id = [tokenizer.eos_token_id, *configured]
    return model, tokenizer
