from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(params):
    # the tokenizer that built the dataset, which is not always the model's own
    tokenizer = AutoTokenizer.from_pretrained(params.get("tokenizer") or params.name)
    tokenizer.pad_token = params.pad_token  # explicit: many LMs ship without one
    model = AutoModelForCausalLM.from_pretrained(params.name)
    # without this the checkpoint can ignore the token it was trained to end with
    configured = model.generation_config.eos_token_id
    configured = configured if isinstance(configured, list) else [configured]
    if tokenizer.eos_token_id not in configured:
        model.generation_config.eos_token_id = [tokenizer.eos_token_id, *configured]
    return model, tokenizer
