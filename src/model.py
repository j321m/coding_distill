from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(params):
    # the tokenizer that built the dataset, which is not always the model's own:
    # a base model can share a vocabulary while declaring different special
    # tokens, and the data is the authority on those
    tokenizer = AutoTokenizer.from_pretrained(params.get("tokenizer") or params.name)
    tokenizer.pad_token = params.pad_token  # explicit: many LMs ship without one
    model = AutoModelForCausalLM.from_pretrained(params.name)
    # The tokenizer's eos has to be among the ids generation stops on, or the
    # checkpoint ignores the very token it was trained to end with.
    configured = model.generation_config.eos_token_id
    configured = configured if isinstance(configured, list) else [configured]
    if tokenizer.eos_token_id not in configured:
        model.generation_config.eos_token_id = [tokenizer.eos_token_id, *configured]
    return model, tokenizer
