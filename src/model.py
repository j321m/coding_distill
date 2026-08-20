from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(params):
    tokenizer = AutoTokenizer.from_pretrained(params.name)
    tokenizer.pad_token = params.pad_token  # explicit: many LMs ship without one
    model = AutoModelForCausalLM.from_pretrained(params.name)
    return model, tokenizer
