from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(params):
    tokenizer = AutoTokenizer.from_pretrained(params.name)
    assert tokenizer.chat_template, f"{params.name} has no chat template"
    assert tokenizer.pad_token_id is not None, f"{params.name} has no pad token"
    model = AutoModelForCausalLM.from_pretrained(params.name)
    return model, tokenizer
