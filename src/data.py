import torch
from datasets import load_dataset


def tokenize_chat(messages, tokenizer):
    """One sample, ragged: no padding, no truncation. Loss only on the
    assistant turn; the prompt (`messages[:1]`, rendered with the assistant
    header) gets label -100."""
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True
    )
    # the chat template already has the special tokens (eos = <|im_end|>)
    input_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    assert input_ids[: len(prompt_ids)] == prompt_ids, "prompt is not a prefix"

    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    return {"input_ids": input_ids, "labels": labels}


def build_dataset(params, tokenizer):
    raw = load_dataset(params.name, params.config, split=params.split)
    # traces cut off by the generator's token limit have no final answer
    raw = raw.filter(lambda r: r["finish_reason"] != "length")

    dataset = raw.map(
        lambda r: tokenize_chat(r["messages"], tokenizer),
        remove_columns=raw.column_names,
    )
    longest = max(len(ids) for ids in dataset["input_ids"])
    assert longest <= params.max_length, f"{longest} > max_length"
    return dataset


def make_collator(pad_token_id):
    """Pad each batch to its longest sample. The attention mask comes from the
    true lengths, not from token ids, so a pad id that also occurs in the text
    is still attended to."""

    def collate(batch):
        lengths = [len(x["input_ids"]) for x in batch]
        max_len = max(lengths)
        input_ids = torch.full((len(batch), max_len), pad_token_id)
        labels = torch.full((len(batch), max_len), -100)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, (x, n) in enumerate(zip(batch, lengths)):
            input_ids[i, :n] = torch.tensor(x["input_ids"])
            labels[i, :n] = torch.tensor(x["labels"])
            attention_mask[i, :n] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate
