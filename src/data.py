from datasets import load_dataset


def build_dataset(params, tokenizer):
    raw = load_dataset(params.name, params.config, split=params.split)

    def tokenize(batch):
        ids = tokenizer(
            batch[params.text_field],
            truncation=params.truncation,
            padding=params.padding,
            max_length=params.max_length,
        )
        ids["labels"] = ids["input_ids"].copy()
        return ids

    return raw.map(tokenize, batched=True, remove_columns=raw.column_names)
