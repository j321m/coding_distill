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
        # Mask padding out of the loss. `attention_mask` is the only reliable
        # signal here: pad_token is usually the same id as eos, so masking by
        # token id would also drop the real end-of-sequence targets.
        ids["labels"] = [
            [tok if keep else -100 for tok, keep in zip(seq, mask)]
            for seq, mask in zip(ids["input_ids"], ids["attention_mask"])
        ]
        return ids

    return raw.map(tokenize, batched=True, remove_columns=raw.column_names)
