# src/utils/preprocess.py

import torch
from src.config import CONFIG

def preprocess_data(row, tokenizer):
    """
    Prepares a single row of data for model input by tokenizing the text fields.
    """
    separator_token = "</s>"
    input_sequence = (
        f"{row['premise']}"
        f"{row['initial']}"
        f"{row['original_ending']} {separator_token} "
        f"{row['premise']} {row['counterfactual']}"
    )

    tokenized_inputs = tokenizer.encode_plus(
        input_sequence, truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
    )

    tokenized_ending = tokenizer.encode_plus(
        row['edited_ending'], truncation=True, return_tensors="pt", max_length=CONFIG["max_length"]
    )

    return {
        'input_ids': tokenized_inputs['input_ids'].squeeze(0),
        'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
        'labels': tokenized_ending['input_ids'].squeeze(0),
    }
