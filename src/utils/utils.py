import torch
from src.utils.config import CONFIG

def calculate_differential_weights(tokenized_labels, tokenizer, differences, high_weight=13, base_weight=1):
    """
    Calculate differential weights for tokenized labels based on differences.
    """
    differential_weights = torch.full(tokenized_labels.shape, fill_value=base_weight, dtype=torch.float)
    difference_tokens_ids = set([item for sublist in [tokenizer.encode(diff, add_special_tokens=False) for diff in differences] for item in sublist])

    for i, token_id in enumerate(tokenized_labels.squeeze().tolist()):
        if token_id in difference_tokens_ids:
            differential_weights[i] = high_weight
        
    return differential_weights    

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
    
    differential_weights = calculate_differential_weights(
        tokenized_ending['input_ids'].squeeze(), tokenizer, row['differences']
    )

    return {
        'input_ids': tokenized_inputs['input_ids'].squeeze(0),
        'attention_mask': tokenized_inputs['attention_mask'].squeeze(0),
        'labels': tokenized_ending['input_ids'].squeeze(0),
        'differential_weights': differential_weights.squeeze(0),
    }
