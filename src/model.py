import torch
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, T5Tokenizer
from src.config import CONFIG

class StoryGenerationModel(pl.LightningModule):
    def __init__(self, model_name_or_path):
        super(StoryGenerationModel, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)

    def forward(self, input_text):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=CONFIG["max_length"], truncation=True)
        output_ids = self.model.generate(input_ids, max_length=CONFIG["max_gen_length"])
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text

    def configure_optimizers(self):
        # Placeholder for RL, as optimization might be handled differently
        return None
