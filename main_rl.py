import os
import datetime
import logging
import torch
import sys
from pathlib import Path
from transformers import T5Tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.plugins import MixedPrecisionPlugin
from src.models.model_T5 import FlanT5FineTuner
from src.utils.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model(model_dir):
    model = FlanT5FineTuner(CONFIG["model_name"], model_dir)
    return model

def setup_trainer(model_dir):
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    
    tensorboard_logger = TensorBoardLogger(save_dir=model_dir, name="training_logs")
    csv_logger = CSVLogger(save_dir=model_dir, name="csv_logs")

    trainer = Trainer(
        max_epochs=CONFIG["max_epochs"],
        accelerator='gpu',
        devices=1,
        callbacks=[checkpoint_callback],
        logger=[tensorboard_logger, csv_logger],
        precision=16,  # Mixed precision training
        plugins=[MixedPrecisionPlugin()],
    )
    return trainer

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    try:
        model_timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        model_dir = CONFIG["models_dir"] / f"model_{model_timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        tokenizer = T5Tokenizer.from_pretrained(CONFIG["model_name"], legacy=False)
        model = setup_model(model_dir)
        trainer = setup_trainer(model_dir)

        trainer.fit(model)
    except Exception as e:
        logger.exception("An unexpected error occurred during the process.")
        sys.exit(1)  # Exit with a failure code
    finally:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
