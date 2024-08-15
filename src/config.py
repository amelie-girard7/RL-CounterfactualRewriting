import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "root_dir": ROOT_DIR,
    "data_dir": ROOT_DIR / "data",
    "models_dir": ROOT_DIR / "results",
    "logs_dir": ROOT_DIR / "logs",
    
    # Model and training configurations
    "model_name": "google/flan-t5-base",  # Example model name
    "batch_size": 2,
    "num_workers": 3,
    "max_epochs": 6,
    "learning_rate": 2e-5,
    
    # preprocess data parameters
    "max_length": 512,

    # Text generation parameters
    "max_gen_length": 250,
}
