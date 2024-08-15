import os
from pathlib import Path

ROOT_DIR = Path(os.getenv('TIMETRAVEL_ROOT', Path(__file__).resolve().parent.parent.parent))

MODEL_NAME = os.getenv('MODEL_NAME', "google/flan-t5-base")
if not MODEL_NAME:
    raise ValueError("MODEL_NAME environment variable is not set!")

CONFIG = {
    "root_dir": ROOT_DIR,
    "models_dir": ROOT_DIR / "models",
    "model_name": MODEL_NAME,
    "batch_size": int(os.getenv('BATCH_SIZE', 2)),
    "num_workers": int(os.getenv('NUM_WORKERS', 3)),
    "max_epochs": int(os.getenv('MAX_EPOCHS', 6)),
    "learning_rate": float(os.getenv('LEARNING_RATE', 2e-5)),
    "output_attentions": False,
    "max_length": 512,
    "max_gen_length": 250,
}
