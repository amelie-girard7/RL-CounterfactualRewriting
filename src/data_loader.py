# src/data_loader.py
import json
from pathlib import Path
from torch.utils.data import Dataset

class StoryDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_datasets(config):
    train_data = StoryDataset(config["data_dir"] / "train.json")
    dev_data = StoryDataset(config["data_dir"] / "dev.json")
    test_data = StoryDataset(config["data_dir"] / "test.json")
    return train_data, dev_data, test_data
