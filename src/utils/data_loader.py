import json
from src.utils.utils import preprocess_data

def load_data(file_path, tokenizer):
    """
    Load and preprocess the dataset for initializing the environment.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            processed_row = preprocess_data(row, tokenizer)
            data.append(processed_row)
    return data

def create_dataloaders(data_dir, tokenizer, batch_size, num_workers):
    """
    Create dataloaders for initial data loading in the RL environment.
    """
    from torch.utils.data import DataLoader, Dataset

    class StoryDataset(Dataset):
        def __init__(self, data_file):
            self.data = list(load_data(data_file, tokenizer))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_data = StoryDataset(data_dir / CONFIG["train_file"])
    dev_data = StoryDataset(data_dir / CONFIG["dev_file"])
    test_data = StoryDataset(data_dir / CONFIG["test_file"])

    return {
        'train': DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True),
        'dev': DataLoader(dev_data, batch_size=batch_size, num_workers=num_workers),
        'test': DataLoader(test_data, batch_size=batch_size, num_workers=num_workers),
    }
