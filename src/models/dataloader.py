import torch
from torch.utils.data import Dataset, DataLoader

class MultiModalDataset(Dataset):
    """
    Dataset for multimodal inputs: text, tabular, (optional) vision, audio, time series.
    Each item is a dict with keys matching model inputs.
    """
    def __init__(self, data, tokenizer=None, max_text_len=128):
        """
        Args:
            data (list of dict): Each dict contains modalities and 'label'.
            tokenizer (callable, optional): Tokenizer for text data.
            max_text_len (int): Max length for text tokenization.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {}

        # Text
        if 'text' in item and self.tokenizer is not None:
            sample['text_inputs'] = self.tokenizer(
                item['text'],
                padding='max_length',
                truncation=True,
                max_length=self.max_text_len,
                return_tensors='pt'
            )
            # Remove batch dimension
            sample['text_inputs'] = {k: v.squeeze(0) for k, v in sample['text_inputs'].items()}
        elif 'text_inputs' in item:
            sample['text_inputs'] = item['text_inputs']

        # Tabular
        if 'tabular' in item:
            sample['tabular_inputs'] = torch.tensor(item['tabular'], dtype=torch.float)

        # Vision
        if 'vision_inputs' in item:
            sample['vision_inputs'] = item['vision_inputs']

        # Audio
        if 'audio_inputs' in item:
            sample['audio_inputs'] = torch.tensor(item['audio_inputs'], dtype=torch.float)

        # Time series
        if 'time_series_inputs' in item:
            sample['time_series_inputs'] = torch.tensor(item['time_series_inputs'], dtype=torch.float)

        # Label
        if 'label' in item:
            sample['label'] = torch.tensor(item['label'], dtype=torch.long)

        return sample

def multimodal_collate_fn(batch):
    """
    Collate function to merge a list of samples to batch.
    """
    batch_out = {}
    keys = batch[0].keys()
    for key in keys:
        if key == 'text_inputs':
            # Merge dict of tensors
            batch_out[key] = {k: torch.stack([b[key][k] for b in batch]) for k in batch[0][key]}
        elif key == 'label':
            batch_out[key] = torch.stack([b[key] for b in batch])
        else:
            batch_out[key] = torch.stack([b[key] for b in batch])
    return batch_out

def get_dataloader(data, tokenizer=None, batch_size=32, shuffle=True, max_text_len=128, num_workers=0):
    """
    Returns a DataLoader for multimodal data.
    """
    dataset = MultiModalDataset(data, tokenizer=tokenizer, max_text_len=max_text_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=multimodal_collate_fn,
        num_workers=num_workers
    )
