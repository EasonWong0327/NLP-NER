import os
import json
from typing import List, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class NERDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: BertTokenizer, config: Config):
        self.tokenizer = tokenizer
        self.config = config
        self.examples = self._load_data(data_path)
        
    def _load_data(self, data_path: str) -> List[Dict]:
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                examples.append(example)
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        text = example['text']
        labels = example['labels']
        
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        start_labels, end_labels, type_labels = self._process_labels(labels, encoding['input_ids'][0])
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_labels': torch.tensor(start_labels),
            'end_labels': torch.tensor(end_labels),
            'type_labels': torch.tensor(type_labels)
        }
    
    def _process_labels(self, labels: List[str], input_ids: torch.Tensor) -> Tuple[List[float], List[float], List[int]]:
        start_labels = [0.0] * self.config.max_length
        end_labels = [0.0] * self.config.max_length
        type_labels = [0] * self.config.max_length
        
        current_pos = 0
        current_entity = None
        current_start = None
        
        for i, label in enumerate(labels):
            if label.startswith('B-'):
                if current_entity is not None:
                    end_labels[current_start] = 1.0
                current_entity = label[2:]
                current_start = i
                start_labels[i] = 1.0
                type_labels[i] = self.config.label2id[label]
            elif label.startswith('I-'):
                if current_entity == label[2:]:
                    type_labels[i] = self.config.label2id[label]
                else:
                    if current_entity is not None:
                        end_labels[current_start] = 1.0
                    current_entity = label[2:]
                    current_start = i
                    start_labels[i] = 1.0
                    type_labels[i] = self.config.label2id[label]
            elif label == 'O':
                if current_entity is not None:
                    end_labels[current_start] = 1.0
                    current_entity = None
                    current_start = None
                type_labels[i] = self.config.label2id[label]
        
        if current_entity is not None:
            end_labels[current_start] = 1.0
        
        return start_labels, end_labels, type_labels

def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    train_dataset = NERDataset(
        os.path.join(config.processed_data_dir, config.train_file),
        tokenizer,
        config
    )
    dev_dataset = NERDataset(
        os.path.join(config.processed_data_dir, config.dev_file),
        tokenizer,
        config
    )
    test_dataset = NERDataset(
        os.path.join(config.processed_data_dir, config.test_file),
        tokenizer,
        config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, dev_loader, test_loader

def process_raw_data(raw_data_path: str, output_path: str):
    processed_data = []
    
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            processed_example = {
                'text': data['text'],
                'labels': data['labels']
            }
            processed_data.append(processed_example)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in processed_data:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    config = Config()
    os.makedirs(config.processed_data_dir, exist_ok=True)
    
    for split in ['train', 'dev', 'test']:
        raw_path = os.path.join(config.raw_data_dir, f'{split}.txt')
        processed_path = os.path.join(config.processed_data_dir, f'{split}.txt')
        process_raw_data(raw_path, processed_path) 