import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from config import Config
from typing import List, Tuple

class PromptNER(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name)
        
        self.bert = BertModel.from_pretrained(config.model_name)
        self.hidden = nn.Linear(self.bert.config.hidden_size, 256)
        self.dropout = nn.Dropout(0.1)
        self.type_classifier = nn.Linear(256, len(config.entity_types))
        self.criterion = nn.CrossEntropyLoss()
        
        self.prompt_templates = {
            'product': '这是一个产品：',
            'region': '这是一个地区：',
            'time': '这是一个时间：',
            'metric': '这是一个指标：',
            'company': '这是一个公司：',
            'currency': '这是一个货币：'
        }
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0]
        hidden_output = self.hidden(cls_output)
        hidden_output = self.dropout(hidden_output)
        logits = self.type_classifier(hidden_output)
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            return loss
        else:
            return logits
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        self.eval()
        with torch.no_grad():
            entities = []
            
            for entity_type in self.config.entity_types:
                prompt = self.prompt_templates[entity_type]
                full_text = prompt + text
                
                encoding = self.tokenizer(
                    full_text,
                    max_length=self.config.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                logits = self(encoding['input_ids'], encoding['attention_mask'])
                probs = torch.softmax(logits, dim=-1)
                
                if probs[0, self.config.entity_types.index(entity_type)] > 0.5:
                    entities.append((text, entity_type))
            
            return entities
    
    def get_prompt_embeddings(self, text: str) -> torch.Tensor:
        prompt_embeddings = []
        
        for entity_type in self.config.entity_types:
            prompt = self.prompt_templates[entity_type]
            full_text = prompt + text
            
            encoding = self.tokenizer(
                full_text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.bert(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
                cls_output = outputs[0][:, 0]
                prompt_embeddings.append(cls_output)
        
        return torch.stack(prompt_embeddings)
    
    def get_entity_embeddings(self, text: str, entity_type: str) -> torch.Tensor:
        prompt = self.prompt_templates[entity_type]
        full_text = prompt + text
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.bert(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            return outputs[0][:, 0] 