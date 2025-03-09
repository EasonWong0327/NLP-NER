import torch
import torch.nn as nn
from transformers import BertModel
from config import Config
from typing import List, Tuple

class TransformerNER(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.bert = BertModel.from_pretrained(config.model_name)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.bert.config.hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=6
        )
        
        self.hidden = nn.Linear(self.bert.config.hidden_size, 256)
        self.dropout = nn.Dropout(0.1)
        self.type_classifier = nn.Linear(256, len(config.label2id))
        self.start_classifier = nn.Linear(256, 1)
        self.end_classifier = nn.Linear(256, 1)
        
        self.type_criterion = nn.CrossEntropyLoss()
        self.boundary_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, input_ids, attention_mask, start_labels=None, end_labels=None, type_labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]
        sequence_output = self.transformer(sequence_output.transpose(0, 1)).transpose(0, 1)
        sequence_output = self.hidden(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)
        type_logits = self.type_classifier(sequence_output)
        
        if start_labels is not None and end_labels is not None and type_labels is not None:
            start_loss = self.boundary_criterion(start_logits, start_labels.float())
            end_loss = self.boundary_criterion(end_logits, end_labels.float())
            type_loss = self.type_criterion(type_logits.view(-1, len(self.config.label2id)), 
                                         type_labels.view(-1))
            
            total_loss = start_loss + end_loss + type_loss
            return total_loss
        else:
            return start_logits, end_logits, type_logits
    
    def predict(self, input_ids, attention_mask) -> List[Tuple[int, int, str]]:
        self.eval()
        with torch.no_grad():
            start_logits, end_logits, type_logits = self(input_ids, attention_mask)
            batch_size = input_ids.size(0)
            entities = []
            
            for b in range(batch_size):
                start_probs = torch.sigmoid(start_logits[b])
                end_probs = torch.sigmoid(end_logits[b])
                type_pred = torch.argmax(type_logits[b], dim=-1)
                
                start_pred = start_probs > 0.3
                end_pred = end_probs > 0.3
                
                current_start = None
                for i in range(len(start_pred)):
                    if start_pred[i] and current_start is None:
                        current_start = i
                    elif end_pred[i] and current_start is not None:
                        entity_type = self.config.id2label[type_pred[i].item()]
                        if entity_type != 'O':
                            entity_type = entity_type[2:]
                            entities.append((current_start, i + 1, entity_type))
                        current_start = None
            
            return entities
    
    def get_attention_weights(self, input_ids, attention_mask) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            sequence_output = outputs[0]
            sequence_output = sequence_output.transpose(0, 1)
            attention_weights = []
            
            for layer in self.transformer.layers:
                attn_output, attn_weights = layer.self_attn(
                    sequence_output, sequence_output, sequence_output
                )
                attention_weights.append(attn_weights)
                
                sequence_output = layer.linear1(attn_output)
                sequence_output = layer.activation(sequence_output)
                sequence_output = layer.linear2(sequence_output)
                sequence_output = layer.dropout(sequence_output)
            
            return torch.stack(attention_weights)
    
    def get_entity_embeddings(self, sequence_output: torch.Tensor, 
                            start_positions: torch.Tensor, 
                            end_positions: torch.Tensor) -> torch.Tensor:
        batch_size = sequence_output.size(0)
        entity_embeddings = []
        
        for b in range(batch_size):
            start = start_positions[b]
            end = end_positions[b]
            span_embedding = torch.mean(sequence_output[b, start:end], dim=0)
            entity_embeddings.append(span_embedding)
        
        return torch.stack(entity_embeddings) 