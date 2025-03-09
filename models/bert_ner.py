import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF
from config import Config

class BertNER(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.bert = BertModel.from_pretrained(config.model_name)
        self.hidden = nn.Linear(self.bert.config.hidden_size, 256)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(256, len(config.label2id))
        self.crf = CRF(
            num_tags=len(config.label2id),
            batch_first=True
        )
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = outputs[0]

        sequence_output = self.hidden(sequence_output)
        sequence_output = self.dropout(sequence_output)

        emissions = self.classifier(sequence_output)

        if labels is not None: # train
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else: # pred
            predictions = self.crf.decode(emissions, mask=attention_mask.byte())
            return predictions
    
    def predict(self, input_ids, attention_mask):
        self.eval()
        with torch.no_grad():
            predictions = self(input_ids, attention_mask)
        return predictions 