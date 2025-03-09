"""
Author: eason wong
Date: 2025.1.5
Description: NER-train
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from config import Config
from utils.data_processor import create_dataloaders
from utils.metrics import compute_metrics, compute_span_metrics, compute_type_metrics
from models.bert_ner import BertNER
from models.span_ner import SpanNER
from models.prompt_ner import PromptNER
from models.transformer_ner import TransformerNER

def get_model(config: Config):
    """根据配置获取对应的模型"""
    if config.model_type == 'bert_ner':
        return BertNER(config)
    elif config.model_type == 'span_ner':
        return SpanNER(config)
    elif config.model_type == 'prompt_ner':
        return PromptNER(config)
    elif config.model_type == 'transformer_ner':
        return TransformerNER(config)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")

def train(config: Config):
    torch.manual_seed(config.seed)
    
    train_loader, dev_loader, test_loader = create_dataloaders(config)
    
    model = get_model(config)
    model.to(config.device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs
    )
    
    best_f1 = 0
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        # 训练阶段
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            start_labels = batch['start_labels'].to(config.device)
            end_labels = batch['end_labels'].to(config.device)
            type_labels = batch['type_labels'].to(config.device)
            
            loss = model(input_ids, attention_mask, start_labels, end_labels, type_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # eval
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                type_labels = batch['type_labels']
                
                predictions = model.predict(input_ids, attention_mask)
                all_predictions.extend(predictions)
                #将type_labels转换为原始标签格式
                for i in range(len(type_labels)):
                    labels = []
                    for j in range(len(type_labels[i])):
                        label_id = type_labels[i][j].item()
                        label = config.id2label[label_id]
                        labels.append(label)
                    all_labels.append(labels)
        

        metrics = compute_metrics(all_predictions, all_labels, config)
        span_metrics = compute_span_metrics(all_predictions, all_labels, config)
        type_metrics = compute_type_metrics(all_predictions, all_labels, config)
        

        print(f"\nEpoch {epoch + 1} 验证集评估结果:")
        print(f"Loss: {total_loss / len(train_loader):.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Span F1: {span_metrics['span_f1']:.4f}")
        print("\n各实体类型的F1分数:")
        for entity_type, scores in type_metrics.items():
            print(f"{entity_type}: {scores['f1']:.4f}")
        

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            os.makedirs(config.model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), 
                      os.path.join(config.model_save_dir, f'{config.model_type}_best.pt'))
            print(f"\n保存最佳模型，F1 Score: {best_f1:.4f}")
    

    print("\n开始测试阶段...")
    model.load_state_dict(torch.load(os.path.join(config.model_save_dir, f'{config.model_type}_best.pt')))
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            type_labels = batch['type_labels']
            
            predictions = model.predict(input_ids, attention_mask)
            all_predictions.extend(predictions)
            for i in range(len(type_labels)):
                labels = []
                for j in range(len(type_labels[i])):
                    label_id = type_labels[i][j].item()
                    label = config.id2label[label_id]
                    labels.append(label)
                all_labels.append(labels)
    
    test_metrics = compute_metrics(all_predictions, all_labels, config)
    test_span_metrics = compute_span_metrics(all_predictions, all_labels, config)
    test_type_metrics = compute_type_metrics(all_predictions, all_labels, config)
    
    print("\n测试集评估结果:")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"Span F1: {test_span_metrics['span_f1']:.4f}")
    print("\n各实体类型的F1分数:")
    for entity_type, scores in test_type_metrics.items():
        print(f"{entity_type}: {scores['f1']:.4f}")

if __name__ == '__main__':
    config = Config()
    train(config) 