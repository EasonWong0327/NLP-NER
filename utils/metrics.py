from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

def compute_metrics(predictions: List[Tuple[int, int, str]], labels: List[List[str]], config) -> Dict[str, float]:
    """计算评估指标"""
    # 将预测结果转换为标签列表
    pred_labels = []
    for pred in predictions:
        start, end, entity_type = pred
        pred_labels.extend(['O'] * start)
        pred_labels.append(f'B-{entity_type}')
        pred_labels.extend([f'I-{entity_type}'] * (end - start - 1))
        pred_labels.extend(['O'] * (config.max_length - end))
    
    # 展平
    flat_labels = [label for seq in labels for label in seq]
    # 移除padding
    pred_labels = [p for p, l in zip(pred_labels, flat_labels) if l != 'O']
    flat_labels = [l for l in flat_labels if l != 'O']
    # 标签转换ID
    pred_ids = [config.label2id[p] for p in pred_labels]
    label_ids = [config.label2id[l] for l in flat_labels]

    f1 = f1_score(label_ids, pred_ids, average='weighted')
    precision = precision_score(label_ids, pred_ids, average='weighted')
    recall = recall_score(label_ids, pred_ids, average='weighted')
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def get_entity_spans(labels: List[str], config) -> List[Tuple[int, int, str]]:
    """获取实体span"""
    spans = []
    start = None
    current_type = None
    
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if start is not None:
                spans.append((start, i, current_type))
            start = i
            current_type = label[2:]
        elif label.startswith('I-'):
            if start is None or current_type != label[2:]:
                if start is not None:
                    spans.append((start, i, current_type))
                start = None
                current_type = None
        else:
            if start is not None:
                spans.append((start, i, current_type))
                start = None
                current_type = None
    
    if start is not None:
        spans.append((start, len(labels), current_type))
    
    return spans

def compute_span_metrics(predictions: List[Tuple[int, int, str]], labels: List[List[str]], config) -> Dict[str, float]:
    """计算基于span的评估指标"""
    pred_labels = []
    for pred in predictions:
        start, end, entity_type = pred
        pred_labels.extend(['O'] * start)
        pred_labels.append(f'B-{entity_type}')
        pred_labels.extend([f'I-{entity_type}'] * (end - start - 1))
        pred_labels.extend(['O'] * (config.max_length - end))

    flat_labels = [label for seq in labels for label in seq]

    pred_spans = get_entity_spans(pred_labels, config)
    label_spans = get_entity_spans(flat_labels, config)
    
    # 计算正确预测span
    correct = 0
    for pred_span in pred_spans:
        if pred_span in label_spans:
            correct += 1

    precision = correct / len(pred_spans) if pred_spans else 0
    recall = correct / len(label_spans) if label_spans else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'span_f1': f1,
        'span_precision': precision,
        'span_recall': recall
    }

def compute_type_metrics(predictions: List[Tuple[int, int, str]], labels: List[List[str]], config) -> Dict[str, float]:
    """计算每个实体类型的评估指标"""
    type_metrics = {}

    pred_labels = []
    for pred in predictions:
        start, end, entity_type = pred
        pred_labels.extend(['O'] * start)
        pred_labels.append(f'B-{entity_type}')
        pred_labels.extend([f'I-{entity_type}'] * (end - start - 1))
        pred_labels.extend(['O'] * (config.max_length - end))

    flat_labels = [label for seq in labels for label in seq]
    
    for entity_type in config.entity_types:
        type_preds = []
        type_labels = []
        
        for p, l in zip(pred_labels, flat_labels):
            if l != 'O':
                type_preds.append(1 if p.endswith(entity_type) else 0)
                type_labels.append(1 if l.endswith(entity_type) else 0)
        
        if type_labels:
            f1 = f1_score(type_labels, type_preds)
            precision = precision_score(type_labels, type_preds)
            recall = recall_score(type_labels, type_preds)
            
            type_metrics[entity_type] = {
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
    
    return type_metrics 