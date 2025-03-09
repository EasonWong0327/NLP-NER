# Multilingual Named Entity Recognition System

A deep learning-based Named Entity Recognition (NER) system supporting both Chinese and English, with multiple advanced model architectures for identifying entities such as products, regions, time expressions, metrics, companies, and currencies.

## Features

- Bilingual support (Chinese & English)
- Multiple entity type recognition
- Advanced model architectures
- Complete training and evaluation pipeline
- Detailed documentation and examples

## Supported Entity Types

- `product`: Product names (e.g., "iPhone 14", "Huawei Mate60")
- `region`: Location names (e.g., "Beijing", "Shanghai")
- `time`: Time expressions (e.g., "2024", "next quarter")
- `metric`: Business metrics (e.g., "revenue", "growth rate")
- `company`: Company names (e.g., "Alibaba", "Apple")
- `currency`: Currency types (e.g., "CNY", "USD")

## Model Architectures

### 1. BERT+CRF (bert_ner)
A sequence labeling model based on BERT with CRF layer for modeling label dependencies.

**Related Papers**:
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)

### 2. Span-based NER (span_ner)
A span-based entity recognition model that directly classifies text segments.

**Related Papers**:
- [Named Entity Recognition as Dependency Parsing](https://aclanthology.org/2020.acl-main.577/)
- [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476)

### 3. Prompt-based NER (prompt_ner)
A prompt-based entity recognition model that transforms NER into a generation task.

**Related Papers**:
- [Template-Based Named Entity Recognition Using BART](https://aclanthology.org/2021.acl-long.489/)
- [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/abs/2012.15723)

### 4. Transformer NER (transformer_ner)
A transformer-based entity recognition model using multi-layer self-attention for feature extraction.

**Related Papers**:
- [A Unified Generative Framework for Various NER Subtasks](https://arxiv.org/abs/2106.01223)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## Quick Start

### Requirements
```bash
Python >= 3.7
PyTorch >= 1.8.0
transformers >= 4.0.0
```

### Installation
```bash
pip install -r requirements.txt
```

### Training Data Note
The training data included in this project is for demonstration and model validation purposes only, with limited data size and scenario coverage. If you plan to apply this project in a production environment, please consider the following:

1. Data Scale: Recommended to use at least 10,000 annotated samples for training
2. Data Quality: Ensure accuracy and consistency of annotations
3. Scenario Coverage: Training data should cover various cases in your target application
4. Data Distribution: Maintain reasonable distribution across entity types

You can obtain more training data through:
- Using public NER datasets
- Manual annotation
- Semi-supervised learning methods
- Data augmentation techniques

### Data Preparation
```bash
python utils/data_processor.py
```

### Model Training
```bash
# Choose model type for training
python train.py --model transformer_ner
python train.py --model bert_ner
python train.py --model span_ner
python train.py --model prompt_ner
```

## Evaluation Metrics

- F1 Score
- Precision
- Recall

## Project Structure
```
NLP-Ner/
├── config.py                 # Configuration
├── train.py                 # Training script
├── models/                  # Model implementations
│   ├── bert_ner.py         # BERT+CRF
│   ├── span_ner.py         # Span-based
│   ├── prompt_ner.py       # Prompt-based
│   └── transformer_ner.py   # Transformer
├── utils/                   # Utilities
│   ├── data_processor.py   # Data processing
│   └── metrics.py          # Evaluation metrics
└── data/                   # Data directory
    ├── raw/                # Raw data
    └── processed/          # Processed data
```

## Contributing
Issues and Pull Requests are welcome.

## License
MIT License

## Citation
If you use this code, please cite:
```bibtex
@misc{nlp-ner,
  author = {EasonWong0327},
  title = {Multilingual Named Entity Recognition System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/EasonWong0327/NLP-NER}
}
``` 