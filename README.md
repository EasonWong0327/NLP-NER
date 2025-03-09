# 中英双语命名实体识别系统

基于深度学习的中英双语命名实体识别（NER）系统，支持多种先进的模型架构，可识别产品、地区、时间、指标、公司和货币等多种实体类型。

## 功能特点

- 支持中英双语
- 多种实体类型识别
- 多种先进模型架构
- 完整的训练评估流程
- 详细的文档和示例

## 支持的实体类型

- `product`：产品名称（如"iPhone 14"、"华为Mate60"）
- `region`：地区名称（如"北京"、"Shanghai"）
- `time`：时间表达（如"2024年"、"下个季度"）
- `metric`：指标名称（如"营收"、"增长率"）
- `company`：公司名称（如"阿里巴巴"、"Apple"）
- `currency`：货币类型（如"人民币"、"USD"）

## 模型架构

### 1. BERT+CRF (bert_ner)
基于BERT的序列标注模型，使用CRF层建模标签依赖关系。

**相关论文**：
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360)

### 2. Span-based NER (span_ner)
基于span的实体识别模型，直接对文本片段进行分类。

**相关论文**：
- [Named Entity Recognition as Dependency Parsing](https://aclanthology.org/2020.acl-main.577/)
- [A Unified MRC Framework for Named Entity Recognition](https://arxiv.org/abs/1910.11476)

### 3. Prompt-based NER (prompt_ner)
基于提示的实体识别模型，将NER转化为生成任务。

**相关论文**：
- [Template-Based Named Entity Recognition Using BART](https://aclanthology.org/2021.acl-long.489/)
- [Making Pre-trained Language Models Better Few-shot Learners](https://arxiv.org/abs/2012.15723)

### 4. Transformer NER (transformer_ner)
基于Transformer的实体识别模型，使用多层自注意力进行特征提取。

**相关论文**：
- [A Unified Generative Framework for Various NER Subtasks](https://arxiv.org/abs/2106.01223)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 快速开始

### 环境要求
```bash
Python >= 3.7
PyTorch >= 1.8.0
transformers >= 4.0.0
```

### 安装依赖
```bash
pip install -r requirements.txt # 这是我的conda环境直接导出来的，关注核心包即可
```

### 训练数据说明
当前项目中包含的训练数据仅用于功能演示和模型验证，数据量较小且覆盖场景有限。如果要将本项目应用于实际工程中，请注意：

1. 数据规模：建议使用至少1万条以上的标注数据进行训练
2. 数据质量：确保标注数据的准确性和一致性
3. 场景覆盖：训练数据应覆盖目标应用场景的各种情况
4. 数据分布：保持训练数据在各个实体类型上的合理分布

可以通过以下方式获取更多训练数据：
- 使用公开的NER数据集
- 进行人工标注
- 使用半监督学习方法扩充数据
- 利用数据增强技术

### 数据准备
```bash
# 准备原始数据
python utils/data_processor.py
```

### 模型训练
```bash
# 选择模型类型进行训练
python train.py --model transformer_ner  # Transformer模型
python train.py --model bert_ner     # BERT+CRF模型
python train.py --model span_ner     # Span-based模型
python train.py --model prompt_ner   # Prompt-based模型
```

## 评估指标

- F1 Score
- Precision
- Recall


## 目录结构
```
NLP-Ner/
├── config.py                 # 配置文件
├── train.py                 # 训练脚本
├── models/                  # 模型实现
│   ├── bert_ner.py         # BERT+CRF模型
│   ├── span_ner.py         # Span-based模型
│   ├── prompt_ner.py       # Prompt-based模型
│   └── transformer_ner.py   # Transformer模型
├── utils/                   # 工具函数
│   ├── data_processor.py   # 数据处理
│   └── metrics.py          # 评估指标
└── data/                   # 数据目录
    ├── raw/                # 原始数据
    └── processed/          # 处理后的数据
```

## 贡献指南
欢迎提交Issue和Pull Request。

## 许可证
MIT License

## 引用
如果您使用了本项目的代码，请引用：
```bibtex
@misc{nlp-ner,
  author = {EasonWong0327},
  title = {Multilingual Named Entity Recognition System},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/EasonWong0327/NLP-NER}
}
``` 