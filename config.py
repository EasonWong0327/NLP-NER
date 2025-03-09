from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class Config:
    # 数据
    data_dir: str = "data"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    train_file: str = "train.txt"
    dev_file: str = "dev.txt"
    test_file: str = "test.txt"
    
    # model
    model_name: str = "bert-base-chinese"  # 中文BERT
    model_type: str = "transformer_ner"  # oth: bert_ner, span_ner, prompt_ner, transformer_ner
    max_length: int = 512
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    
    # 实体类型，和下边映射关联，要改都得改
    entity_types: List[str] = field(default_factory=lambda: [
        "product", "region", "time", "metric", "company", "currency"
    ])
    
    # 标签映射
    label2id: Dict[str, int] = field(default_factory=lambda: {
        "O": 0,
        "B-product": 1, "I-product": 2,
        "B-region": 3, "I-region": 4,
        "B-time": 5, "I-time": 6,
        "B-metric": 7, "I-metric": 8,
        "B-company": 9, "I-company": 10,
        "B-currency": 11, "I-currency": 12
    })
    
    @property
    def id2label(self) -> Dict[int, str]:
        return {v: k for k, v in self.label2id.items()}
    
    # train
    device: str = "cuda"  # 没GPU不建议运行
    seed: int = 42
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # 保存
    output_dir: str = "outputs"
    model_save_dir: str = "outputs/models"
    log_dir: str = "outputs/logs"
    
    # eval
    eval_batch_size: int = 64
    test_batch_size: int = 64 