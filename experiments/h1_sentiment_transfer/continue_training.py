import torch
import torch.nn as nn
from safetensors.torch import load_file
import mlflow
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from collections import defaultdict

# 初始化 MLflow
mlflow.set_experiment("Financial-News-Topic-Classification-LoRA")
mlflow.start_run()

# 1. 加载数据集并均匀拆分为train/val/test
def load_and_split_dataset():
    dataset = load_dataset("zeroshot/twitter-financial-news-topic")
    
    # 合并原始train和validation
    full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    
    # 首先按照标签分组
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(full_dataset['label']):
        label_to_indices[label].append(idx)
    
    # 对每个标签的数据进行划分
    train_indices = []
    val_indices = []
    test_indices = []
    
    for label, indices in label_to_indices.items():
        # 先划分train和temp (70% train, 30% temp)
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=0.3, 
            random_state=42
        )
        # 再划分temp为val和test (各50%，即总15% val, 15% test)
        val_idx, test_idx = train_test_split(
            temp_idx, 
            test_size=0.5, 
            random_state=42
        )
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
    
    # 创建新的DatasetDict
    return DatasetDict({
        'train': full_dataset.select(train_indices),
        'valid': full_dataset.select(val_indices),
        'test': full_dataset.select(test_indices)
    })

dataset = load_and_split_dataset()

# 2. 计算类别权重（处理不平衡问题）
label_counts = {label: 0 for label in range(20)}
for split in ['train', 'valid', 'test']:
    for label in dataset[split]['label']:
        label_counts[label] += 1

total_samples = sum(label_counts.values())
class_weights = {
    label: total_samples / (len(label_counts) * count) 
    for label, count in label_counts.items()
}
weights = torch.tensor([class_weights[i] for i in sorted(label_counts.keys())])
print("Class weights:", weights)

# 3. 加载模型和分词器
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=20,
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True,
)
lora_model_path = "./final_model"


peft_config = PeftConfig.from_pretrained(lora_model_path)

# 3. 创建新的LoRA配置，保持与原始配置相同但更新输出维度
new_lora_config = LoraConfig(
    task_type=peft_config.task_type,
    inference_mode=False,
    r=peft_config.r,
    lora_alpha=peft_config.lora_alpha,
    target_modules=peft_config.target_modules,
    lora_dropout=peft_config.lora_dropout,
    bias=peft_config.bias,
)

# model = PeftModel.from_pretrained(
#     model,
#     lora_model_path,
#     is_trainable=True  # 确保LoRA权重可训练
# )

# def adapt_classification_head(model, original_num_labels=3, new_num_labels=20):
#     # 获取原始分类头权重
#     original_head = model.classifier.out_proj
#     original_weight = original_head.weight.data
#     original_bias = original_head.bias.data
    
#     # 随机初始化新分类头
#     new_head = torch.nn.Linear(
#         original_head.in_features,
#         new_num_labels,
#         bias=original_head.bias is not None
#     )
    
#     # 部分权重迁移策略
#     if new_num_labels > original_num_labels:
#         # 保留原始分类头的权重
#         new_head.weight.data[:original_num_labels] = original_weight
#         if original_head.bias is not None:
#             new_head.bias.data[:original_num_labels] = original_bias
        
#         # 对新类别使用原始权重的均值初始化
#         mean_weight = original_weight.mean(dim=0, keepdim=True)
#         new_head.weight.data[original_num_labels:] = mean_weight.repeat(
#             new_num_labels - original_num_labels, 1
#         )
        
#         if original_head.bias is not None:
#             mean_bias = original_bias.mean(dim=0, keepdim=True)
#             new_head.bias.data[original_num_labels:] = mean_bias.repeat(
#                 new_num_labels - original_num_labels
#             )
    
#     # 替换分类头
#     model.classifier.out_proj = new_head
#     return model

# # 应用分类头适配
# model = adapt_classification_head(model, original_num_labels=3, new_num_labels=20)

# 4. 配置 LoRA
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     inference_mode=False,
#     r=8,  # 增大秩以适应更多类别
#     lora_alpha=32,
#     lora_dropout=0.1,
#     target_modules=["q_proj", "v_proj"],  # 扩展目标模块
#     bias="lora_only",
# )
model = get_peft_model(model, new_lora_config)
lora_state_dict = load_file(f"{lora_model_path}/adapter_model.safetensors")
# state_dict = torch.load(f"{lora_model_path}/adapter_model.safetensors")
# lora_state_dict = {}
filtered_state_dict = {}
for key, value in lora_state_dict.items():
    # 只加载非分类头的LoRA权重
    if 'classifier' not in key:
        filtered_state_dict[key] = value

# 7. 加载筛选后的权重
model.load_state_dict(filtered_state_dict, strict=False)

# 7. 重新初始化分类头
model.classifier = nn.Linear(model.config.hidden_size, 20)

# 8. 确保模型处于训练模式
model.train()
model.print_trainable_parameters()  # 打印可训练参数数量

# 5. 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=320,  # 缩短max_length以适应金融新闻推文
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 6. 自定义损失函数（处理类别不平衡）
class WeightedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, weight=None):
        super().__init__(weight=weight)

    def forward(self, input, target):
        return super().forward(input, target)

# 7. 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
    }

# 8. 训练参数配置
training_args = TrainingArguments(
    output_dir="./new_cls_with_sent_results",
    eval_strategy="steps",
    eval_steps=500,  # 更频繁的评估
    save_strategy="steps",
    save_steps=500,
    logging_strategy="steps",
    logging_steps=500,
    learning_rate=1e-4,  # 降低学习率
    per_device_train_batch_size=80,  # 减小batch size
    per_device_eval_batch_size=80,
    num_train_epochs=15,  # 增加epoch
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    bf16=True,
    report_to=["mlflow"],
    # gradient_accumulation_steps=2,  # 增加梯度累积
    # warmup_ratio=0.1,  # 增加warmup
)

# 9. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3)  # 增加耐心值
    ],
)

# 10. 训练并记录到 MLflow
with mlflow.start_run(nested=True):
    # 记录类别分布
    mlflow.log_dict(label_counts, "label_distribution.json")
    
    # 记录训练参数
    mlflow.log_params({
        "model": model_name,
        "lora_r": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "target_modules": str(peft_config.target_modules),
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "num_epochs": training_args.num_train_epochs,
    })
    
    trainer.train()
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    mlflow.log_metrics(eval_results)

# 11. 保存模型和日志
trainer.save_model("./new_cls_with_sent_final_model")
mlflow.end_run()

# 打印类别映射
topics = {
    "LABEL_0": "Analyst Update",
    "LABEL_1": "Fed | Central Banks",
    "LABEL_2": "Company | Product News",
    "LABEL_3": "Treasuries | Corporate Debt",
    "LABEL_4": "Dividend",
    "LABEL_5": "Earnings",
    "LABEL_6": "Energy | Oil",
    "LABEL_7": "Financials",
    "LABEL_8": "Currencies",
    "LABEL_9": "General News | Opinion",
    "LABEL_10": "Gold | Metals | Materials",
    "LABEL_11": "IPO",
    "LABEL_12": "Legal | Regulation",
    "LABEL_13": "M&A | Investments",
    "LABEL_14": "Macro",
    "LABEL_15": "Markets",
    "LABEL_16": "Politics",
    "LABEL_17": "Personnel Change",
    "LABEL_18": "Stock Commentary",
    "LABEL_19": "Stock Movement",
}
print("\nTopic Label Mapping:")
for k, v in topics.items():
    print(f"{k}: {v}")