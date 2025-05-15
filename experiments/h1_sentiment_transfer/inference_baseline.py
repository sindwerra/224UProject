import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    AutoConfig
)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 1. 加载测试数据集
def load_test_dataset():
    dataset = load_dataset("zeroshot/twitter-financial-news-topic")
    
    # 合并原始train和validation
    full_dataset = concatenate_datasets([dataset['train'], dataset['validation']])
    
    # 首先按照标签分组
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(full_dataset['label']):
        label_to_indices[label].append(idx)
    
    # 对每个标签的数据进行划分
    test_indices = []
    
    for label, indices in label_to_indices.items():
        # 先划分train和temp (70% train, 30% temp)
        _, temp_idx = train_test_split(
            indices, 
            test_size=0.3, 
            random_state=42
        )
        # 再划分temp为val和test (各50%，即总15% val, 15% test)
        _, test_idx = train_test_split(
            temp_idx, 
            test_size=0.5, 
            random_state=42
        )
        
        test_indices.extend(test_idx)
    
    return full_dataset.select(test_indices)

test_dataset = load_test_dataset()

# 2. 加载基础模型和分词器
base_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# 加载配置和分词器
config = AutoConfig.from_pretrained(base_model_name, num_labels=20)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充token

# 加载基础模型
model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name,
    config=config,
    ignore_mismatched_sizes=True
)

# 3. 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=320,
    )

tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 4. 定义评估指标和混淆矩阵
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # 计算宏观指标
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    
    # 计算每个类别的指标
    per_class_precision = precision_score(labels, preds, average=None)
    per_class_recall = recall_score(labels, preds, average=None)
    per_class_f1 = f1_score(labels, preds, average=None)
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    return {
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist(),
    }

# 5. 初始化Trainer进行推理
trainer = Trainer(
    model=model,
    compute_metrics=compute_metrics,
)

# 6. 评估模型
eval_results = trainer.evaluate(tokenized_test)

# 7. 打印结果和可视化混淆矩阵
print("\nEvaluation Results:")
print(f"Macro Precision: {eval_results['eval_precision_macro']:.4f}")
print(f"Macro Recall: {eval_results['eval_recall_macro']:.4f}")
print(f"Macro F1: {eval_results['eval_f1_macro']:.4f}")

print("\nPer Class Metrics:")
for i in range(20):
    print(f"Class {i}:")
    print(f"  Precision: {eval_results['eval_per_class_precision'][i]:.4f}")
    print(f"  Recall: {eval_results['eval_per_class_recall'][i]:.4f}")
    print(f"  F1: {eval_results['eval_per_class_f1'][i]:.4f}")

# 可视化混淆矩阵
cm = np.array(eval_results["eval_confusion_matrix"])
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(20), yticklabels=range(20))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Base Model)')
plt.savefig("base_model_confusion_matrix.png")
plt.show()