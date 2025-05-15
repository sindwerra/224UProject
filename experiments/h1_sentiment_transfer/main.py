import torch
import mlflow
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset

# 初始化 MLflow
mlflow.set_experiment("Financial-Sentiment-Analysis-LoRA")
mlflow.start_run()

# 1. 加载数据集
dataset = load_dataset("sjyuxyz/financial-sentiment-analysis")

# 2. 计算类别权重（处理不平衡问题）
label_counts = {0: 17022, 1: 40957, 2: 22050}  # 根据你的数据分布
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
    num_labels=3,
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True,
)

# 4. 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,  # LoRA 的秩
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],  # 适用于大多数LLM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # 打印可训练参数数量

# 5. 数据预处理
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=340,
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
        "f1": f1_score(labels, predictions, average="macro"),
    }

# 8. 训练参数配置
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=3e-4,
    per_device_train_batch_size=80,
    per_device_eval_batch_size=80,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    bf16=True,
    report_to=["mlflow"],  # 将日志发送到 MLflow
)

# 9. 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# 10. 训练并记录到 MLflow
with mlflow.start_run(nested=True):
    trainer.train()
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    mlflow.log_metrics(eval_results)

# 11. 保存模型和日志
trainer.save_model("./final_model")
mlflow.end_run()
