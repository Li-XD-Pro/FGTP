import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# 加载数据
with open('Ontology_Tuple.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 预处理数据
objects = [item['object'] for item in data]
properties = [item['property'] for item in data]

# 对属性进行编码
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(properties)

# 划分训练集和测试集
objects_train, objects_val, labels_train, labels_val = train_test_split(objects, labels, test_size=0.1, random_state=42)

# 加载分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 分词和编码数据
train_encodings = tokenizer(objects_train, truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(objects_val, truncation=True, padding=True, return_tensors='pt')

# 创建TensorDataset
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(labels_train))
val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(labels_val))

# 创建DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

# 加载预训练的BERT模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# 将模型移至GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 设置优化器和调度器
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def evaluate(model, val_loader, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs.loss
            val_loss += loss.item()
            predictions = outputs.logits.argmax(dim=-1)
            correct += (predictions == inputs['labels']).sum().item()
            total += inputs['labels'].size(0)
    return val_loss / len(val_loader), correct / total

def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            model.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.2f}, Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}")

train(model, train_loader, val_loader, optimizer, scheduler, device, epochs)

# 指定保存模型的路径
model_path = 'fine_tuned_bert_base_uncased'
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

print(f"模型已经被保存到 {model_path}")


