import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, BertConfig, BertModel

# 重制csv文件
file_root = 'nlp_data/train_set.csv'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def csv_rewritting(text_dir, max_length=511):
    # 读取csv文件数据
    dataload = pd.read_csv(text_dir, sep='\t')
    text_data = dataload['text']
    label_data = dataload['label']

    f_text_data = []
    f_label_data = []
    for per_text, per_label in tqdm(zip(text_data, label_data)):

        # 统一数据类型
        per_text = per_text.split(' ')
        per_text = [int(per) for per in per_text]
        per_label = int(per_label)
        if 100 < len(per_text) < 20000:
            # 添加PAD
            while len(per_text) % max_length != 0:
                per_text.append(0)

            # 截断数据，文本最大长度为max_length
            per_label_len = len([per_text[i: i + max_length] for i in range(0, len(per_text), max_length)])
            f_text_data.extend([2] + per_text[i: i + max_length] for i in range(0, len(per_text), max_length))
            for _ in range(per_label_len):
                f_label_data.append(per_label)

    f_text_data = [' '.join(map(str, input_ids)) for input_ids in f_text_data]
    dataframe = pd.DataFrame({'labels': f_label_data, 'input_ids': f_text_data})
    dataframe.to_csv("./nlp_data/train.csv", index=False)
    return per_label_len


def data_make(data, PAD_ID=0):
    text_data = data['input_ids']
    label_data = data['labels']
    attention_mask = []
    token_type_ids = []
    f_text_data = []
    f_label_data = []

    # 制作数据集和mask
    for text in text_data:
        text = text.split(' ')
        text = [int(word) for word in text]
        mask = [0 if word == PAD_ID else 1 for word in text]
        type_id = [0 for _ in text]
        token_type_ids.append(type_id)
        attention_mask.append(mask)
        f_text_data.append(text)

    for label in label_data:
        label = int(label)
        f_label_data.append(label)

    data['input_ids'] = torch.tensor(f_text_data, device=device)
    data['token_type_ids'] = torch.tensor(token_type_ids, device=device)
    data['attention_mask'] = torch.tensor(attention_mask, device=device)
    data['labels'] = torch.tensor(f_label_data, device=device)

    return data


# csv_rewritting(file_root)
dataset = load_dataset("csv", data_dir="./nlp_data", data_files="train.csv")
dataset = dataset.shuffle(seeds=42)

label_class = len(set(dataset['train']['labels']))
trainer_dataset = dataset['train'].map(function=data_make, batched=True)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=label_class)
config = AutoConfig.from_pretrained('bert-base-uncased')


# config = BertConfig(
#     vocab_size=8000,
#     hidden_size=512,
#     num_hidden_layers=6,
#     num_attention_heads=8,
#     intermediate_size=2048,
#     hidden_dropout_prob=0.1,
#     attention_probs_dropout_prob=0.1,
# )
#
bert_model = BertModel(config)
#
#
# class Classifier(nn.Module):
#     def __init__(self, bert_model, num_classes):
#         super().__init__()
#         self.num_labels = num_classes
#         self.bert_model = bert_model.cuda()
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, num_classes).cuda()
#
#     def forward(self, input_ids, attention_mask, labels=None):
#         # print('input_ids:', input_ids)
#         outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         pooled_output = self.tanh(pooled_output)
#         logits = self.classifier(pooled_output)
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss().cuda()
#             # print('logits: ', logits)
#             # print('labels: ', labels)
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             return {'loss': loss, 'logits': logits}
#         else:
#             return {'logits', logits}
#
#
# model = Classifier(bert_model=bert_model, num_classes=label_class)

args = TrainingArguments(
    output_dir='./nlp_weight',
    per_device_train_batch_size=8,
    save_strategy='epoch',
    logging_steps=100,
    num_train_epochs=10,
    learning_rate=2e-6,
    weight_decay=0.001,
    warmup_steps=200,

)

# 冻结大模型参数
for name, param in model.bert.named_parameters():
    param.requires_grad = False

# 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=trainer_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

trainer.train()



