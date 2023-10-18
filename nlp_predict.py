import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, pipeline, BertModel, BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_csv_rewritting(text_dir, target_dir, max_length=511):
    # 读取csv文件数据
    dataload = pd.read_csv(text_dir, sep='\t')
    text_data = dataload['text']

    f_text_data = []
    label_len_list = []
    for per_text in tqdm(text_data):

        # 统一数据类型
        per_text = per_text.split(' ')
        per_text = [int(per) for per in per_text]

        # 添加PAD
        while len(per_text) % max_length != 0:
            per_text.append(0)

        # 截断数据，文本最大长度为max_length
        per_label_len = len([per_text[i: i + max_length] for i in range(0, len(per_text), max_length)])
        label_len_list.append(per_label_len)
        f_text_data.extend([2] + per_text[i: i + max_length] for i in range(0, len(per_text), max_length))

    f_text_data = [' '.join(map(str, input_ids)) for input_ids in f_text_data]

    dataframe = pd.DataFrame({'input_ids': f_text_data})
    dataframe.to_csv(target_dir, index=False)
    return label_len_list


def test_data_make(data, PAD_ID=0):
    text_data = data['input_ids']
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

    data['input_ids'] = torch.tensor(f_text_data, device=device)
    # data['token_type_ids'] = torch.tensor(token_type_ids, device=device)
    data['attention_mask'] = torch.tensor(attention_mask, device=device)

    return data


config = BertConfig(
    vocab_size=8000,  # 词汇表大小，对于英文BERT模型，这个值是30522
    hidden_size=512,  # Transformer中隐藏层的大小
    num_hidden_layers=6,  # Transformer中隐藏层的数量
    num_attention_heads=8,  # 注意力头的数量
    intermediate_size=2048,  # feedforward网络的大小
    hidden_dropout_prob=0.1,  # dropout概率
    attention_probs_dropout_prob=0.1,  # 注意力权重的dropout概率
)

# 然后，我们可以使用这个配置对象来实例化一个模型
bert_model = BertModel(config)


# 最后，我们创建一个分类器，它由BERT模型和一个线性层组成
class Classifier(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.num_labels = num_classes
        self.bert_model = bert_model.cuda()
        self.classifier = nn.Linear(config.hidden_size, num_classes).cuda()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss().cuda()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        else:
            return logits


model = Classifier(bert_model=bert_model, num_classes=14)
model.load_state_dict(torch.load('nlp_weight/checkpoint-168006/pytorch_model.bin'))


test_dir = 'nlp_data/test_a.csv'
len_list = test_csv_rewritting(test_dir, target_dir='nlp_data/test.csv')
dataset = load_dataset("csv", data_dir="./nlp_data", data_files="test.csv")
test_dataset = dataset['train'].map(function=test_data_make, batched=True)
dataset_size = len(test_dataset)
#
# # 抽取30%的数据作为训练集
train_size = int(0.998* dataset_size)
test_size = dataset_size - train_size
#
# # 随机划分数据集
train_dataset, test_dataset = random_split(test_dataset, [train_size, test_size])
dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
# model = AutoModelForSequenceClassification.from_pretrained('nlp_weight/checkpoint-50526').cuda()
# 遍历 DataLoader 中的每一批数据

text_output = []
for batch in tqdm(dataloader):
    # 将每一批数据输入到模型中
    batch = {k: torch.stack(v).cuda() for k, v in batch.items()}
    outputs = model(batch['input_ids'], batch['attention_mask'], labels=None)
    outputs = F.softmax(outputs, dim=-1)
    outputs = torch.argmax(outputs, dim=-1)
    text_output.append(int(outputs[0]))

i = 0
label_list = []
for per_len in len_list:
    output = text_output[i:i + per_len - 1]
    counter = Counter(output)
    label = counter.most_common()[0][0]
    label_list.append(label)
    i = i + per_len
    dataframe = pd.DataFrame({'labels': label_list})
    dataframe.to_csv("./submission.csv", index=False)





