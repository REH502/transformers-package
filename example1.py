import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification, pipeline

# 加载数据集
ner_datasets = load_dataset("peoples_daily_ner", cache_dir='./example1_data')
label_list = ner_datasets['train'].features['ner_tags'].feature.names

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base', )


# 数据预处理函数
def pre_process(examples):
    tokenize_examples = tokenizer(examples['tokens'], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenize_examples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenize_examples['labels'] = labels
    return tokenize_examples


tokenized_datasets = ner_datasets.map(pre_process, batched=True)
print(type(tokenized_datasets['train']['input_ids']))

# 建立模型
# model = AutoModelForTokenClassification.from_pretrained('hfl/chinese-macbert-base', num_labels=len(label_list))
model = AutoModelForTokenClassification.from_pretrained('example1_weight/checkpoint-981')
model.config.id2label = {idx: label for idx, label in enumerate(label_list)}

# 评价函数
seqeval = evaluate.load('seqeval')


def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)

    true_predictions = [[label_list[p] for p, l in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    true_labels = [[label_list[l] for p, l in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]

    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode='strict', scheme='IOB2')

    return {'f1': result['overall_f1']}


# 配置训练参数
args = TrainingArguments(
    output_dir='./example1_weight',
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    metric_for_best_model='f1',
    load_best_model_at_end=True,
    logging_steps=50,
    num_train_epochs=3
)

# 创建训练器
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=eval_metric,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)

# 模型训练
trainer.train()


ner = pipeline('token-classification', model=model, tokenizer=tokenizer, device=0, aggregation_strategy='simple')

sen_input = input('请输入：')

result = ner(sen_input)
result_dic = {}
for r in result:
    if r['entity_group'] not in result_dic:
        result_dic[r['entity_group']] = []
    result_dic[r['entity_group']].append(sen_input[r['start']: r['end']])

print(result_dic)

