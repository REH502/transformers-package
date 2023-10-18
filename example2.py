import nltk
import collections
import evaluate
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, \
    DefaultDataCollator, pipeline

# 加载数据集
datasets = load_dataset('cmrc2018', cache_dir='example2_data')

# 数据预处理
tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')
sample_datasets = datasets['train'].select(range(10))


def pre_process(examples):
    tokenized_examples = tokenizer(text=sample_datasets['question'],
                                   text_pair=sample_datasets['context'],
                                   return_offsets_mapping=True,
                                   return_overflowing_tokens=True,
                                   stride=128,
                                   max_length=384, truncation='only_second', padding='max_length')
    sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')

    start_positions = []
    end_positions = []
    example_ids = []

    for idx, _ in enumerate(sample_mapping):
        answer = sample_datasets['answers'][sample_mapping[idx]]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])

        context_start = tokenized_examples.sequence_ids(idx).index(1)
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1

        offset = tokenized_examples.get('offset_mapping')[idx]

        if offset[context_end][1] < start_char or offset[context_start][0] > end_char:
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offset[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id
            while token_id >= context_start and offset[token_id][1] > end_char:
                token_id -= 1
            end_token_pos = token_id
        start_positions.append(start_token_pos)
        start_positions.append(end_token_pos)
        example_ids.append(examples['id'][sample_mapping[idx]])

        tokenized_examples['offset_mapping'][idx] = [(o if tokenized_examples.sequence_ids(idx)[k] == 1 else None)
                                                     for k, o in enumerate(tokenized_examples['offset_mapping'][idx])]

        tokenized_examples['start_positions'] = start_positions
        tokenized_examples['end_positions'] = end_positions
        tokenized_examples['example_ids'] = example_ids

    return tokenized_examples


tokenized_datasets = datasets.map(pre_process, batched=True, remove_columns=datasets['train'].cloumn_names)

# 获取模型输出
def get_result(start_logits, end_logits, examples, features):
    predictions = {}
    references = {}

    example_to_feature = collections.defaultdict(list)
    for idx, example_id in enumerate(features['example_id']):
        example_to_feature[example_id].append(idx)
