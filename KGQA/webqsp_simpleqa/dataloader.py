import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import os
import unicodedata
import re
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from tqdm.utils import RE_ANSI
from transformers import *


class DatasetMetaQA(Dataset):
    def __init__(self, data, entities, entity2idx, relation2idx):
        self.data = data
        self.entities = entities
        self.entity2idx = entity2idx
        self.pos_dict = defaultdict(list)
        self.neg_dict = defaultdict(list)
        self.index_array = list(self.entities.keys())
        self.tokenizer_class = RobertaTokenizer
        self.pretrained_weights = 'roberta-base'
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self.relation2idx = relation2idx

    def __len__(self):
        return len(self.data)
    
    def pad_sequence(self, arr, max_len=128):
        num_to_add = max_len - len(arr)
        for _ in range(num_to_add):
            arr.append('<pad>')
        return arr

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        batch_size = len(indices)
        vec_len = len(self.entity2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        one_hot.scatter_(0, indices, 1)
        return one_hot
    
    def relToOneHot(self, indices):
        # sample 4 indices here
        num_rel = len(indices)
        indices = torch.LongTensor(indices)
        vec_len = len(self.relation2idx)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        # one_hot = -torch.ones(vec_len, dtype=torch.float32)
        value = 1 / num_rel
        one_hot.scatter_(0, indices, value)
        return one_hot
    
    def sampleRelation(self, indices):
        # sample 4 indices here
        sample_size = 4
        if len(indices) >= sample_size:
            indices = indices[0:sample_size]
            sampled_neighbors = indices
        else:
            lef_len = sample_size - len(indices)
            tmp = np.random.choice(list(indices), size=lef_len, replace=len(indices) < sample_size)
            indices.extend(tmp)
            sampled_neighbors = indices
        
        return torch.LongTensor(sampled_neighbors)

    def __getitem__(self, index):
        data_point = self.data[index]
        question_text = data_point[1]
        question_tokenized, attention_mask = self.tokenize_question(question_text)
        head_id = self.entity2idx[data_point[0].strip()]
        tail_ids = []
        for tail_name in data_point[2]:
            tail_name = tail_name.strip()
            #TODO: dunno if this is right way of doing things
            if tail_name in self.entity2idx:
                tail_ids.append(self.entity2idx[tail_name])
        tail_onehot = self.toOneHot(tail_ids)

        rel_ids = []
        for rel_name in data_point[3]:
            rel_name = rel_name.strip()
            if rel_name in self.relation2idx:
                rel_ids.append(self.relation2idx[rel_name])
            else:
                error = 0
        rel_score = self.sampleRelation(rel_ids)

        return question_tokenized, attention_mask, head_id, tail_onehot, rel_score

    def tokenize_question(self, question):
        question = "<s> " + question + " </s>"
        question_tokenized = self.tokenizer.tokenize(question)
        question_tokenized = self.pad_sequence(question_tokenized, 64)
        question_tokenized = torch.tensor(self.tokenizer.encode(question_tokenized, add_special_tokens=False))
        attention_mask = []
        for q in question_tokenized:
            # 1 means padding token
            if q == 1:
                attention_mask.append(0)
            else:
                attention_mask.append(1)
        return question_tokenized, torch.tensor(attention_mask, dtype=torch.long)

# def _collate_fn(batch):
#     print(len(batch))
#     exit(0)
#     question_tokenized = batch[0]
#     attention_mask = batch[1]
#     head_id = batch[2]
#     tail_onehot = batch[3]
#     question_tokenized = torch.stack(question_tokenized, dim=0)
#     attention_mask = torch.stack(attention_mask, dim=0)
#     return question_tokenized, attention_mask, head_id, tail_onehot 

class DataLoaderMetaQA(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoaderMetaQA, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn

