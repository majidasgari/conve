#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تست خیلی سریع با 3 نمونه
"""

import torch
import numpy as np
from model import ConvE
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.utils.global_config import Config
from conve_reranker import ConvEReranker

# تنظیمات
Config.backend = 'pytorch'
Config.cuda = True
Config.embedding_dim = 200

print("بارگذاری...")
input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
p = Pipeline('FarsPredict', keys=input_keys)
p.load_vocabs()

vocab = p.state['vocab']
num_entities = vocab['e1'].num_token

class Args:
    embedding_dim = 200
    embedding_shape1 = 20
    hidden_drop = 0.3
    input_drop = 0.2
    feat_drop = 0.2
    hidden_size = 9728
    use_bias = False

model = ConvE(Args(), num_entities, vocab['rel'].num_token)
model.load_state_dict(torch.load('saved_models/FarsPredict_conve_0.2_0.3.epoch_281.model', weights_only=False))
model.cuda()
model.eval()

class LimitedBatcher:
    def __init__(self, original_batcher, limit=3):
        self.original_batcher = original_batcher
        self.limit = limit
        self.count = 0
    def __iter__(self):
        self.count = 0
        self.iter = iter(self.original_batcher)
        return self
    def __next__(self):
        if self.count >= self.limit:
            raise StopIteration
        self.count += 1
        return next(self.iter)

test_batcher = StreamBatcher('FarsPredict', 'test_ranking', 1, randomize=False, loader_threads=1, keys=input_keys)
limited = LimitedBatcher(test_batcher, limit=100)

reranker = ConvEReranker(model=model, vocab=vocab, use_gpu=True, k=20, st_model_name="BAAI/bge-m3", data_path="data/FarsPredict")

print("\nتست با 100 نمونه...\n")
with torch.no_grad():
    reranker.ranking_and_hits_with_reranking(limited, 'Test')
