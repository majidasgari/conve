#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تست سریع با تعداد محدود samples برای بررسی سرعت cache
"""

import torch
import time
from model import ConvE
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.utils.global_config import Config
from conve_reranker import ConvEReranker

# تنظیمات
Config.backend = 'pytorch'
Config.cuda = True
Config.embedding_dim = 200

print("=" * 70)
print("بارگذاری Dataset و Vocabulary")
print("=" * 70)

# بارگذاری vocabulary
input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
p = Pipeline('FB15k-237', keys=input_keys)

try:
    p.load_vocabs()
except:
    print("\n⚠️  Vocabulary outdated یا وجود ندارد!")
    import sys
    sys.exit(1)

vocab = p.state['vocab']
num_entities = vocab['e1'].num_token
num_relations = vocab['rel'].num_token

print(f"✓ تعداد Entities: {num_entities}")
print(f"✓ تعداد Relations: {num_relations}")

print("\n" + "=" * 70)
print("ایجاد مدل")
print("=" * 70)

# ایجاد args object
class Args:
    embedding_dim = 200
    embedding_shape1 = 20
    hidden_drop = 0.3
    input_drop = 0.2
    feat_drop = 0.2
    hidden_size = 9728
    use_bias = False

args = Args()

# ایجاد مدل
model = ConvE(args, num_entities, num_relations)

# بارگذاری وزن‌ها
model_path = 'saved_models/FB15k-237_conve_0.2_0.3.model'
print(f"بارگذاری وزن‌ها از: {model_path}")
model_params = torch.load(model_path, weights_only=False)
model.load_state_dict(model_params)

# انتقال به GPU
model.cuda()
model.eval()

print("\n" + "=" * 70)
print("تست سریع با 10 نمونه اول")
print("=" * 70)

# ایجاد یک custom batcher که فقط 50 نمونه اول را برمی‌گرداند
class LimitedBatcher:
    def __init__(self, original_batcher, limit=50):
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

# بارگذاری test batcher
test_rank_batcher = StreamBatcher(
    'FB15k-237', 
    'test_ranking', 
    128, 
    randomize=False, 
    loader_threads=4, 
    keys=input_keys
)

limited_batcher = LimitedBatcher(test_rank_batcher, limit=10)

# ایجاد reranker
print("\nایجاد Reranker با BGE...")
reranker = ConvEReranker(
    model=model,
    vocab=vocab,
    use_gpu=True,
    k=10,  # کاهش k برای تست سریعتر
    st_model_name="BAAI/bge-m3",
    data_path="data/FB15k-237"
)

print(f"\nشروع ارزیابی 10 نمونه اول...")
start_time = time.time()

# ارزیابی با re-ranking
with torch.no_grad():
    reranker.ranking_and_hits_with_reranking(limited_batcher, 'Quick Test (10 samples)')

end_time = time.time()
elapsed = end_time - start_time

print(f"\n⏱️  زمان کل: {elapsed:.2f} ثانیه")
print(f"⏱️  زمان هر نمونه: {elapsed/10:.3f} ثانیه")
print(f"📊 تعداد embeddings در cache: {len(reranker.embedding_cache)}")

print("\n" + "=" * 70)
print("تست کامل شد!")
print("=" * 70)
