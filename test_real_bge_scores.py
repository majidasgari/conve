#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تست واقعی: چک کردن BGE scores واقعی
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

print("=" * 70)
print("بارگذاری FarsPredict و مدل")
print("=" * 70)

input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
p = Pipeline('FarsPredict', keys=input_keys)
p.load_vocabs()

vocab = p.state['vocab']
num_entities = vocab['e1'].num_token
num_relations = vocab['rel'].num_token

class Args:
    embedding_dim = 200
    embedding_shape1 = 20
    hidden_drop = 0.3
    input_drop = 0.2
    feat_drop = 0.2
    hidden_size = 9728
    use_bias = False

args = Args()
model = ConvE(args, num_entities, num_relations)
model_path = 'saved_models/FarsPredict_conve_0.2_0.3.epoch_281.model'
model_params = torch.load(model_path, weights_only=False)
model.load_state_dict(model_params)
model.cuda()
model.eval()

# ایجاد reranker
print("\nایجاد Reranker...")
reranker = ConvEReranker(
    model=model,
    vocab=vocab,
    use_gpu=True,
    k=10,
    st_model_name="BAAI/bge-m3",
    data_path="data/FarsPredict"
)

# تست یک نمونه
test_rank_batcher = StreamBatcher(
    'FarsPredict', 
    'test_ranking', 
    128, 
    randomize=False, 
    loader_threads=4, 
    keys=input_keys
)

print("\n" + "=" * 70)
print("تحلیل واقعی BGE scores برای یک نمونه")
print("=" * 70)

all_entity_ids = np.arange(num_entities)

for i, str2var in enumerate(test_rank_batcher):
    e1 = str2var['e1']
    rel = str2var['rel']
    e2 = str2var['e2']
    
    with torch.no_grad():
        pred1 = model.forward(e1, rel)
        pred1 = pred1.data
        e1, e2 = e1.data, e2.data
    
    # فقط اولین نمونه
    batch_idx = 0
    h_true = e1[batch_idx, 0].item()
    t_true = e2[batch_idx, 0].item()
    r_true = rel[batch_idx, 0].item()
    
    kge_scores_t = pred1[batch_idx].cpu().numpy()
    
    # پیدا کردن top-10
    top_k_indices_t = np.argsort(-kge_scores_t)[:10]
    top_k_tail_ids = all_entity_ids[top_k_indices_t]
    
    # گرفتن نام‌ها
    h_name = reranker._get_names(h_true, reranker.id2entity)[0]
    r_name = reranker._get_names(r_true, reranker.id2relation)[0]
    t_true_name = reranker._get_names(t_true, reranker.id2entity)[0]
    
    print(f"\nنمونه: ({h_name}, {r_name}, {t_true_name})")
    print(f"  h_true={h_true}, r_true={r_true}, t_true={t_true}")
    
    # اضافه کردن t_true به لیست
    ids_to_score = list(set(top_k_tail_ids) | {t_true})
    
    print(f"\n  محاسبه BGE scores برای {len(ids_to_score)} entities...")
    bge_scores = reranker._calculate_bge_scores_batch(
        h_name, r_name, ids_to_score, reranker.id2entity
    )
    
    print(f"\n  BGE Scores (lower is better):")
    sorted_by_bge = sorted(bge_scores.items(), key=lambda x: x[1])
    for rank, (eid, score) in enumerate(sorted_by_bge[:15], 1):
        ename = reranker._get_names(eid, reranker.id2entity)[0]
        marker = "← TRUE" if eid == t_true else ""
        print(f"    Rank {rank:2d}: entity {eid:6d} | score {score:8.5f} | {ename[:40]} {marker}")
    
    # چک: آیا scores یکسانند؟
    unique_scores = len(set(bge_scores.values()))
    print(f"\n  تعداد BGE scores یکتا: {unique_scores} از {len(bge_scores)}")
    
    if unique_scores < len(bge_scores) / 2:
        print(f"  ⚠️ اکثر scores یکسانند! احتمالاً مشکل داریم.")
    else:
        print(f"  ✓ scores متنوع هستند.")
    
    # محاسبه rank
    true_score = bge_scores[t_true]
    better = sum(1 for s in bge_scores.values() if s < true_score)
    equal = sum(1 for s in bge_scores.values() if s == true_score)
    rank_with_tie = better + (equal + 1) / 2.0
    
    print(f"\n  Rank محاسبه شده:")
    print(f"    Better than true: {better}")
    print(f"    Equal to true: {equal}")
    print(f"    Final rank: {rank_with_tie:.1f}")
    
    break

print("\n" + "=" * 70)
