#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script: چک کردن محاسبات rank
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
print("بارگذاری FarsPredict")
print("=" * 70)

input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
p = Pipeline('FarsPredict', keys=input_keys)
p.load_vocabs()

vocab = p.state['vocab']
num_entities = vocab['e1'].num_token
num_relations = vocab['rel'].num_token

# ایجاد مدل
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
print("تحلیل یک batch برای debug")
print("=" * 70)

all_entity_ids = np.arange(num_entities)
k = 10

# گرفتن اولین batch
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
    
    # پیدا کردن top-k
    top_k_indices_t = np.argsort(-kge_scores_t)[:k]
    top_k_tail_ids = all_entity_ids[top_k_indices_t]
    
    print(f"\nنمونه اول:")
    print(f"  h_true = {h_true}")
    print(f"  r_true = {r_true}")
    print(f"  t_true = {t_true}")
    print(f"\n  Top-{k} predictions from ConvE:")
    for i, idx in enumerate(top_k_indices_t[:10]):
        print(f"    Rank {i+1}: entity {idx}, score {kge_scores_t[idx]:.6f}")
    
    print(f"\n  آیا t_true در top-{k} است؟")
    if t_true in top_k_tail_ids:
        position = np.where(top_k_tail_ids == t_true)[0][0] + 1
        print(f"    ✓ بله! در رتبه {position} از top-{k}")
    else:
        print(f"    ✗ خیر! جواب صحیح در top-{k} نیست")
        # پیدا کردن رتبه واقعی
        sorted_indices = np.argsort(-kge_scores_t)
        true_rank = np.where(sorted_indices == t_true)[0][0] + 1
        print(f"    رتبه واقعی در ConvE: {true_rank}")
        print(f"    score جواب صحیح: {kge_scores_t[t_true]:.6f}")
    
    # شبیه‌سازی محاسبه rank با کد فعلی
    final_scores_t = np.full(num_entities, float("inf"))
    ids_to_score = set(top_k_tail_ids)
    ids_to_score.add(t_true)
    
    # فرض کنیم همه BGE scores برابر 0.5 هستند (برای تست)
    for entity_id in ids_to_score:
        final_scores_t[entity_id] = 0.5  # همه یکسان
    
    true_score_t = final_scores_t[t_true]
    print(f"\n  true_score_t = {true_score_t}")
    
    # محاسبه rank
    rank_t = np.sum(final_scores_t < true_score_t) + 1
    print(f"  محاسبه rank:")
    print(f"    تعداد entities با score < {true_score_t}: {np.sum(final_scores_t < true_score_t)}")
    print(f"    rank = {rank_t}")
    
    # چک: چند entity score دارند (نه inf)
    num_scored = np.sum(final_scores_t != float("inf"))
    print(f"\n  تعداد entities که score دارند (نه inf): {num_scored}")
    print(f"  پس rank ماکزیمم می‌تواند باشد: {num_scored}")
    
    break

print("\n" + "=" * 70)
print("نتیجه‌گیری:")
print("=" * 70)
print("اگر همه scores یکسان باشند (0.5)، rank همیشه 1 میشه!")
print("چون هیچ entity ای score کمتر از 0.5 نداره.")
print("=" * 70)
