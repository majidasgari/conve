#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تحلیل Coverage: چند درصد از جواب‌های صحیح تو top-k ConvE هستند؟
"""

import torch
import numpy as np
from model import ConvE
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.utils.global_config import Config

# تنظیمات
Config.backend = 'pytorch'
Config.cuda = True
Config.embedding_dim = 200

print("=" * 70)
print("بارگذاری Dataset و مدل")
print("=" * 70)

# بارگذاری vocabulary
input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
p = Pipeline('FarsPredict', keys=input_keys)
p.load_vocabs()

vocab = p.state['vocab']
num_entities = vocab['e1'].num_token
num_relations = vocab['rel'].num_token

print(f"✓ Entities: {num_entities}, Relations: {num_relations}")

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

# بارگذاری test batcher
test_rank_batcher = StreamBatcher(
    'FarsPredict', 
    'test_ranking', 
    128, 
    randomize=False, 
    loader_threads=4, 
    keys=input_keys
)

print("\n" + "=" * 70)
print("تحلیل Coverage برای k های مختلف")
print("=" * 70)

k_values = [10, 50, 100, 500, 1000]
coverage_stats = {k: {'tail': 0, 'head': 0, 'total': 0} for k in k_values}

total_samples = 0

with torch.no_grad():
    for i, str2var in enumerate(test_rank_batcher):
        e1 = str2var['e1']
        rel = str2var['rel']
        rel_reverse = str2var['rel_eval']
        e2 = str2var['e2']
        
        # پیش‌بینی
        pred1 = model.forward(e1, rel)  # tail prediction
        pred2 = model.forward(e2, rel_reverse)  # head prediction
        pred1, pred2 = pred1.data, pred2.data
        e1, e2 = e1.data, e2.data
        
        batch_size = e1.shape[0]
        
        for batch_idx in range(batch_size):
            h_true = e1[batch_idx, 0].item()
            t_true = e2[batch_idx, 0].item()
            
            # Tail prediction
            kge_scores_t = pred1[batch_idx].cpu().numpy()
            
            # Head prediction
            kge_scores_h = pred2[batch_idx].cpu().numpy()
            
            for k in k_values:
                # چک کردن tail در top-k
                top_k_indices_t = np.argsort(-kge_scores_t)[:k]
                if t_true in top_k_indices_t:
                    coverage_stats[k]['tail'] += 1
                
                # چک کردن head در top-k
                top_k_indices_h = np.argsort(-kge_scores_h)[:k]
                if h_true in top_k_indices_h:
                    coverage_stats[k]['head'] += 1
                
                coverage_stats[k]['total'] += 2  # هم tail هم head
            
            total_samples += 1
        
        if i >= 100:  # فقط 100 batch اول (تقریباً 12800 نمونه)
            break

print(f"\nتعداد کل نمونه‌ها: {total_samples}")
print(f"تعداد کل predictions: {total_samples * 2} (tail + head)\n")

print("Coverage Analysis:")
print("-" * 70)
for k in k_values:
    tail_coverage = coverage_stats[k]['tail'] / total_samples * 100
    head_coverage = coverage_stats[k]['head'] / total_samples * 100
    total_coverage = coverage_stats[k]['tail'] + coverage_stats[k]['head']
    overall_coverage = total_coverage / (total_samples * 2) * 100
    
    print(f"k={k:4d} | Tail: {tail_coverage:5.2f}% | Head: {head_coverage:5.2f}% | Overall: {overall_coverage:5.2f}%")

print("\n" + "=" * 70)
print("نتیجه‌گیری:")
print("=" * 70)
print("اگر Overall Coverage برای k=10 کمتر از 95% باشه،")
print("یعنی BGE نمی‌تونه خیلی از جواب‌های صحیح رو ببینه!")
print("=" * 70)
