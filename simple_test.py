#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
اسکریپت ساده برای تست سریع مدل
"""

import torch
from model import ConvE
from spodernet.preprocessing.pipeline import Pipeline
from spodernet.preprocessing.batching import StreamBatcher
from spodernet.utils.global_config import Config
from evaluation import ranking_and_hits

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

# سعی در بارگذاری vocabulary
try:
    p.load_vocabs()
except:
    print("\n⚠️  Vocabulary outdated یا وجود ندارد!")
    print("در حال rebuild کردن vocabulary از فایل‌های JSON...")
    import sys
    sys.exit(1)

vocab = p.state['vocab']

num_entities = vocab['e1'].num_token
num_relations = vocab['rel'].num_token

# بررسی صحت vocabulary
if num_entities < 100 or num_relations < 10:
    print(f"\n❌ خطا: Vocabulary معتبر نیست!")
    print(f"   تعداد Entities: {num_entities} (باید 14543 باشد)")
    print(f"   تعداد Relations: {num_relations} (باید 476 باشد)")
    print("\n💡 لطفاً مراحل زیر را انجام دهید:")
    print("   1. فایل‌های vocabulary قدیمی را پاک کنید:")
    print("      rm -rf ~/.data/FB15k-237/vocab*")
    print("   2. دوباره preprocessing را اجرا کنید (فقط تا vocab ساخته شود):")
    print("      python main.py --data FB15k-237 --preprocess")
    print("      (بعد از اینکه vocab ساخته شد، با Ctrl+C متوقفش کنید)")
    import sys
    sys.exit(1)

print(f"✓ تعداد Entities: {num_entities}")
print(f"✓ تعداد Relations: {num_relations}")

# بارگذاری batch loaders
test_rank_batcher = StreamBatcher(
    'FB15k-237', 
    'test_ranking', 
    128, 
    randomize=False, 
    loader_threads=4, 
    keys=input_keys
)

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
    use_bias = False  # مدل بدون bias ذخیره شده است

args = Args()

# ایجاد مدل
model = ConvE(args, num_entities, num_relations)
print(model)

# بارگذاری وزن‌ها
model_path = 'saved_models/FB15k-237_conve_0.2_0.3.model'
print(f"\n بارگذاری وزن‌ها از: {model_path}")
model_params = torch.load(model_path, weights_only=False)
model.load_state_dict(model_params)

# انتقال به GPU
model.cuda()
model.eval()

print("\n" + "=" * 70)
print("شروع ارزیابی Test Set")
print("=" * 70)

# تست
with torch.no_grad():
    ranking_and_hits(model, test_rank_batcher, vocab, 'Test Evaluation')

print("\n" + "=" * 70)
print("تست تکمیل شد!")
print("=" * 70)
