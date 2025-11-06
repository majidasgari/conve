#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
اسکریپت برای ساخت vocabulary بدون training
"""

from spodernet.preprocessing.pipeline import Pipeline, DatasetStreamer
from spodernet.preprocessing.processors import (
    JsonLoaderProcessors, Tokenizer, AddToVocab, SaveLengthsToState, 
    StreamToHDF5, SaveMaxLengthsToState, CustomTokenizer,
    ConvertTokenToIdx, ApplyFunction, ToLower, DictKey2ListMapper
)
from spodernet.utils.global_config import Config

Config.backend = 'pytorch'

dataset_name = 'FB15k-237'

full_path = f'data/{dataset_name}/e1rel_to_e2_full.json'
train_path = f'data/{dataset_name}/e1rel_to_e2_train.json'
dev_ranking_path = f'data/{dataset_name}/e1rel_to_e2_ranking_dev.json'
test_ranking_path = f'data/{dataset_name}/e1rel_to_e2_ranking_test.json'

keys2keys = {}
keys2keys['e1'] = 'e1'
keys2keys['rel'] = 'rel'
keys2keys['rel_eval'] = 'rel'
keys2keys['e2'] = 'e1'
keys2keys['e2_multi1'] = 'e1'
keys2keys['e2_multi2'] = 'e1'
input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']

print("=" * 70)
print("ساخت Vocabulary برای FB15k-237")
print("=" * 70)
print()

d = DatasetStreamer(input_keys)
d.add_stream_processor(JsonLoaderProcessors())
d.add_stream_processor(DictKey2ListMapper(input_keys))

# ساخت vocabulary از فایل full
d.set_path(full_path)
p = Pipeline(dataset_name, delete_data=True, keys=input_keys, skip_transformation=True)
p.add_sent_processor(ToLower())
p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
p.add_token_processor(AddToVocab())

print("در حال پردازش فایل full برای ساخت vocabulary...")
p.execute(d)

print("در حال ذخیره vocabulary...")
p.save_vocabs()

print()
print("✓ Vocabulary با موفقیت ساخته شد!")
print("✓ مسیر: ~/.data/FB15k-237/vocab")
print()

# پردازش فایل‌های train, dev, test
print("در حال پردازش فایل‌های train, dev, test...")
p.skip_transformation = False

for path, name in zip([train_path, dev_ranking_path, test_ranking_path], 
                      ['train', 'dev_ranking', 'test_ranking']):
    print(f"  - پردازش {name}...")
    d.set_path(path)
    p.clear_processors()
    p.add_sent_processor(ToLower())
    p.add_sent_processor(CustomTokenizer(lambda x: x.split(' ')), keys=['e2_multi1', 'e2_multi2'])
    p.add_post_processor(ConvertTokenToIdx(keys2keys=keys2keys), 
                        keys=['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2'])
    p.add_post_processor(StreamToHDF5(name, samples_per_file=1000, keys=input_keys))
    p.execute(d)

print()
print("=" * 70)
print("✓ همه چیز آماده است!")
print("=" * 70)
print()
print("حالا می‌تونید:")
print("  1. مدل را train کنید: python main.py --data FB15k-237")
print("  2. مدل را test کنید: python simple_test.py")
print()
