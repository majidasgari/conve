#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
اسکریپت برای لود کردن و تست مدل آموزش‌دیده ConvE
"""

import json
import torch
import pickle
import numpy as np
import argparse
import sys
import os
import datetime

from evaluation import ranking_and_hits
from model import ConvE, DistMult, Complex

from spodernet.preprocessing.pipeline import Pipeline
from spodernet.utils.global_config import Config
from spodernet.preprocessing.batching import StreamBatcher

np.set_printoptions(precision=3)


def load_and_test_model(args):
    """
    تابع اصلی برای لود کردن و تست مدل
    """
    print("=" * 70)
    print("شروع تست مدل")
    print("=" * 70)
    
    # بارگذاری vocab
    input_keys = ['e1', 'rel', 'rel_eval', 'e2', 'e2_multi1', 'e2_multi2']
    p = Pipeline(args.data, keys=input_keys)
    
    # بررسی اینکه آیا vocab وجود دارد یا نه
    try:
        p.load_vocabs()
        vocab = p.state['vocab']
    except Exception as e:
        print(f"\n❌ خطا در بارگذاری vocabulary: {e}")
        print("\n💡 لطفاً ابتدا dataset را preprocess کنید:")
        print(f"   python main.py --data {args.data} --preprocess")
        return
    
    num_entities = vocab['e1'].num_token
    num_relations = vocab['rel'].num_token
    
    # بررسی صحت vocabulary
    if num_entities < 100 or num_relations < 10:
        print(f"\n❌ خطا: Vocabulary معتبر نیست!")
        print(f"   تعداد Entities: {num_entities} (خیلی کم است!)")
        print(f"   تعداد Relations: {num_relations} (خیلی کم است!)")
        print("\n💡 Vocabulary باید دوباره ساخته شود. لطفاً دستور زیر را اجرا کنید:")
        print(f"   python main.py --data {args.data} --preprocess")
        return
    
    print(f"\n✓ تعداد Entities: {num_entities}")
    print(f"✓ تعداد Relations: {num_relations}")
    
    # ایجاد batch loaders برای dev و test
    dev_rank_batcher = StreamBatcher(
        args.data, 
        'dev_ranking', 
        args.test_batch_size, 
        randomize=False, 
        loader_threads=args.loader_threads, 
        keys=input_keys
    )
    
    test_rank_batcher = StreamBatcher(
        args.data, 
        'test_ranking', 
        args.test_batch_size, 
        randomize=False, 
        loader_threads=args.loader_threads, 
        keys=input_keys
    )
    
    # ایجاد مدل
    print(f"\nایجاد مدل: {args.model}")
    if args.model == 'conve':
        model = ConvE(args, num_entities, num_relations)
    elif args.model == 'distmult':
        model = DistMult(args, num_entities, num_relations)
    elif args.model == 'complex':
        model = Complex(args, num_entities, num_relations)
    else:
        raise Exception(f"مدل ناشناخته: {args.model}")
    
    # بارگذاری وزن‌های مدل
    print(f"\nبارگذاری مدل از: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"خطا: فایل مدل یافت نشد: {args.model_path}")
        print("\nمدل‌های موجود در پوشه saved_models:")
        if os.path.exists('saved_models'):
            for f in os.listdir('saved_models'):
                if f.endswith('.model'):
                    print(f"  - {f}")
        return
    
    model_params = torch.load(args.model_path)
    model.load_state_dict(model_params)
    
    # انتقال مدل به GPU
    if args.cuda and torch.cuda.is_available():
        model.cuda()
        print("مدل به GPU منتقل شد")
    else:
        print("مدل روی CPU اجرا می‌شود")
    
    # تنظیم مدل در حالت evaluation
    model.eval()
    
    # نمایش اطلاعات مدل
    print("\nاطلاعات مدل:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"تعداد کل پارامترها: {total_params:,}")
    
    print("\nساختار مدل:")
    print(model)
    
    # ذخیره نتایج در فایل
    results_file = f"test_results_{args.data}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # تست روی test set
    print("\n" + "=" * 70)
    print("ارزیابی روی Test Set")
    print("=" * 70)
    with torch.no_grad():
        test_results = ranking_and_hits(model, test_rank_batcher, vocab, 'Test Evaluation')
    
    # تست روی dev set
    print("\n" + "=" * 70)
    print("ارزیابی روی Dev Set")
    print("=" * 70)
    with torch.no_grad():
        dev_results = ranking_and_hits(model, dev_rank_batcher, vocab, 'Dev Evaluation')
    
    print("\n" + "=" * 70)
    print("تست با موفقیت به پایان رسید!")
    print(f"نتایج در فایل log ذخیره شده است")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='تست مدل آموزش‌دیده Knowledge Graph')
    
    # پارامترهای مدل
    parser.add_argument('--model', type=str, default='conve', 
                        help='نوع مدل: {conve, distmult, complex}')
    parser.add_argument('--data', type=str, default='FB15k-237', 
                        help='Dataset: {FB15k-237, YAGO3-10, WN18RR, umls, nations, kinship}')
    parser.add_argument('--model-path', type=str, default='saved_models/FB15k-237_conve_0.2_0.3.model',
                        help='مسیر فایل مدل ذخیره‌شده')
    
    # پارامترهای معماری مدل (باید با مدل آموزش‌داده شده همخوانی داشته باشد)
    parser.add_argument('--embedding-dim', type=int, default=200,
                        help='بعد embedding (پیش‌فرض: 200)')
    parser.add_argument('--embedding-shape1', type=int, default=20,
                        help='بعد اول embedding 2D (پیش‌فرض: 20)')
    parser.add_argument('--hidden-drop', type=float, default=0.3,
                        help='Dropout برای hidden layer (پیش‌فرض: 0.3)')
    parser.add_argument('--input-drop', type=float, default=0.2,
                        help='Dropout برای input embeddings (پیش‌فرض: 0.2)')
    parser.add_argument('--feat-drop', type=float, default=0.2,
                        help='Dropout برای convolutional features (پیش‌فرض: 0.2)')
    parser.add_argument('--hidden-size', type=int, default=9728,
                        help='اندازه hidden layer (پیش‌فرض: 9728)')
    parser.add_argument('--use-bias', action='store_true',
                        help='استفاده از bias در convolutional layer')
    
    # پارامترهای تست
    parser.add_argument('--test-batch-size', type=int, default=128,
                        help='اندازه batch برای تست (پیش‌فرض: 128)')
    parser.add_argument('--loader-threads', type=int, default=4,
                        help='تعداد thread برای batch loader (پیش‌فرض: 4)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='استفاده از CUDA')
    parser.add_argument('--seed', type=int, default=17,
                        help='random seed (پیش‌فرض: 17)')
    
    args = parser.parse_args()
    
    # تنظیم global config
    Config.backend = 'pytorch'
    Config.cuda = args.cuda and torch.cuda.is_available()
    Config.embedding_dim = args.embedding_dim
    
    # تنظیم random seed
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # اجرای تست
    load_and_test_model(args)
