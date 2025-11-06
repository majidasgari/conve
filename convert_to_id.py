#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
تبدیل فایل‌های train/valid/test به فرمت ID
"""

import os

def load_mapping(filepath):
    """بارگذاری mapping از فایل"""
    mapping = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        f.readline()  # خط اول را نادیده بگیر
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                name = parts[0]
                id_val = int(parts[1])
                mapping[name] = id_val
    return mapping

def convert_to_id(input_file, output_file, entity2id, relation2id):
    """تبدیل فایل به فرمت ID"""
    triples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                h, r, t = parts
                h_id = entity2id.get(h, -1)
                r_id = relation2id.get(r, -1)
                t_id = entity2id.get(t, -1)
                if h_id != -1 and r_id != -1 and t_id != -1:
                    triples.append((h_id, t_id, r_id))
    
    # نوشتن به فایل
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"{len(triples)}\n")
        for h, t, r in triples:
            f.write(f"{h} {t} {r}\n")
    
    print(f"تبدیل شد: {input_file} -> {output_file} ({len(triples)} triples)")

if __name__ == "__main__":
    data_path = "data/FB15k-237"
    
    # بارگذاری mappings
    print("بارگذاری mappings...")
    entity2id = load_mapping(os.path.join(data_path, "entity2id.txt"))
    relation2id = load_mapping(os.path.join(data_path, "relation2id.txt"))
    
    print(f"تعداد entities: {len(entity2id)}")
    print(f"تعداد relations: {len(relation2id)}")
    
    # تبدیل فایل‌ها
    convert_to_id(
        os.path.join(data_path, "train.txt"),
        os.path.join(data_path, "train2id.txt"),
        entity2id, relation2id
    )
    
    convert_to_id(
        os.path.join(data_path, "valid.txt"),
        os.path.join(data_path, "valid2id.txt"),
        entity2id, relation2id
    )
    
    convert_to_id(
        os.path.join(data_path, "test.txt"),
        os.path.join(data_path, "test2id.txt"),
        entity2id, relation2id
    )
    
    print("\nتمام!")
