#!/usr/bin/env python3
"""
Create entity2id.txt and relation2id.txt for ConvE from train/valid/test files.
"""

import os
from collections import OrderedDict

def create_id_files(base_dir):
    """Create entity2id.txt and relation2id.txt from the triple files."""
    
    files = ['train.txt', 'valid.txt', 'test.txt']
    
    entities = OrderedDict()
    relations = OrderedDict()
    
    # Collect all unique entities and relations
    for filename in files:
        filepath = os.path.join(base_dir, filename)
        print(f"Processing {filepath}...")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    head, rel, tail = parts
                    
                    if head not in entities:
                        entities[head] = len(entities)
                    if tail not in entities:
                        entities[tail] = len(entities)
                    if rel not in relations:
                        relations[rel] = len(relations)
    
    print(f"\nFound {len(entities)} unique entities")
    print(f"Found {len(relations)} unique relations")
    
    # Write entity2id.txt
    entity_file = os.path.join(base_dir, 'entity2id.txt')
    print(f"\nWriting {entity_file}...")
    with open(entity_file, 'w', encoding='utf-8') as f:
        for entity, eid in entities.items():
            f.write(f"{entity}\t{eid}\n")
    
    # Write relation2id.txt
    relation_file = os.path.join(base_dir, 'relation2id.txt')
    print(f"Writing {relation_file}...")
    with open(relation_file, 'w', encoding='utf-8') as f:
        for relation, rid in relations.items():
            f.write(f"{relation}\t{rid}\n")
    
    print("\nDone!")

if __name__ == "__main__":
    base_dir = "/data/ConvE/data/FarsPredict"
    create_id_files(base_dir)
