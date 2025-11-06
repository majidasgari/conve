#!/usr/bin/env python3
"""
Convert FarsPredict dataset from OpenKE format (ID-based) to ConvE format (text-based).
"""

import os

def load_id_mappings(entity_file, relation_file):
    """Load entity and relation ID to name mappings."""
    print(f"Loading entity mappings from {entity_file}...")
    entity_dict = {}
    with open(entity_file, 'r', encoding='utf-8') as f:
        num_entities = int(f.readline().strip())
        print(f"Number of entities: {num_entities}")
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity_name, entity_id = parts
                entity_dict[int(entity_id)] = entity_name
    
    print(f"Loading relation mappings from {relation_file}...")
    relation_dict = {}
    with open(relation_file, 'r', encoding='utf-8') as f:
        num_relations = int(f.readline().strip())
        print(f"Number of relations: {num_relations}")
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                relation_name, relation_id = parts
                relation_dict[int(relation_id)] = relation_name
    
    return entity_dict, relation_dict

def convert_id_file_to_text(id_file, output_file, entity_dict, relation_dict):
    """Convert ID-based triple file to text-based file."""
    print(f"Converting {id_file} to {output_file}...")
    
    with open(id_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # Skip the first line (count)
        first_line = fin.readline().strip()
        num_triples = int(first_line)
        print(f"Number of triples: {num_triples}")
        
        count = 0
        for line in fin:
            parts = line.strip().split()
            if len(parts) == 3:
                head_id, tail_id, rel_id = map(int, parts)
                
                # Get entity and relation names
                head = entity_dict.get(head_id, f"entity_{head_id}")
                tail = entity_dict.get(tail_id, f"entity_{tail_id}")
                rel = relation_dict.get(rel_id, f"relation_{rel_id}")
                
                # Write in ConvE format: head \t relation \t tail
                fout.write(f"{head}\t{rel}\t{tail}\n")
                count += 1
                
                if count % 10000 == 0:
                    print(f"  Processed {count} triples...")
        
        print(f"  Total converted: {count} triples")

def main():
    base_dir = "/data/ConvE/FarsPredict"
    
    # Load mappings
    entity_dict, relation_dict = load_id_mappings(
        os.path.join(base_dir, "entity2id.txt"),
        os.path.join(base_dir, "relation2id.txt")
    )
    
    print(f"\nLoaded {len(entity_dict)} entities and {len(relation_dict)} relations\n")
    
    # Convert train, valid, test files
    convert_id_file_to_text(
        os.path.join(base_dir, "train2id.txt"),
        os.path.join(base_dir, "train.txt"),
        entity_dict, relation_dict
    )
    
    convert_id_file_to_text(
        os.path.join(base_dir, "valid2id.txt"),
        os.path.join(base_dir, "valid.txt"),
        entity_dict, relation_dict
    )
    
    convert_id_file_to_text(
        os.path.join(base_dir, "test2id.txt"),
        os.path.join(base_dir, "test.txt"),
        entity_dict, relation_dict
    )
    
    print("\nConversion completed!")

if __name__ == "__main__":
    main()
