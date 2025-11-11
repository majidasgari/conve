#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-ranker برای مدل ConvE با استفاده از BGE-M3 Sentence Embeddings
این کد مشابه RerankTester در OpenKE است اما برای ConvE تطبیق داده شده
"""

import torch
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from itertools import combinations
import torch.nn.functional as F
import os


class ConvEReranker:
    """
    Re-ranker برای ConvE که از BGE sentence embeddings استفاده می‌کند
    """

    def __init__(
        self,
        model=None,  # مدل ConvE
        vocab=None,  # Vocabulary از spodernet
        use_gpu=True,
        k=20,  # تعداد کاندیدهای top-k برای re-rank
        st_model_name="BAAI/bge-m3",  # مدل Sentence Transformer
        bge_score_method="avg_pairwise_sim",
        data_path="data/FB15k-237",
    ):
        self.model = model
        self.vocab = vocab
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.k = k
        self.st_model_name = st_model_name
        self.bge_score_method = bge_score_method
        self.data_path = data_path

        if self.model is None:
            raise ValueError("ConvEReranker requires a ConvE 'model' to be provided.")
        if self.vocab is None:
            raise ValueError("ConvEReranker requires a 'vocab'.")

        # انتقال مدل ConvE به device
        if self.use_gpu:
            print(f"Moving ConvE model to {self.device}")
            self.model.cuda()
        self.model.eval()

        # بارگذاری Sentence Transformer Model
        self.st_model = None
        self._load_st_model()

        # بارگذاری mappings
        self.id2entity = self._load_mapping(
            os.path.join(self.data_path, "entity2id.txt"), "entities"
        )
        self.id2relation = self._load_mapping(
            os.path.join(self.data_path, "relation2id.txt"), "relations"
        )

        # بارگذاری تمام triple های true برای filtering
        self.all_true_triples = self._load_all_true_triples()
        if not self.all_true_triples:
            print("Warning: True triples not loaded; filtering might be incomplete.")

        self.num_entities = vocab['e1'].num_token
        self.num_relations = vocab['rel'].num_token

        # Cache برای embeddings
        self.embedding_cache = {}
        print("Embedding cache initialized.")

    def _load_st_model(self):
        """بارگذاری Sentence Transformer model"""
        print(
            f"Loading Sentence Transformer '{self.st_model_name}' to {self.device}..."
        )
        try:
            self.st_model = SentenceTransformer(self.st_model_name, device=self.device)
            self.st_model.to(self.device)
            self.st_model.eval()
            print(f"Sentence Transformer loaded successfully on {self.device}.")
        except Exception as e:
            print(f"Error loading Sentence Transformer model: {e}")
            self.st_model = None

    def _load_mapping(self, filepath, map_type="entities"):
        """بارگذاری ID to Name mapping"""
        mapping = {}
        print(f"Loading {map_type} mapping from: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    count_line = f.readline()
                    if not count_line:
                        raise ValueError("File is empty or count line missing.")
                    total = int(count_line.strip())
                except ValueError as e:
                    print(f"Error reading count from first line: {e}")
                    total = -1
                    f.seek(0)

                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        name = parts[0]
                        id_str = parts[1]
                        try:
                            mapping[int(id_str)] = name
                        except ValueError:
                            pass

            print(f"Successfully loaded {len(mapping)} {map_type}.")
        except FileNotFoundError:
            print(f"Error: Mapping file not found at {filepath}")
        except Exception as e:
            print(f"Error loading mapping from {filepath}: {e}")
        return mapping

    def _get_names(self, ids, mapping):
        """تبدیل IDs به names"""
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()

        if isinstance(ids, (int, float)):
            ids_list = [int(ids)]
        elif np.isscalar(ids) or (isinstance(ids, np.ndarray) and ids.ndim == 0):
            ids_list = [int(ids.item())]
        elif isinstance(ids, np.ndarray):
            ids_list = [int(id_val) for id_val in ids.flatten().tolist()]
        elif isinstance(ids, list):
            ids_list = [int(id_val) for id_val in ids]
        else:
            print(f"Warning: Unexpected type for ids: {type(ids)}")
            try:
                ids_list = [int(ids)]
            except (TypeError, ValueError):
                print(f"Error: Could not convert input ids to list of integers.")
                return ["ERROR_CONVERTING_ID"]

        return [mapping.get(id_val, f"UNKNOWN_{id_val}") for id_val in ids_list]

    def _load_all_true_triples(self):
        """بارگذاری تمام triple های true"""
        all_triples = set()
        files_to_load = ["train2id.txt", "valid2id.txt", "test2id.txt"]
        print("Loading all true triples for filtering...")
        
        for filename in files_to_load:
            filepath = os.path.join(self.data_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        try:
                            num_lines_str = f.readline().strip()
                            if not num_lines_str:
                                continue
                            num_lines = int(num_lines_str)
                        except (ValueError, TypeError):
                            f.seek(0)
                            num_lines = float("inf")

                        count_read = 0
                        for line in f:
                            if count_read >= num_lines and num_lines != float("inf"):
                                break
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) == 3:
                                    try:
                                        h, t, r = map(int, parts)
                                        all_triples.add((h, r, t))
                                        count_read += 1
                                    except ValueError:
                                        pass
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
            else:
                print(f"Warning: File not found {filepath}")
        
        print(f"Loaded {len(all_triples)} unique true triples for filtering.")
        return all_triples

    def _calculate_bge_scores_batch(self, h_or_t_name, r_name, candidate_ids, id_mapping, is_head_prediction=False):
        """
        محاسبه BGE scores برای چندین کاندید به صورت batch با استفاده از cache
        
        Args:
            h_or_t_name: نام head (برای tail prediction) یا tail (برای head prediction)
            r_name: نام relation
            candidate_ids: لیست IDهای کاندیدها
            id_mapping: دیکشنری mapping از ID به نام
            is_head_prediction: آیا در حال پیش‌بینی head هستیم؟
        
        Returns:
            دیکشنری از {candidate_id: bge_score}
        """
        if not self.st_model:
            return {cid: float("inf") for cid in candidate_ids}

        if self.bge_score_method != "avg_pairwise_sim":
            return {cid: float("inf") for cid in candidate_ids}

        # پاک‌سازی نام‌ها
        base_clean = h_or_t_name.replace("_", " ").replace("/", " ")
        r_clean = r_name.replace("_", " ").replace("/", " ")
        
        try:
            # ساخت تمام textها و نگاشت آنها
            all_texts = []
            text_to_info = []  # (text, cid, text_type) text_type: 0=h, 1=hr, 2=hrt
            texts_needed = set()  # textهایی که نیاز به encode دارند
            
            for cid in candidate_ids:
                cand_name = self._get_names(cid, id_mapping)[0]
                cand_clean = cand_name.replace("_", " ").replace("/", " ")
                
                if is_head_prediction:
                    str_h = cand_clean
                    str_hr = f"{cand_clean} {r_clean}"
                    str_hrt = f"{cand_clean} {r_clean} {base_clean}"
                else:
                    str_h = base_clean
                    str_hr = f"{base_clean} {r_clean}"
                    str_hrt = f"{base_clean} {r_clean} {cand_clean}"
                
                # بررسی کدام textها در cache نیستند
                for text in [str_h, str_hr, str_hrt]:
                    if text not in self.embedding_cache and text not in texts_needed:
                        texts_needed.add(text)
                        all_texts.append(text)
            
            # Encode کردن textهای جدید به صورت batch
            if all_texts:
                embeddings = self.st_model.encode(
                    all_texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                    batch_size=64,
                )
                
                # ذخیره در cache
                for text, emb in zip(all_texts, embeddings):
                    self.embedding_cache[text] = emb
            
            # حالا محاسبه scores
            result = {}
            for cid in candidate_ids:
                cand_name = self._get_names(cid, id_mapping)[0]
                cand_clean = cand_name.replace("_", " ").replace("/", " ")
                
                if is_head_prediction:
                    str_h = cand_clean
                    str_hr = f"{cand_clean} {r_clean}"
                    str_hrt = f"{cand_clean} {r_clean} {base_clean}"
                else:
                    str_h = base_clean
                    str_hr = f"{base_clean} {r_clean}"
                    str_hrt = f"{base_clean} {r_clean} {cand_clean}"
                
                # دریافت embeddings از cache
                emb_h = self.embedding_cache[str_h]
                emb_hr = self.embedding_cache[str_hr]
                emb_hrt = self.embedding_cache[str_hrt]
                
                # محاسبه pairwise similarities
                similarities = []
                for emb_i, emb_j in [(emb_h, emb_hr), (emb_h, emb_hrt), (emb_hr, emb_hrt)]:
                    sim = F.cosine_similarity(
                        emb_i.unsqueeze(0), 
                        emb_j.unsqueeze(0)
                    )
                    similarities.append(sim.item())
                
                average_sim = np.mean(similarities) if similarities else 0.0
                result[cid] = -average_sim  # Lower score is better
            
            return result
            
        except Exception as e:
            print(f"\nWarn: Batch BGE score calc failed: {e}")
            return {cid: float("inf") for cid in candidate_ids}

    def _calculate_bge_score(self, h_name, r_name, t_name):
        """محاسبه BGE score برای یک triple (برای backward compatibility)"""
        if not self.st_model:
            return float("inf")

        if self.bge_score_method == "avg_pairwise_sim":
            # پاک‌سازی نام‌ها
            h_clean = h_name.replace("_", " ").replace("/", " ")
            r_clean = r_name.replace("_", " ").replace("/", " ")
            t_clean = t_name.replace("_", " ").replace("/", " ")
            
            str_h = h_clean
            str_hr = f"{h_clean} {r_clean}"
            str_hrt = f"{h_clean} {r_clean} {t_clean}"
            texts_to_encode = [str_h, str_hr, str_hrt]
            
            try:
                embeddings = self.st_model.encode(
                    texts_to_encode,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                )
                similarities = []
                for i, j in combinations(range(embeddings.shape[0]), 2):
                    sim = F.cosine_similarity(
                        embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
                    )
                    similarities.append(sim.item())
                average_sim = np.mean(similarities) if similarities else 0.0
                return -average_sim  # Lower score is better
            except Exception as e:
                print(f"\nWarn: BGE score calc failed for ('{h_name}', '{r_name}', '{t_name}'): {e}")
                return float("inf")
        else:
            print(f"Error: Unknown bge_score_method '{self.bge_score_method}'")
            return float("inf")

    def ranking_and_hits_with_reranking(self, dev_rank_batcher, name):
        """
        ارزیابی با re-ranking
        مشابه تابع ranking_and_hits ولی با BGE re-ranking
        """
        if self.st_model is None:
            print("Error: Sentence Transformer model failed to load. Cannot perform re-ranking.")
            return

        print("\n" + "=" * 70)
        print(f"{name} - با BGE Re-ranking (Top-{self.k})")
        print("=" * 70)

        hits_left = []
        hits_right = []
        hits = []
        ranks = []
        ranks_left = []
        ranks_right = []
        
        for i in range(10):
            hits_left.append([])
            hits_right.append([])
            hits.append([])

        all_entity_ids = np.arange(self.num_entities, dtype=np.int64)

        for i, str2var in enumerate(tqdm(dev_rank_batcher, desc="Re-ranking Progress")):
            e1 = str2var['e1']
            e2 = str2var['e2']
            rel = str2var['rel']
            rel_reverse = str2var['rel_eval']
            e2_multi1 = str2var['e2_multi1'].float()
            e2_multi2 = str2var['e2_multi2'].float()

            # پیش‌بینی با مدل ConvE
            with torch.no_grad():
                pred1 = self.model.forward(e1, rel)
                pred2 = self.model.forward(e2, rel_reverse)
                pred1, pred2 = pred1.data, pred2.data
                e1, e2 = e1.data, e2.data
                e2_multi1, e2_multi2 = e2_multi1.data, e2_multi2.data

            # پردازش هر نمونه در batch
            for batch_idx in range(e1.shape[0]):
                filter1 = e2_multi1[batch_idx].long()
                filter2 = e2_multi2[batch_idx].long()

                # --- Right/Tail Prediction (پیش‌بینی t با داشتن h و r) ---
                h_true = e1[batch_idx, 0].item()
                t_true = e2[batch_idx, 0].item()
                r_true = rel[batch_idx, 0].item()

                # دریافت KGE scores
                kge_scores_t = pred1[batch_idx].cpu().numpy()
                
                # پیدا کردن top-k کاندیدها
                top_k_indices_t = np.argsort(-kge_scores_t)[:self.k]
                top_k_tail_ids = all_entity_ids[top_k_indices_t]

                # آماده‌سازی نام‌ها
                h_name = self._get_names(h_true, self.id2entity)[0]
                r_name = self._get_names(r_true, self.id2relation)[0]

                # محاسبه BGE scores برای top-k و جواب صحیح (batch processing)
                final_scores_t = np.full(self.num_entities, float("inf"))
                ids_to_score_bge_t = set(top_k_tail_ids)
                ids_to_score_bge_t.add(t_true)

                # Batch processing برای BGE
                bge_score_map_t = self._calculate_bge_scores_batch(
                    h_name, r_name, list(ids_to_score_bge_t), self.id2entity
                )

                # اختصاص BGE scores
                for tail_id, score in bge_score_map_t.items():
                    if 0 <= tail_id < self.num_entities:
                        final_scores_t[tail_id] = score

                # Filtering: صفر کردن triple های true دیگر
                true_score_t = final_scores_t[t_true]
                filtered_final_scores_t = final_scores_t.copy()
                
                for other_t in range(self.num_entities):
                    if other_t != t_true and (h_true, r_true, other_t) in self.all_true_triples:
                        filtered_final_scores_t[other_t] = float("inf")

                # محاسبه rank
                rank_t = np.sum(filtered_final_scores_t < true_score_t) + 1
                ranks.append(rank_t)
                ranks_left.append(rank_t)

                for hits_level in range(10):
                    if rank_t <= hits_level + 1:
                        hits[hits_level].append(1.0)
                        hits_left[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                        hits_left[hits_level].append(0.0)

                # --- Left/Head Prediction (پیش‌بینی h با داشتن t و r_reverse) ---
                kge_scores_h = pred2[batch_idx].cpu().numpy()
                
                # پیدا کردن top-k کاندیدها
                top_k_indices_h = np.argsort(-kge_scores_h)[:self.k]
                top_k_head_ids = all_entity_ids[top_k_indices_h]

                # آماده‌سازی نام‌ها
                t_name = self._get_names(t_true, self.id2entity)[0]
                r_reverse_val = rel_reverse[batch_idx, 0].item()
                r_name_reverse = self._get_names(r_reverse_val, self.id2relation)[0]

                # محاسبه BGE scores (batch processing)
                final_scores_h = np.full(self.num_entities, float("inf"))
                ids_to_score_bge_h = set(top_k_head_ids)
                ids_to_score_bge_h.add(h_true)

                # Batch processing برای BGE
                bge_score_map_h = self._calculate_bge_scores_batch(
                    t_name, r_name_reverse, list(ids_to_score_bge_h), self.id2entity, is_head_prediction=True
                )

                # اختصاص BGE scores
                for head_id, score in bge_score_map_h.items():
                    if 0 <= head_id < self.num_entities:
                        final_scores_h[head_id] = score

                # Filtering
                true_score_h = final_scores_h[h_true]
                filtered_final_scores_h = final_scores_h.copy()
                
                for other_h in range(self.num_entities):
                    if other_h != h_true and (other_h, r_true, t_true) in self.all_true_triples:
                        filtered_final_scores_h[other_h] = float("inf")

                # محاسبه rank
                rank_h = np.sum(filtered_final_scores_h < true_score_h) + 1
                ranks.append(rank_h)
                ranks_right.append(rank_h)

                for hits_level in range(10):
                    if rank_h <= hits_level + 1:
                        hits[hits_level].append(1.0)
                        hits_right[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)
                        hits_right[hits_level].append(0.0)

            # Set loss state if available (for compatibility with regular batchers)
            if hasattr(dev_rank_batcher, 'state'):
                dev_rank_batcher.state.loss = [0]

        # چاپ نتایج
        print("\n" + "=" * 70)
        print("Results with BGE Re-ranking:")
        print("=" * 70)
        print(f'Cache Statistics: {len(self.embedding_cache)} unique embeddings cached')
        print("=" * 70)
        for i in range(10):
            print(f'Hits left @{i+1}: {np.mean(hits_left[i]):.4f}')
            print(f'Hits right @{i+1}: {np.mean(hits_right[i]):.4f}')
            print(f'Hits @{i+1}: {np.mean(hits[i]):.4f}')
        print(f'Mean rank left: {np.mean(ranks_left):.2f}')
        print(f'Mean rank right: {np.mean(ranks_right):.2f}')
        print(f'Mean rank: {np.mean(ranks):.2f}')
        print(f'Mean reciprocal rank left: {np.mean(1./np.array(ranks_left)):.4f}')
        print(f'Mean reciprocal rank right: {np.mean(1./np.array(ranks_right)):.4f}')
        print(f'MRR: {np.mean(1./np.array(ranks)):.4f}')
        print("=" * 70 + "\n")

        return np.mean(1./np.array(ranks)), np.mean(ranks), np.mean(hits[9])
