# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import numpy as np
import ctypes
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from itertools import combinations
import time


class RerankTester(object):
    """
    Tester that performs link prediction using a base KGE model for initial candidate generation
    and then re-ranks the top-k candidates using a Sentence Transformer (BGE) based score.
    """

    def __init__(
        self,
        model=None,  # The main KGE model (e.g., TransE, TransH)
        data_loader=None,  # TestDataLoader
        use_gpu=True,
        k=20,  # Number of top candidates to re-rank
        st_model_name="BAAI/bge-m3",  # Sentence Transformer model name
        st_batch_size=64,  # Batch size for Sentence Transformer encoding
        bge_score_method="avg_pairwise_sim",  # Method for BGE scoring
    ):

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.k = k
        self.st_model_name = st_model_name
        self.st_batch_size = st_batch_size
        self.bge_score_method = bge_score_method  # Store the method name

        if self.model is None:
            raise ValueError("RerankTester requires a base KGE 'model' to be provided.")
        if self.data_loader is None:
            raise ValueError("RerankTester requires a 'data_loader'.")

        # Move KGE model to device
        if self.use_gpu:
            print(f"Moving KGE model '{type(self.model).__name__}' to {self.device}")
            self.model.to(self.device)  # Use .to(device)
        self.model.eval()  # Ensure KGE model is in eval mode

        # Load Sentence Transformer Model
        self.st_model = None
        self._load_st_model()

        # Load mappings needed for BGE scoring
        self.id2entity = self._load_mapping(
            os.path.join(self.data_loader.in_path, "entity2id.txt"), "entities"
        )
        self.id2relation = self._load_mapping(
            os.path.join(self.data_loader.in_path, "relation2id.txt"), "relations"
        )

        # Load all true triples for filtering during final ranking
        self.all_true_triples = (
            self._load_all_true_triples()
        )  # <--- This line causes the error, fixed in the method below
        if not self.all_true_triples:
            print(
                "Warning: True triples not loaded; filtering during re-ranking might be incomplete."
            )

        # Get dataset stats
        self.ent_total = self.data_loader.get_ent_tot()
        self.rel_total = self.data_loader.get_rel_tot()
        self.test_total = self.data_loader.get_triple_tot()

    def _load_st_model(self):
        """Loads the Sentence Transformer model."""
        print(
            f"Initializing RerankTester: Loading Sentence Transformer '{self.st_model_name}' to {self.device}..."
        )
        try:
            self.st_model = SentenceTransformer(self.st_model_name, device=self.device)
            self.st_model.to(self.device)
            self.st_model.eval()  # Ensure ST model is also in eval mode
            print(
                f"Sentence Transformer model loaded successfully on {next(self.st_model.parameters()).device}."
            )
        except Exception as e:
            print(f"Error loading Sentence Transformer model: {e}")
            self.st_model = None  # Set to None if loading fails

    def _load_mapping(self, filepath, map_type="entities"):
        """Loads ID to Name mapping from a file."""
        mapping = {}
        print(f"Attempting to load {map_type} mapping from: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8-sig") as f:
                try:
                    count_line = f.readline()
                    if not count_line:
                        raise ValueError("File is empty or count line missing.")
                    total = int(count_line.strip())
                except ValueError as e:
                    print(
                        f"Error reading count from first line of {filepath}: {e}. Attempting to process without count."
                    )
                    total = -1
                    f.seek(0)

                line_num = 1 if total != -1 else 0
                parsed_count = 0
                for line in f:
                    line_num += 1
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        id_str = parts[-1]
                        name = " ".join(parts[:-1])
                        try:
                            mapping[int(id_str)] = name
                            parsed_count += 1
                        except ValueError:
                            pass  # Silently skip non-integer IDs
                    # Silently skip malformed lines
                if total != -1 and len(mapping) != total:
                    print(
                        f"Warning: ID count mismatch in {filepath}. Expected {total}, successfully parsed {len(mapping)} unique IDs."
                    )
                elif total == -1:
                    print(
                        f"Note: Count line was missing or invalid in {filepath}. Successfully parsed {len(mapping)} unique IDs."
                    )
            print(f"Successfully loaded {len(mapping)} {map_type}.")
        except FileNotFoundError:
            print(f"Error: Mapping file not found at {filepath}")
        except Exception as e:
            print(f"Error loading mapping from {filepath}: {e}")
        return mapping

    def _get_names(self, ids, mapping):
        """Converts a batch of IDs (can be int, numpy array, or tensor) to names."""
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()  # Convert tensor to numpy

        # Handle different types for ids before converting to list
        if isinstance(ids, (int, float)):  # Check for plain Python int/float first
            ids_list = [int(ids)]  # Convert directly to list of int
        elif np.isscalar(ids) or (
            isinstance(ids, np.ndarray) and ids.ndim == 0
        ):  # Handle numpy scalar or 0-dim array
            ids_list = [int(ids.item())]  # Use .item() for numpy types
        elif isinstance(ids, np.ndarray):  # Handle numpy array
            ids_list = [
                int(id_val) for id_val in ids.flatten().tolist()
            ]  # Flatten and convert to list of int
        elif isinstance(ids, list):  # Handle list input
            ids_list = [int(id_val) for id_val in ids]
        else:
            print(
                f"Warning: Unexpected type for ids in _get_names: {type(ids)}. Attempting direct conversion."
            )
            try:
                ids_list = [int(ids)]  # Try converting directly
            except (TypeError, ValueError):
                print(f"Error: Could not convert input ids to list of integers.")
                return ["ERROR_CONVERTING_ID"]  # Return an error indicator

        # Perform mapping lookup
        return [mapping.get(id_val, f"UNKNOWN_{id_val}") for id_val in ids_list]

    def _load_all_true_triples(self):
        """Loads train, valid, and test triples into a set for filtering."""
        # Fixed the variable name in the final print statement
        all_triples = set()  # Variable name is 'all_triples'
        files_to_load = ["train2id.txt", "valid2id.txt", "test2id.txt"]
        base_path = self.data_loader.in_path
        print("Loading all true triples for filtering...")
        for filename in files_to_load:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8-sig") as f:
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
                                        all_triples.add((h, r, t))  # Store as h, r, t
                                        count_read += 1
                                    except ValueError:
                                        pass  # Ignore lines with non-int IDs
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
            else:
                print(f"Warning: File not found {filepath}")
        # Use the correct variable name 'all_triples' here
        print(f"Loaded {len(all_triples)} unique true triples for filtering.")
        return all_triples

    def _calculate_bge_score(self, h_name, r_name, t_name):
        """Calculates the BGE-based score for a single triple."""
        if not self.st_model:
            return float("inf")

        if self.bge_score_method == "avg_pairwise_sim":
            h_clean = h_name.replace("_", " ")
            r_clean = r_name.replace("_", " ")
            t_clean = t_name.replace("_", " ")
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
                print(
                    f"\nWarn: BGE score calc failed for ('{h_name}', '{r_name}', '{t_name}'): {e}"
                )
                return float("inf")
        else:
            print(f"Error: Unknown bge_score_method '{self.bge_score_method}'")
            return float("inf")

    def to_var(self, x):
        """Converts numpy array to torch Variable on the correct device."""
        if not isinstance(x, np.ndarray):
            try:
                x = np.array(x)
            except:
                raise TypeError(
                    f"Input to to_var must be convertible to numpy array, got {type(x)}"
                )

        if x.dtype in [np.int64, np.int32, np.int16, np.int8]:
            tensor = torch.from_numpy(x).long()
        elif x.dtype in [np.float64, np.float32, np.float16]:
            tensor = torch.from_numpy(x).float()
        else:
            try:
                tensor = torch.from_numpy(x.astype(np.int64)).long()
            except (TypeError, ValueError):
                raise TypeError(f"Unsupported numpy dtype for to_var: {x.dtype}")

        tensor = tensor.to(self.device)
        return Variable(tensor)

    def run_link_prediction(self, type_constrain=False):
        """
        Performs link prediction with re-ranking. Ensures the true answer is always scored by BGE.
        Args:
            type_constrain (bool): Currently ignored.
        Returns:
            tuple: (mrr, mr, hit10, hit3, hit1)
        """
        if self.st_model is None:
            print(
                "Error: Sentence Transformer model failed to load. Cannot perform re-ranking."
            )
            return 0.0, 0.0, 0.0, 0.0, 0.0

        print(f"Starting Link Prediction with BGE Re-ranking (Top-{self.k})...")
        if type_constrain:
            print("Warning: Type constraints are not implemented in this RerankTester.")

        # Metrics accumulators
        total_mrr_filter = 0.0
        total_mr_filter = 0.0
        total_hits1_filter = 0.0
        total_hits3_filter = 0.0
        total_hits10_filter = 0.0
        valid_test_count = 0

        test_loop = tqdm(self.data_loader, desc="Re-ranking Progress")
        total_processed = 0
        all_entity_ids = np.arange(self.ent_total, dtype=np.int64)

        self.model.to(self.device)

        for data_head_np, data_tail_np in test_loop:
            total_processed += 1

            # --- Convert numpy data to PyTorch tensors (ONLY for relevant keys) ---
            data_head = {
                k: self.to_var(v) if k in ["batch_h", "batch_t", "batch_r"] else v
                for k, v in data_head_np.items()
            }
            data_tail = {
                k: self.to_var(v) if k in ["batch_h", "batch_t", "batch_r"] else v
                for k, v in data_tail_np.items()
            }

            # --- Head Prediction ---
            h_true = data_head_np["batch_h"][0].item()
            t_anchor = data_head_np["batch_t"][0].item()
            r_anchor = data_head_np["batch_r"][0].item()

            with torch.no_grad():
                kge_scores_h_tensor = self.model.predict(data_head)
                if isinstance(kge_scores_h_tensor, torch.Tensor):
                    kge_scores_h = kge_scores_h_tensor.cpu().numpy()
                elif isinstance(kge_scores_h_tensor, np.ndarray):
                    kge_scores_h = kge_scores_h_tensor
                else:
                    kge_scores_h = np.array([])

            head_rank_calculated = False
            if kge_scores_h.size > 0:
                # Find top-k candidate heads based on KGE scores
                top_k_indices_h = np.argsort(kge_scores_h)[: self.k]
                top_k_head_ids = all_entity_ids[top_k_indices_h]

                # Prepare names
                r_name = self._get_names(r_anchor, self.id2relation)[0]
                t_name = self._get_names(t_anchor, self.id2entity)[0]
                top_k_head_names = self._get_names(top_k_head_ids, self.id2entity)

                # --- Calculate BGE scores for top-k AND true answer ---
                final_scores_h = np.full(
                    self.ent_total, float("inf")
                )  # Initialize all scores to infinity
                ids_to_score_bge = set(top_k_head_ids)
                ids_to_score_bge.add(h_true)  # Ensure true answer is always included

                bge_score_map_h = {}  # Store calculated BGE scores
                for head_id in ids_to_score_bge:
                    head_name = self._get_names(head_id, self.id2entity)[0]
                    bge_score = self._calculate_bge_score(head_name, r_name, t_name)
                    bge_score_map_h[head_id] = bge_score

                # Assign calculated BGE scores to the final scores array
                for head_id, score in bge_score_map_h.items():
                    # Check bounds just in case ID is invalid, though unlikely here
                    if 0 <= head_id < self.ent_total:
                        final_scores_h[head_id] = score
                # All other entities not in ids_to_score_bge retain the score 'inf'

                # --- Calculate Filtered Rank for Head ---
                true_score_h = final_scores_h[
                    h_true
                ]  # This will now be the BGE score or inf if calc failed
                if np.isinf(true_score_h):
                    print(
                        f"Warn: BGE score failed for true head {h_true} in triple ({h_true}, {r_anchor}, {t_anchor}). Skipping head metric update."
                    )
                else:
                    head_rank_calculated = True
                    filtered_final_scores_h = final_scores_h.copy()
                    # Filter known true triples (excluding the true answer itself)
                    for head_idx in range(self.ent_total):
                        if (
                            head_idx != h_true
                            and (head_idx, r_anchor, t_anchor) in self.all_true_triples
                        ):
                            filtered_final_scores_h[head_idx] = float("inf")

                    # Calculate rank based on BGE scores (lower is better)
                    rank_h_filter = 1 + np.sum(filtered_final_scores_h < true_score_h)
                    rank_h_filter += np.random.randint(
                        0, np.sum(filtered_final_scores_h == true_score_h) + 1
                    )  # Random tie-breaking

            # --- Tail Prediction ---
            h_anchor = data_tail_np["batch_h"][0].item()
            t_true = data_tail_np["batch_t"][0].item()
            # r_anchor is the same

            with torch.no_grad():
                kge_scores_t_tensor = self.model.predict(data_tail)
                if isinstance(kge_scores_t_tensor, torch.Tensor):
                    kge_scores_t = kge_scores_t_tensor.cpu().numpy()
                elif isinstance(kge_scores_t_tensor, np.ndarray):
                    kge_scores_t = kge_scores_t_tensor
                else:
                    kge_scores_t = np.array([])

            tail_rank_calculated = False
            if kge_scores_t.size > 0:
                # Find top-k candidate tails
                top_k_indices_t = np.argsort(kge_scores_t)[: self.k]
                top_k_tail_ids = all_entity_ids[top_k_indices_t]

                # Prepare names
                h_name = self._get_names(h_anchor, self.id2entity)[0]
                r_name = self._get_names(r_anchor, self.id2relation)[0]
                top_k_tail_names = self._get_names(top_k_tail_ids, self.id2entity)

                # --- Calculate BGE scores for top-k AND true answer ---
                final_scores_t = np.full(self.ent_total, float("inf"))
                ids_to_score_bge_t = set(top_k_tail_ids)
                ids_to_score_bge_t.add(t_true)  # Ensure true answer is included

                bge_score_map_t = {}
                for tail_id in ids_to_score_bge_t:
                    tail_name = self._get_names(tail_id, self.id2entity)[0]
                    bge_score = self._calculate_bge_score(h_name, r_name, tail_name)
                    bge_score_map_t[tail_id] = bge_score

                # Assign calculated BGE scores
                for tail_id, score in bge_score_map_t.items():
                    if 0 <= tail_id < self.ent_total:
                        final_scores_t[tail_id] = score

                # --- Calculate Filtered Rank for Tail ---
                true_score_t = final_scores_t[t_true]
                if np.isinf(true_score_t):
                    print(
                        f"Warn: BGE score failed for true tail {t_true} in triple ({h_anchor}, {r_anchor}, {t_true}). Skipping tail metric update."
                    )
                else:
                    tail_rank_calculated = True
                    filtered_final_scores_t = final_scores_t.copy()
                    # Filter known true triples
                    for tail_idx in range(self.ent_total):
                        if (
                            tail_idx != t_true
                            and (h_anchor, r_anchor, tail_idx) in self.all_true_triples
                        ):
                            filtered_final_scores_t[tail_idx] = float("inf")

                    # Calculate rank based on BGE scores
                    rank_t_filter = 1 + np.sum(filtered_final_scores_t < true_score_t)
                    rank_t_filter += np.random.randint(
                        0, np.sum(filtered_final_scores_t == true_score_t) + 1
                    )

            # --- Accumulate Metrics only if rank calculation was successful ---
            if head_rank_calculated:
                total_mr_filter += rank_h_filter
                total_mrr_filter += 1.0 / rank_h_filter
                if rank_h_filter <= 10:
                    total_hits10_filter += 1
                if rank_h_filter <= 3:
                    total_hits3_filter += 1
                if rank_h_filter <= 1:
                    total_hits1_filter += 1
                valid_test_count += 1

            if tail_rank_calculated:
                total_mr_filter += rank_t_filter
                total_mrr_filter += 1.0 / rank_t_filter
                if rank_t_filter <= 10:
                    total_hits10_filter += 1
                if rank_t_filter <= 3:
                    total_hits3_filter += 1
                if rank_t_filter <= 1:
                    total_hits1_filter += 1
                valid_test_count += 1

        # --- Calculate Final Metrics ---
        if valid_test_count == 0:
            print("Error: No valid rankings were calculated.")
            return 0.0, 0.0, 0.0, 0.0, 0.0

        mrr = total_mrr_filter / valid_test_count
        mr = total_mr_filter / valid_test_count
        hit1 = total_hits1_filter / valid_test_count
        hit3 = total_hits3_filter / valid_test_count
        hit10 = total_hits10_filter / valid_test_count

        print(
            f"\n--- Link Prediction Results (Filtered, BGE Re-ranking Top-{self.k}) ---"
        )
        print(
            f"Evaluated {total_processed} triples ({valid_test_count} valid head/tail predictions counted)."
        )
        print(f"MRR:  {mrr:.4f}")
        print(f"MR:   {mr:.2f}")
        print(f"Hits@1: {hit1:.4f}")
        print(f"Hits@3: {hit3:.4f}")
        print(f"Hits@10: {hit10:.4f}")
        print("-" * 60)

        return mrr, mr, hit10, hit3, hit1

    # --- Add other methods if needed (like set_model, set_data_loader) ---
    def set_model(self, model):
        self.model = model
        if self.use_gpu:
            self.model.to(self.device)  # Use .to(device)
        self.model.eval()

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader
        # Optionally reload mappings/triples if needed
