import json
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse
import utils
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AutoModel
# datasets.disable_progress_bar()
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import os
from torch.nn.functional import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoModelForSeq2SeqLM# ,  AutoModelForSequenceClassification#, CrossEncoder
from sentence_transformers import CrossEncoder  # ? Correct
import numpy as np
from pcst_fast import pcst_fast
# from peft import AutoPeftModelForCausalLM
import torch
import re
from collections import defaultdict
from torch import nn
from sentence_transformers import SentenceTransformer, SparseEncoder
import pandas as pd
import time

N_CPUS = (
    int(os.environ["SLURM_CPUS_PER_TASK"]) if "SLURM_CPUS_PER_TASK" in os.environ else 1
)
PATH_RE = r"<PATH>(.*)<\/PATH>"
INSTRUCTION="""Please generate a valid relation path that can be helpful for answering the following question: """


class NPTripletRetriever:
    """
    Non-parametric baselines.
    kind:
      - "tfidf": TF-IDF + cosine similarity
      - "bm25": BM25 (implemented via rank_bm25 if available; otherwise raises with a clear msg)
    """
    def __init__(self, kind: str = "tfidf"):
        assert kind in {"tfidf", "bm25"}
        self.kind = kind

        # Optional dependency for BM25
        self._bm25_ok = False
        if kind == "bm25":
            try:
                from rank_bm25 import BM25Okapi  # pip install rank-bm25
                self.BM25Okapi = BM25Okapi
                self._bm25_ok = True
            except Exception:
                self.BM25Okapi = None
                self._bm25_ok = False

    def retrieve(self, question, triplet_list, top_k=50):
        if not triplet_list:
            return []

        triplet_texts = [f"{h} {r} {t}" for h, r, t in triplet_list]
        N = len(triplet_texts)
        k = min(top_k, N)
        if k <= 0:
            return []

        if self.kind == "tfidf":
            # TF-IDF retrieval (your original "BM25 (TF-IDF) retrieval" block, just packaged)
            tfidf = TfidfVectorizer().fit(triplet_texts)
            sparse_q = tfidf.transform([question])
            sparse_matrix = tfidf.transform(triplet_texts)

            # use sklearn-compatible cosine: dot product is fine because tfidf uses L2 norm by default
            # (keeps it minimal; no extra helper needed)
            scores = (sparse_matrix @ sparse_q.T).toarray().ravel()
            idx = np.argpartition(scores, -k)[-k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
            return [triplet_list[i] for i in idx]

        # BM25 retrieval
        if not self._bm25_ok:
            raise ImportError(
                "BM25 baseline requires `rank-bm25`. Install with: pip install rank-bm25"
            )

        # minimal tokenization (no extra helpers/classes): whitespace split lowercased
        tokenized_corpus = [t.lower().split() for t in triplet_texts]
        bm25 = self.BM25Okapi(tokenized_corpus)
        scores = np.array(bm25.get_scores(question.lower().split()), dtype=np.float32)

        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [triplet_list[i] for i in idx]



#class HybridTripletRetriever(nn.Module):
#    def __init__(self, encoder_name="distilbert/distilbert-base-uncased", cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2", device="cuda:2"):
#        super().__init__()
##        self.query_encoder = AutoModel.from_pretrained(encoder_name).to(device)
##        self.key_encoder = AutoModel.from_pretrained(encoder_name).to(device)
##        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
#        self.cross_encoder = CrossEncoder(cross_encoder_name, device=device) 
#        self.device = device
#
#        self.query_tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
#        self.key_tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-context-encoder")
#        self.query_encoder = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
#        self.key_encoder = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder")
#        
#
#        @torch.no_grad()
#    def encode_queries(self, texts):
#        if self.encoder in ["bge", "jina", "splade"]:
#            vec = self.query_encoder.encode(texts)
#            vec = torch.tensor(vec, dtype=torch.float32, device=self._dev())
#            return vec
#        else:
#            return self._encode(self.query_encoder, self.query_tokenizer, texts)                    
#
#    @torch.no_grad()
#    def encode_keys(self, texts, batch_size=2048, max_len=128):
#        self.eval()
#        outs = []
#
#        for i in range(0, len(texts), batch_size):
#            batch = texts[i:i+batch_size]
#            toks = self.key_tokenizer(
#                batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
#            ).to(self._dev())
#            out = self.key_encoder(**toks)
#            rep = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
#            rep = rep[:, 0, :] if self.use_cls else _mean_pool(rep, toks["attention_mask"])
#            outs.append(rep)
#    
#        # ? Prevent torch.cat error
#        if not outs:
#            return torch.empty(0)  # or return None, depending on your use case
#    
#        return torch.cat(outs, dim=0)
        
#    def encode_queries(self, texts):
#        tokens = self.query_tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
#        return self.query_encoder(**tokens).last_hidden_state[:, 0]  # CLS
#
#    def encode_keys(self, texts, batch_size=512):
#        all_outputs = []
#        for i in range(0, len(texts), batch_size):
#            batch = texts[i:i + batch_size]
#            tokens = self.key_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
#            with torch.no_grad():
#                out = self.key_encoder(**tokens).last_hidden_state[:, 0]
#            all_outputs.append(out)
#        return torch.cat(all_outputs, dim=0)
#
#    @torch.no_grad()
#    def retrieve(self, question, triplet_list, top_k=50):
#        self.eval()
#        triplet_texts = [f"{h} {r} {t}" for h, r, t in triplet_list]
#
#        # BM25 (TF-IDF) retrieval
#        tfidf = TfidfVectorizer().fit(triplet_texts)
#        sparse_q = tfidf.transform([question])
#        sparse_matrix = tfidf.transform(triplet_texts)
#        sparse_scores = cos_sim(sparse_q, sparse_matrix)[0]
#        bm25_top = np.argsort(sparse_scores)[-top_k:][::-1]
#
#        # Dense (TwoTower) retrieval
#        dense_q = self.encode_queries([question])  # (1, D)
#        dense_keys = self.encode_keys(triplet_texts)  # (N, D)
#        dense_scores = torch.nn.functional.cosine_similarity(dense_q, dense_keys).cpu().numpy()
#        dpr_top = np.argsort(dense_scores)[-top_k:][::-1]
#
#        # Merge candidates
#        candidate_indices = list(set(bm25_top) | set(dpr_top))
#        candidate_texts = [triplet_texts[i] for i in candidate_indices]
#        candidate_triplets = [triplet_list[i] for i in candidate_indices]
#
#        # Cross-encoder reranking
#        cross_inputs = [(question, ctx) for ctx in candidate_texts]
#        rerank_scores = self.cross_encoder.predict(cross_inputs)
#        top_indices = np.argsort(rerank_scores)[-top_k:][::-1]
#
#        top_triplets = [candidate_triplets[i] for i in top_indices]
#        return top_triplets



def get_output_file(path, force=False):
    if not os.path.exists(path) or force:
        fout = open(path, "w")
        return fout, []
    else:
        with open(path, "r") as f:
            processed_results = []
            for line in f:
                try:
                    results = json.loads(line)
                except:
                    raise ValueError("Error in line: ", line)
                processed_results.append(results["id"])
        fout = open(path, "a")
        return fout, processed_results


def parse_prediction(prediction):
    """
    Parse a list of predictions to a list of rules

    Args:
        prediction (_type_): _description_

    Returns:
        _type_: _description_
    """
    results = []
    for p in prediction:
        path = re.search(PATH_RE, p)
        if path is None:
            continue
        path = path.group(1)
        path = path.split("<SEP>")
        if len(path) == 0:
            continue
        rules = []
        for rel in path:
            rel = rel.strip()
            if rel == "":
                continue
            rules.append(rel)
        results.append(rules)
    return results


def generate_seq(
    model, input_text, tokenizer, num_beam=3, do_sample=False, max_new_tokens=100
):
    # tokenize the question
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda:1")
    # generate sequences
    output = model.generate(
        input_ids=input_ids,
        num_beams=num_beam,
        num_return_sequences=num_beam,
        early_stopping=False,
        do_sample=do_sample,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=max_new_tokens,
    )
    prediction = tokenizer.batch_decode(
        output.sequences[:, input_ids.shape[1] :], skip_special_tokens=True
    )
    prediction = [p.strip() for p in prediction]

    if num_beam > 1:
        scores = output.sequences_scores.tolist()
        norm_scores = torch.softmax(output.sequences_scores, dim=0).tolist()
    else:
        scores = [1]
        norm_scores = [1]

    return {"paths": prediction, "scores": scores, "norm_scores": norm_scores}


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
    count  = mask.sum(dim=1).clamp(min=1e-9)                        # (B,1)
    return summed / count


class TwoTowerModel(torch.nn.Module):
    def __init__(self, encoder="distilbert", device = "cuda:0", use_cls=True):
        super().__init__()
        




        self.encoder = encoder
        self.device = device
        self.use_cls= use_cls
        if encoder == "distilbert":
            encoder_name = "distilbert/distilbert-base-uncased"
            self.query_encoder = AutoModel.from_pretrained(encoder_name)
            self.key_encoder = AutoModel.from_pretrained(encoder_name)
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.query_tokenizer = self.tokenizer
            self.key_tokenizer = self.tokenizer
        elif encoder == "mxbai":
            name = "mixedbread-ai/mxbai-rerank-base-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            self.query_encoder = AutoModel.from_pretrained(name)   # <-- not CausalLM
            self.key_encoder   = AutoModel.from_pretrained(name)
            self.query_tokenizer = self.tokenizer
            self.key_tokenizer = self.tokenizer
        elif encoder == "bge":
            encoder_name = "BAAI/bge-m3"
            self.query_encoder = SentenceTransformer(encoder_name, device = self.device)
            self.key_encoder = SentenceTransformer(encoder_name, device = self.device)
        elif encoder == 'jina':
            encoder_name = "jinaai/jina-embedding-b-en-v1"
            self.query_encoder = SentenceTransformer(encoder_name, trust_remote_code=True)
            self.key_encoder = SentenceTransformer(encoder_name, trust_remote_code=True)
            
        elif encoder == "splade":
            encoder_name = "naver/splade-cocondenser-ensembledistil"
            self.query_encoder = SparseEncoder("naver/efficient-splade-V-large-query")
            self.key_encoder = SparseEncoder("naver/efficient-splade-V-large-doc")
        elif encoder == 'bce':
            # encoder_name = "maidalun1020/bce-reranker-base_v1"
            encoder_name = "maidalun1020/bce-embedding-base_v1"
            self.query_encoder = AutoModel.from_pretrained(encoder_name)
            self.key_encoder = AutoModel.from_pretrained(encoder_name)
            
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
            self.query_tokenizer = self.tokenizer
            self.key_tokenizer = self.tokenizer
        elif encoder == 'twolar':
            self.tokenizer = AutoTokenizer.from_pretrained("Dundalia/TWOLAR-large")
            self.query_encoder = AutoModelForSeq2SeqLM.from_pretrained("Dundalia/TWOLAR-large")
            self.key_encoder = AutoModelForSeq2SeqLM.from_pretrained("Dundalia/TWOLAR-large")
            self.query_tokenizer = self.tokenizer
            self.key_tokenizer = self.tokenizer
        elif encoder == 'colbert':
            self.tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
            self.query_encoder = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
            self.key_encoder = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
            self.query_tokenizer = self.tokenizer
            self.key_tokenizer = self.tokenizer
        elif encoder in ["dragon", "grag", "gretriever", 'hybrid', 'skp']:
            self.query_tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
            self.key_tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-context-encoder")
            self.query_encoder = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder")
            self.key_encoder = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder")
            if encoder == 'hybrid':
                self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", device= self.device)
        elif encoder == "kamr":
            qname = "facebook/dragon-plus-query-encoder"
            kname = "facebook/dragon-plus-context-encoder"
            self.query_tokenizer = AutoTokenizer.from_pretrained(qname)
            self.key_tokenizer   = AutoTokenizer.from_pretrained(kname)
            self.entity_tokenizer= AutoTokenizer.from_pretrained(kname)
            self.rel_tokenizer   = AutoTokenizer.from_pretrained(kname)
            self.query_encoder   = AutoModel.from_pretrained(qname)
            self.key_encoder     = AutoModel.from_pretrained(kname)
            self.entity_encoder  = AutoModel.from_pretrained(kname)
            # self.rel_encoder     = AutoModel.from_pretrained(kname)
#        hdim = getattr(self.query_encoder.config, "hidden_size", None)
#        if hdim is None and hasattr(self.query_encoder.config, "d_model"):
#            hdim = self.query_encoder.config.d_model
#        assert hdim is not None, "Cannot determine hidden size."

        # Multi-hop classifier head on query representation
#         self.cls_head = nn.Linear(hdim, 1)

        # Temperature for similarity logits
        self.logit_scale = nn.Parameter(torch.tensor(1.0))  # you can init with ln(1/tau) if you prefer
    def _dev(self):
        return next(self.parameters()).device
            
    def _encode(self, encoder, tokenizer, texts, max_len=128):
        tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_len).to(self._dev())
        out = encoder(**tokens)
        # Handle Seq2Seq encoders (twolar) vs encoder-only
        if hasattr(out, "last_hidden_state"):
            rep = out.last_hidden_state
        else:
            # Some models might return tuple; assume first is hidden states
            rep = out[0]
        return rep[:, 0, :] if self.use_cls else _mean_pool(rep, tokens["attention_mask"])









    @torch.no_grad()
    def encode_queries(self, texts):
        if self.encoder in ["bge", "jina", "splade"]:
            vec = self.query_encoder.encode(texts)
            vec = torch.tensor(vec, dtype=torch.float32, device=self._dev())
            return vec
        else:
            return self._encode(self.query_encoder, self.query_tokenizer, texts)                    

    @torch.no_grad()
    def encode_keys(self, texts, batch_size=2048, max_len=128):
        self.eval()
        outs = []
        if self.encoder in ["bge", "jina", "splade"]:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                vec = self.key_encoder.encode(batch)
                vec = torch.tensor(vec, dtype=torch.float32, device=self._dev())
                outs.append(vec)
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                toks = self.key_tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len
                ).to(self._dev())
                out = self.key_encoder(**toks)
                rep = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
                rep = rep[:, 0, :] if self.use_cls else _mean_pool(rep, toks["attention_mask"])
                outs.append(rep)
    
        # ? Prevent torch.cat error
        if not outs:
            return torch.empty(0)  # or return None, depending on your use case
    
        return torch.cat(outs, dim=0)
        #return self._encode(self.key_encoder,   self.key_tokenizer,   texts)

    @torch.no_grad()
    def encode_entities(self, texts):
        return self._encode(self.entity_encoder, self.entity_tokenizer, texts)
    def encode_relations(self, texts):
        return self._encode(self.entity_encoder,    self.entity_tokenizer,    texts)
        
#    def encode_queries(self, texts):
#        if self.encoder in ["bge", "jina", "splade"]:
#            return torch.tensor(self.query_encoder.encode(texts), dtype=torch.float32).to(self.device)
#        tokens = self.query_tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
#        with torch.no_grad():
#            out = self.query_encoder(**tokens)
#            # return _mean_pool(out.last_hidden_state, tokens["attention_mask"])
#            return out.last_hidden_state[:, 0, :]
#    def encode_keys(self, texts, batch_size=2048):
#        outs = []
#        for i in range(0, len(texts), batch_size):
#            batch = texts[i:i+batch_size]
#            if self.encoder in ["bge", "jina", "splade"]:
#                outs.append(torch.tensor(self.key_encoder.encode(batch), dtype=torch.float32).to(self.device))
#            else:
#                tokens = self.key_tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
#                with torch.no_grad():
#                    out = self.key_encoder(**tokens)
#                    outs.append(out.last_hidden_state[:, 0, :])
#                    # outs.append(_mean_pool(out.last_hidden_state, tokens["attention_mask"]))
#                    # return out.last_hidden_state[:, 0, :]
#        return torch.cat(outs, dim=0)
    def _build_entity2triplets(self, triplet_list):
        entity2idx = defaultdict(list)
        for i, (h, r, t) in enumerate(triplet_list):
            entity2idx[h].append(i)
            entity2idx[t].append(i)
        return entity2idx
    def pair_logits(self, a_emb, k_emb):
        # Feature fusion: [a, k, |a-k|, a*k]
        feats = torch.cat([a_emb, k_emb, torch.abs(a_emb - k_emb), a_emb * k_emb], dim=-1)
        return self.classifier(feats).squeeze(-1)  # (batch,)

    def pair_scores(self, a_emb, k_emb):
        return torch.sigmoid(self.pair_logits(a_emb, k_emb))
        
        
        

    def retrieve_expand_1hop_component_keys(
        self,
        question,
        triplet_list,
        top_k=50,
        seed_top=10,
        *,
        entity2idx=None,          # <- prebuilt entity -> [trip idx]
        concept_embeds=None,      # <- dict[str]->Tensor(D,) optional
        concept_scores=None,      # <- dict[str]->float optional
        precomputed_keys=None,    # <- Tensor(N,D) for triplet texts (seed stage) optional
        w_rel=1.0,
        w_ent=1.0,
    ):
        N = len(triplet_list)
        if N == 0:
            return []

        # --- Encode query once ---
        q = self.encode_queries([question])  # (1, D)
        if q.is_sparse: q = q.to_dense()

        # --- Seed selection by triplet text (unchanged) ---
        triplet_texts = [f"{h} | {r} | {t}" for h, r, t in triplet_list]
        if precomputed_keys is None:
            K_trip = self.encode_keys(triplet_texts)  # (N, D)
        else:
            K_trip = precomputed_keys
        if K_trip.is_sparse: K_trip = K_trip.to_dense()

        seed_scores = torch.nn.functional.cosine_similarity(q, K_trip).cpu().numpy()
        seed_top = min(seed_top, top_k, N)
        seed_indices = np.argsort(seed_scores)[-seed_top:][::-1].tolist()

        # --- Ensure entity2idx available ---
        if entity2idx is None:
            entity2idx = build_entity_index(triplet_list)

        # --- Precompute concept embeddings & scores if needed ---
        if concept_scores is None:
            concepts = unique_concepts(triplet_list)
            if concept_embeds is None:
                E = self.encode_keys(concepts)
                if E.is_sparse: E = E.to_dense()
                concept_embeds = {c: E[i] for i, c in enumerate(concepts)}
            concept_scores = {}
            for c, e in concept_embeds.items():
                concept_scores[c] = torch.nn.functional.cosine_similarity(
                    q, e.unsqueeze(0)
                ).item()

        def comp_score(idx: int) -> float:
            h, r, t = triplet_list[idx]
            h, r, t = h.strip(), r.strip(), t.strip()
            return max(
                w_rel * concept_scores.get(r, 0.0),
                w_ent * concept_scores.get(h, 0.0),
                w_ent * concept_scores.get(t, 0.0),
            )

        # --- Select seeds first ---
        selected, selected_set = [], set()
        for si in seed_indices:
            selected.append(si); selected_set.add(si)

        # --- Neighbor budget ---
        remaining = max(0, top_k - len(seed_indices))
        if seed_indices and remaining > 0:
            base = remaining // len(seed_indices)
            extra = remaining % len(seed_indices)
        else:
            base = 0; extra = 0

        # --- Expand per seed using entity2idx (indices only, no tuple rebuilds) ---
        for rank, si in enumerate(seed_indices):
            h, r, t = triplet_list[si]
            h, r, t = h.strip(), r.strip(), t.strip()

            cand_idx_set = set(entity2idx.get(h, [])) | set(entity2idx.get(t, []))
            cand_idx_set.discard(si)
            cand_idx = [i for i in cand_idx_set if i not in selected_set]
            if not cand_idx:
                continue

            cand_sc = [(i, comp_score(i)) for i in cand_idx]
            cand_sc.sort(key=lambda x: x[1], reverse=True)

            take = base + (1 if rank < extra else 0)
            for i, _ in cand_sc[:take]:
                selected.append(i); selected_set.add(i)

            if len(selected) >= top_k:
                break

        # --- Backfill by component-key score if still short ---
        if len(selected) < top_k:
            rest = [(i, comp_score(i)) for i in range(N) if i not in selected_set]
            rest.sort(key=lambda x: x[1], reverse=True)
            need = top_k - len(selected)
            selected.extend([i for i, _ in rest[:need]])

        selected = selected[:top_k]
        lst = [triplet_list[i] for i in selected]  # length 50
        half = len(lst) // 2
        
        # Interleave [0..24] and [25..49]
        interleaved = [x for pair in zip(lst[:half], lst[half:]) for x in pair]
        
        # If odd length (just in case), append leftover
        if len(lst) % 2:
            interleaved.append(lst[-1])
        
        return interleaved
        
        
        # return [triplet_list[i] for i in selected]


    @torch.no_grad()
    def hybrid_retrieve(self, question, triplet_list, top_k=50):
        self.eval()
        triplet_texts = [f"{h} {r} {t}" for h, r, t in triplet_list]

        # BM25 (TF-IDF) retrieval
        tfidf = TfidfVectorizer().fit(triplet_texts)
        sparse_q = tfidf.transform([question])
        sparse_matrix = tfidf.transform(triplet_texts)
        sparse_scores = cos_sim(sparse_q, sparse_matrix)[0]
        bm25_top = np.argsort(sparse_scores)[-top_k:][::-1]

        # Dense (TwoTower) retrieval
        dense_q = self.encode_queries([question])  # (1, D)
        dense_keys = self.encode_keys(triplet_texts)  # (N, D)
        dense_scores = torch.nn.functional.cosine_similarity(dense_q, dense_keys).cpu().numpy()
        dpr_top = np.argsort(dense_scores)[-top_k:][::-1]

        # Merge candidates
        candidate_indices = list(set(bm25_top) | set(dpr_top))
        candidate_texts = [triplet_texts[i] for i in candidate_indices]
        candidate_triplets = [triplet_list[i] for i in candidate_indices]

        # Cross-encoder reranking
        cross_inputs = [(question, ctx) for ctx in candidate_texts]
        rerank_scores = self.cross_encoder.predict(cross_inputs)
        top_indices = np.argsort(rerank_scores)[-top_k:][::-1]

        top_triplets = [candidate_triplets[i] for i in top_indices]
        return top_triplets



    def retrieve_2hop_component_keys_exclude_joint_seed_backfill(
    self,
    question,
    triplet_list,
    top_k=50,
    seed_top=10,
    *,
    hop1_k=None,               # budget for 1-hop neighbors (after seeds). default: half of remaining
    hop2_k=None,               # budget for 2-hop neighbors (after seeds+hop1). default: rest
    entity2idx=None,           # entity -> [trip idx]
    concept_embeds=None,       # dict[str] -> Tensor(D,), optional
    concept_scores=None,       # dict[str] -> float, optional (cos(q, concept))
    precomputed_keys=None,     # Optional: dict with 'hr','rt','ht' -> Tensor(N,D); else None
    w_rel=1.0,
    w_ent=1.0,
):

        import numpy as np
        import torch
        import torch.nn.functional as F
    
        N = len(triplet_list)
        if N == 0:
            return []
    
        # -----------------------
        # Encode query once
        # -----------------------
        q = self.encode_queries([question])  # (1, D)
        if hasattr(q, "is_sparse") and q.is_sparse:
            q = q.to_dense()
    
        # -----------------------
        # Seed selection by pair-combinations
        # -----------------------
        hr_texts = [f"{h} | {r}" for (h, r, t) in triplet_list]
        rt_texts = [f"{r} | {t}" for (h, r, t) in triplet_list]
        ht_texts = [f"{h} | {t}" for (h, r, t) in triplet_list]
    
        if isinstance(precomputed_keys, dict) and all(k in precomputed_keys for k in ("hr", "rt", "ht")):
            K_hr, K_rt, K_ht = precomputed_keys["hr"], precomputed_keys["rt"], precomputed_keys["ht"]
        else:
            K_hr = self.encode_keys(hr_texts)  # (N, D)
            K_rt = self.encode_keys(rt_texts)  # (N, D)
            K_ht = self.encode_keys(ht_texts)  # (N, D)
    
        # Densify if needed (make sure we actually overwrite variables)
        if hasattr(K_hr, "is_sparse") and K_hr.is_sparse:
            K_hr = K_hr.to_dense()
        if hasattr(K_rt, "is_sparse") and K_rt.is_sparse:
            K_rt = K_rt.to_dense()
        if hasattr(K_ht, "is_sparse") and K_ht.is_sparse:
            K_ht = K_ht.to_dense()
    
        s_hr = F.cosine_similarity(q, K_hr).detach().cpu().numpy()  # (N,)
        s_rt = F.cosine_similarity(q, K_rt).detach().cpu().numpy()  # (N,)
        s_ht = F.cosine_similarity(q, K_ht).detach().cpu().numpy()  # (N,)
        seed_scores = np.maximum(np.maximum(s_hr, s_rt), s_ht)      # (N,)
    
        ranked_all = np.argsort(seed_scores)[::-1].tolist()
        seed_top = min(seed_top, top_k, N)
        seed_indices = ranked_all[:seed_top]
    
        # -----------------------
        # entity2idx map
        # -----------------------
        if entity2idx is None:
            entity2idx = {}
            for i, (h, _, t) in enumerate(triplet_list):
                h_ = h.strip()
                t_ = t.strip()
                entity2idx.setdefault(h_, []).append(i)
                entity2idx.setdefault(t_, []).append(i)
    
        # -----------------------
        # concept_scores (entity/rel towers)
        # -----------------------
        if concept_scores is None:
            ent_list, rel_list = [], []
            seen_ent, seen_rel = set(), set()
            for (h, r, t) in triplet_list:
                h = h.strip()
                t = t.strip()
                r = r.strip().split(".")[-1]
                if h not in seen_ent:
                    seen_ent.add(h)
                    ent_list.append(h)
                if t not in seen_ent:
                    seen_ent.add(t)
                    ent_list.append(t)
                if r not in seen_rel:
                    seen_rel.add(r)
                    rel_list.append(r)
    
            if concept_embeds is None:
                concept_embeds = {}
    
                if ent_list:
                    E_ent = self.encode_entities(ent_list)  # (|E|, D)
                    if hasattr(E_ent, "is_sparse") and E_ent.is_sparse:
                        E_ent = E_ent.to_dense()
                    for i, e_name in enumerate(ent_list):
                        concept_embeds[e_name] = E_ent[i]
    
                if rel_list:
                    E_rel = self.encode_relations(rel_list)  # (|R|, D)
                    if hasattr(E_rel, "is_sparse") and E_rel.is_sparse:
                        E_rel = E_rel.to_dense()
                    for i, r_name in enumerate(rel_list):
                        concept_embeds[r_name] = E_rel[i]
    
            concept_scores = {}
            for c_name, c_vec in concept_embeds.items():
                concept_scores[c_name] = F.cosine_similarity(q, c_vec.unsqueeze(0)).item()
    
        # -----------------------
        # Neighbor scoring with joint-entity exclusion
        # -----------------------
        def comp_score_excl_joint(idx: int, joint_entities: set) -> float:
            h, r, t = triplet_list[idx]
            h = h.strip()
            t = t.strip()
            r = r.strip().split(".")[-1]
    
            candidates = [w_rel * concept_scores.get(r, 0.0)]
            if h not in joint_entities:
                candidates.append(w_ent * concept_scores.get(h, 0.0))
            if t not in joint_entities:
                candidates.append(w_ent * concept_scores.get(t, 0.0))
            return max(candidates) if candidates else 0.0
    
        # -----------------------
        # Budget split
        # -----------------------
        remaining_total = max(0, top_k - len(seed_indices))
        if hop1_k is None and hop2_k is None:
            hop1_k = remaining_total // 2
            hop2_k = remaining_total - hop1_k
        elif hop1_k is None:
            hop2_k = min(hop2_k, remaining_total)
            hop1_k = remaining_total - hop2_k
        elif hop2_k is None:
            hop1_k = min(hop1_k, remaining_total)
            hop2_k = remaining_total - hop1_k
        else:
            hop1_k = min(hop1_k, remaining_total)
            hop2_k = min(hop2_k, remaining_total - hop1_k)
    
        # -----------------------
        # Select seeds first
        # -----------------------
        selected_set = set()
        selected_triplets = []
        selected_indices = []
    
        for si in seed_indices:
            if si in selected_set:
                continue
            selected_set.add(si)
            selected_indices.append(si)
            selected_triplets.append(triplet_list[si])
            if len(selected_triplets) >= top_k:
                return selected_triplets
    
        # -----------------------
        # 1-hop expansion from seeds
        # -----------------------
        hop1_added = []
        hop1_budget = min(hop1_k, top_k - len(selected_triplets))
    
        if seed_indices and hop1_budget > 0:
            base1 = hop1_budget // len(seed_indices)
            extra1 = hop1_budget % len(seed_indices)
    
            for rank, si in enumerate(seed_indices):
                if len(selected_triplets) >= top_k:
                    break
    
                sh, _, st = triplet_list[si]
                sh = sh.strip()
                st = st.strip()
                joint_pair = {sh, st}
    
                cand_idx_set = set(entity2idx.get(sh, [])) | set(entity2idx.get(st, []))
                cand_idx_set.discard(si)
    
                cand_idx = [i for i in cand_idx_set if i not in selected_set]
                if not cand_idx:
                    continue
    
                scored = []
                for i in cand_idx:
                    h2, _, t2 = triplet_list[i]
                    h2 = h2.strip()
                    t2 = t2.strip()
                    joint_entities = joint_pair.intersection({h2, t2})
                    nb_sc = comp_score_excl_joint(i, joint_entities)
                    scored.append((i, nb_sc))
    
                scored.sort(key=lambda x: x[1], reverse=True)
                take = base1 + (1 if rank < extra1 else 0)
                if take <= 0:
                    continue
    
                for i, _nb_sc in scored[:take]:
                    if i in selected_set:
                        continue
                    selected_set.add(i)
                    selected_indices.append(i)
                    selected_triplets.append(triplet_list[i])
                    hop1_added.append(i)
                    if len(selected_triplets) >= top_k:
                        return selected_triplets
    
        # -----------------------
        # 2-hop expansion: use hop1-added triplets as new anchors
        # -----------------------
        hop2_budget = min(hop2_k, top_k - len(selected_triplets))
        if hop1_added and hop2_budget > 0:
            base2 = hop2_budget // len(hop1_added)
            extra2 = hop2_budget % len(hop1_added)
    
            for rank, ai in enumerate(hop1_added):
                if len(selected_triplets) >= top_k:
                    break
    
                ah, _, at = triplet_list[ai]
                ah = ah.strip()
                at = at.strip()
                joint_pair = {ah, at}
    
                cand_idx_set = set(entity2idx.get(ah, [])) | set(entity2idx.get(at, []))
                cand_idx_set.discard(ai)
    
                # IMPORTANT: do not re-score triplets already retrieved in previous rounds
                cand_idx = [i for i in cand_idx_set if i not in selected_set]
                if not cand_idx:
                    continue
    
                scored = []
                for i in cand_idx:
                    h2, _, t2 = triplet_list[i]
                    h2 = h2.strip()
                    t2 = t2.strip()
                    joint_entities = joint_pair.intersection({h2, t2})
                    nb_sc = comp_score_excl_joint(i, joint_entities)
                    scored.append((i, nb_sc))
    
                scored.sort(key=lambda x: x[1], reverse=True)
                take = base2 + (1 if rank < extra2 else 0)
                if take <= 0:
                    continue
    
                for i, _nb_sc in scored[:take]:
                    if i in selected_set:
                        continue
                    selected_set.add(i)
                    selected_indices.append(i)
                    selected_triplets.append(triplet_list[i])
                    if len(selected_triplets) >= top_k:
                        return selected_triplets
    
        # -----------------------
        # Backfill from remaining globally-ranked seeds
        # -----------------------
        if len(selected_triplets) < top_k:
            need = top_k - len(selected_triplets)
            for idx in ranked_all[seed_top:]:
                if idx not in selected_set:
                    selected_set.add(idx)
                    selected_indices.append(idx)
                    selected_triplets.append(triplet_list[idx])
                    need -= 1
                    if need == 0:
                        break
    
        return selected_triplets


    def retrieve_expand_1hop_component_keys_exclude_joint_seed_backfill(
        self,
        question,
        triplet_list,
        top_k=50,
        seed_top=10,
        *,
        entity2idx=None,          # entity -> [trip idx]
        concept_embeds=None,      # dict[str] -> Tensor(D,), optional
        concept_scores=None,      # dict[str] -> float, optional (cos(q, concept))
        precomputed_keys=None,    # Optional: dict with 'hr','rt','ht' -> Tensor(N,D); else None
        w_rel=1.0,
        w_ent=1.0,
    ):
        """
        Seeds scored by max over pair-combinations: max( cos(q, [h|r]), cos(q, [r|t]), cos(q, [h|t]) ).
        Expansion uses joint-entity exclusion. Backfill uses remaining seeds only.
        Returns:
          selected_with_scores: list of (triplet, score, seed_score_ref)
            - Seeds/backfilled seeds: (triplet, seed_pair_score, None)
            - Neighbors: (triplet, neighbor_score_excl_joint, seed_pair_score_of_its_seed)
        """
        import numpy as np
        import torch
        import torch.nn.functional as F
    
        N = len(triplet_list)
        if N == 0:
            return []
    
        # --- Encode query once ---
        q = self.encode_queries([question])  # (1, D)
        if hasattr(q, "is_sparse") and q.is_sparse:
            q = q.to_dense()
    
 #        full_text = [f"{h} | {r} | {t}" for (h, r, t) in triplet_list]
#        # --- Seed selection by pair-combinations: [h|r], [r|t], [h|t] ---
        hr_texts = [f"{h} | {r}" for (h, r, t) in triplet_list]
        rt_texts = [f"{r} | {t}" for (h, r, t) in triplet_list]
        ht_texts = [f"{h} | {t}" for (h, r, t) in triplet_list]
    
        # Allow precomputed pair embeddings via dict
        if isinstance(precomputed_keys, dict) and all(k in precomputed_keys for k in ("hr", "rt", "ht")):
            K_hr, K_rt, K_ht = precomputed_keys["hr"], precomputed_keys["rt"], precomputed_keys["ht"]
        else:
            K_hr = self.encode_keys(hr_texts)  # (N, D)
            K_rt = self.encode_keys(rt_texts)  # (N, D)
            K_ht = self.encode_keys(ht_texts)  # (N, D)
    
        # Densify if needed
        for name, K in (("hr", K_hr), ("rt", K_rt), ("ht", K_ht)):
            if hasattr(K, "is_sparse") and K.is_sparse:
                locals()[f"K_{name}"] = K.to_dense()
    
        # Cosines for each pair-list
        s_hr = F.cosine_similarity(q, K_hr).cpu().numpy()  # (N,)
        s_rt = F.cosine_similarity(q, K_rt).cpu().numpy()  # (N,)
        s_ht = F.cosine_similarity(q, K_ht).cpu().numpy()  # (N,)
    
        # Seed score per triplet = max over the three pair scores
        seed_scores = np.maximum(np.maximum(s_hr, s_rt), s_ht)  # (N,)
#        
#        K_full = self.encode_keys(full_text)
#        seed_scores = F.cosine_similarity(q, K_full).cpu().numpy()
        
        ranked_all = np.argsort(seed_scores)[::-1].tolist()
        seed_top = min(seed_top, top_k, N)
        seed_indices = ranked_all[:seed_top]
    
        # --- Ensure entity2idx available ---
        if entity2idx is None:
            entity2idx = {}
            for i, (h, _, t) in enumerate(triplet_list):
                h_ = h.strip(); t_ = t.strip()
                entity2idx.setdefault(h_, []).append(i)
                entity2idx.setdefault(t_, []).append(i)
    
        # --- Precompute concept embeddings & scores (entity/rel towers) ---
        if concept_scores is None:
            # Collect uniques separately for entities and relations
            ent_list, rel_list = [], []
            seen_ent, seen_rel = set(), set()
            for (h, r, t) in triplet_list:
                h, r, t = h.strip(), r.strip().split('.')[-1], t.strip()
                if h not in seen_ent:
                    seen_ent.add(h); ent_list.append(h)
                if t not in seen_ent:
                    seen_ent.add(t); ent_list.append(t)
                if r not in seen_rel:
                    seen_rel.add(r); rel_list.append(r)
    
            # Build concept_embeds if not provided
            if concept_embeds is None:
                concept_embeds = {}
    
                # Encode entities with the entity tower
                if len(ent_list) > 0:
                    E_ent = self.encode_entities(ent_list)  # (|E|, D)
                    if hasattr(E_ent, "is_sparse") and E_ent.is_sparse:
                        E_ent = E_ent.to_dense()
                    for i, e_name in enumerate(ent_list):
                        concept_embeds[e_name] = E_ent[i]
    
                # Encode relations with the relation tower
                if len(rel_list) > 0:
                    E_rel = self.encode_relations(rel_list)  # (|R|, D)
                    if hasattr(E_rel, "is_sparse") and E_rel.is_sparse:
                        E_rel = E_rel.to_dense()
                    for i, r_name in enumerate(rel_list):
                        concept_embeds[r_name] = E_rel[i]
    
            # Compute cosine scores to the query for all concepts
            concept_scores = {}
            for c_name, c_vec in concept_embeds.items():
                concept_scores[c_name] = F.cosine_similarity(q, c_vec.unsqueeze(0)).item()
    
        # --- Neighbor scoring with joint-entity exclusion ---
        def comp_score_excl_joint(idx: int, joint_entities: set) -> float:
            h, r, t = triplet_list[idx]
            h, r, t = h.strip(), r.strip().split('.')[-1], t.strip()
            # score = max( w_rel*s(q,r), w_ent*s(q, non-joint entity) )
            candidates = [w_rel * concept_scores.get(r, 0.0)]
            if h not in joint_entities:
                candidates.append(w_ent * concept_scores.get(h, 0.0))
            if t not in joint_entities:
                candidates.append(w_ent * concept_scores.get(t, 0.0))
            return max(candidates) if candidates else 0.0
    
        # --- Select seeds first (record scores) ---
        selected, selected_set = [], set()
        selected_with_scores = []  # (triplet, score, seed_score_ref)
    
        for si in seed_indices:
            selected.append(si); selected_set.add(si)
            seed_sc = float(seed_scores[si])
            selected_with_scores.append(triplet_list[si])
            # selected_with_scores.append((triplet_list[si], seed_sc, None))
    
        # --- Neighbor budget ---
        remaining = max(0, top_k - len(seed_indices))
        if seed_indices and remaining > 0:
            base = remaining // len(seed_indices)
            extra = remaining % len(seed_indices)
        else:
            base = 0; extra = 0
    
        # --- Expand per seed using joint-entity exclusion (record both scores) ---
        for rank, si in enumerate(seed_indices):
            sh, sr, st = triplet_list[si]
            sh, st = sh.strip(), st.strip()
            seed_sc = float(seed_scores[si])  
    
            cand_idx_set = set(entity2idx.get(sh, [])) | set(entity2idx.get(st, []))
            cand_idx_set.discard(si)
            cand_idx = [i for i in cand_idx_set if i not in selected_set]
            if not cand_idx:
                continue
    
            joint_pair = {sh, st}
            scored = []

                
              

            
            
            for i in cand_idx:              
                h2, _, t2 = triplet_list[i]
                h2, t2 = h2.strip(), t2.strip()
                joint_entities = joint_pair.intersection({h2, t2})
                nb_sc = comp_score_excl_joint(i, joint_entities) 
                scored.append((i, nb_sc))
    
            scored.sort(key=lambda x: x[1], reverse=True)
            take = base + (1 if rank < extra else 0)
            for i, nb_sc in scored[:take]:
                selected.append(i); selected_set.add(i)
                # selected_with_scores.append((triplet_list[i], float(nb_sc), seed_sc))
                selected_with_scores.append(triplet_list[i])
    
            if len(selected) >= top_k:
                break
    
        # --- Backfill strictly from the remaining globally-ranked seeds (record scores) ---
        if len(selected) < top_k:
            need = top_k - len(selected)
            for idx in ranked_all[seed_top:]:
                if idx not in selected_set:
                    selected.append(idx); selected_set.add(idx)
                    seed_sc = float(seed_scores[idx])
                    # selected_with_scores.append((triplet_list[idx], seed_sc, None))
                    selected_with_scores.append(triplet_list[idx])
                    need -= 1
                    if need == 0:
                        break
    
        # Keep only top_k items (already in ranked order of selection)
        selected_with_scores = selected_with_scores
        return selected_with_scores



    def retrieve(self, question, triplet_list, top_k=50):
        # 0) Early exit if no candidates
        if not triplet_list:
            return []
    
        triplet_texts = [f"{h} | {r} | {t}" for h, r, t in triplet_list]
    
        # 1) Encode query
        q = self.encode_queries([question])  # shape (1, D) or (D,)
        if getattr(q, "is_sparse", False):
            q = q.to_dense()
        if q.dim() == 2 and q.size(0) == 1:
            q = q.squeeze(0)  # (D,)
        elif q.dim() == 1:
            pass  # already (D,)
        else:
            raise ValueError(f"Unexpected query embedding shape: {tuple(q.shape)}")
    
        # 2) Encode keys
        K = self.encode_keys(triplet_texts)  # should be (N, D)
        if getattr(K, "is_sparse", False):
            K = K.to_dense()
    
        # Handle empty/edge shapes from encoder
        if K.numel() == 0:
            return []  # no keys ? nothing to rank
        if K.dim() == 1:
            # Happens when N==1 and encoder returns (D,)
            K = K.unsqueeze(0)  # (1, D)
    
        # 3) Sanity-check dims
        Dq = q.size(-1)
        Dk = K.size(-1)
        if Dq != Dk:
            raise ValueError(f"Embedding dim mismatch: query={Dq}, keys={Dk}")
    
        # 4) Cosine scores via normalize + matmul
        q = torch.nn.functional.normalize(q, dim=-1)        # (D,)
        K = torch.nn.functional.normalize(K, dim=-1)        # (N, D)
        scores = (K @ q)                                    # (N,)
        scores_np = scores.detach().cpu().numpy()
        
        
#        # ---- NEW: print global rank of a specific triplet (even if not in top-k) ----
#        target = ('ernest_augustus_i_of_hanover', 'nationality', 'united_kingdom')
#        # find its index in triplet_list
#        try:
#            t_idx = triplet_list.index(target)   # works if elements are exactly lists like `target`
#        except ValueError:
#            t_idx = None
#        
#        if t_idx is None:
#            print("Target triplet not found in triplet_list.")
#        else:
#            t_score = scores_np[t_idx]
#        
#            # "best" rank with ties (dense-ish): all items strictly greater come before it
#            rank_best = 1 + int((scores_np > t_score).sum())
#        
#            # "worst" rank with ties: counts items >= (so ties come before it)
#            rank_worst = 1 + int((scores_np >= t_score).sum()) - 1
#        
#            print(f"Target score = {t_score:.6f}, rank in all N={len(scores_np)}: "
#                  f"{rank_best} (best-tie) .. {rank_worst} (worst-tie)")
#        # ---------------------------------------------------------------------------
#        
#        # 5) Select top-k safely
#        N = K.size(0)
#        k = min(top_k, N)
#        if k <= 0:
#            return []
#        
#        idx = np.argpartition(scores_np, -k)[-k:]
#        idx = idx[np.argsort(scores_np[idx])[::-1]]
#        
#        return [triplet_list[i] for i in idx]
#    
        # 5) Select top-k safely
        N = K.size(0)
        k = min(top_k, N)
        if k <= 0:
            return []
        # argpartition is O(N); then sort those k
        idx = np.argpartition(scores_np, -k)[-k:]
        idx = idx[np.argsort(scores_np[idx])[::-1]]
    
        return [triplet_list[i] for i in idx]
    
    def filter_retrieve(
        self,
        question: str,
        triplet_list,
        top_k: int = 50,
        threshold: float = 0.5,
        key_batch_size: int = 128,
        return_scores: bool = False,
    ):
        """
        1) Compute cosine similarity for all keys (batched).
        2) Select top_k by cosine.
        3) Run classifier on those top_k only.
        4) Filter by `threshold`; return [] if none pass.
        """
        device = self.device
        triplet_texts = [f"{h} | {r} | {t}" for h, r, t in triplet_list]
        top_k = min(top_k, len(triplet_texts))
        if top_k == 0:
            return []
    
        self.eval()
        with torch.no_grad():
            # ---- Encode query ----
            q = self.encode_queries([question]).to(device)        # (1, D)
            if getattr(q, "is_sparse", False): q = q.to_dense()
    
    
            # ---- Encode all keys in batches & compute cosine scores ----
            cos_scores = []
            key_embeds = []  # store to reuse for the top_k classifier pass
            for i in range(0, len(triplet_texts), key_batch_size):
                batch_texts = triplet_texts[i:i + key_batch_size]
                k = self.encode_keys(batch_texts).to(device)      # (B, D)
                if getattr(k, "is_sparse", False): k = k.to_dense()
    
                key_embeds.append(k)
                cs = torch.nn.functional.cosine_similarity(q, k, dim=-1)  # (B,)
                cos_scores.append(cs.cpu())
    
            cos_scores = torch.cat(cos_scores, dim=0).numpy()     # (N,)
            key_embeds = torch.cat(key_embeds, dim=0)             # (N, D)
    
            # ---- Select top_k by cosine ----
            top_idx_np = np.argsort(cos_scores)[-top_k:][::-1]
            top_idx = torch.as_tensor(top_idx_np.copy(), dtype=torch.long, device=key_embeds.device)
            
            # ---- Classifier on top_k only ----
            k_top = key_embeds.index_select(0, top_idx)     # (K, D)
            q_rep = q.expand(k_top.size(0), -1)             # (K, D)
            probs = torch.sigmoid(self.pair_logits(q_rep, k_top)).cpu().numpy()
            
            # ---- Filter ----
            keep = probs >= threshold
            if not np.any(keep):
                return []
            kept_probs = probs[keep]
            kept_idx = top_idx.cpu().numpy()[keep]
            order = np.argsort(kept_probs)[::-1]
            final_idx = kept_idx[order]
            final_probs = kept_probs[order]
            
            results = [triplet_list[i] for i in final_idx.tolist()]
            return list(zip(results, final_probs.tolist())) if return_scores else results

@torch.no_grad()
def build_1hop_dict(triplets: list) -> dict:
    graph_dict = defaultdict(list)
    for h, r, t in triplets:
        h, r, t = h.strip(), r.strip(), t.strip()
        graph_dict[h].append((r, t))
        graph_dict[t].append((r, h))  # keep undirected, no "_inv"
    return dict(graph_dict)

@torch.no_grad()
def precompute_graph_cache(retriever, triplet_list, device=None):
    entity2idx = build_entity_index(triplet_list)
    concepts = unique_concepts(triplet_list)
    E = retriever.encode_keys(concepts)
    if E.is_sparse: E = E.to_dense()
    if device is not None:
        E = E.to(device)
    concept_embeds = {c: E[i] for i, c in enumerate(concepts)}
    return {"entity2idx": entity2idx, "concept_embeds": concept_embeds}
    
def build_entity_index(triplet_list):
    """entity -> [triplet indices]"""
    idx = defaultdict(list)
    for i, (h, r, t) in enumerate(triplet_list):
        h, r, t = h.strip(), r.strip(), t.strip()
        idx[h].append(i)
        idx[t].append(i)
    return dict(idx)

def unique_concepts(triplet_list):
    s = set()
    for h, r, t in triplet_list:
        s.add(h.strip()); s.add(r.strip()); s.add(t.strip())
    return sorted(s)
    
@torch.no_grad()
def gen_prediction(args):
    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d, args.split, args.retriever)
    os.makedirs(output_dir, exist_ok=True)
    
    


    prediction_file = os.path.join(output_dir, f"retrieved_triplets_{args.top_k}_{args.retriever}_time.jsonl")
    f, processed_results = get_output_file(prediction_file, force=args.force)
    if args.d in ["RoG-cwq", "RoG-webqsp"]:
        dataset = load_dataset(input_file, split=args.split)
    elif args.d == "gr_h":
        dataset = pd.read_json("GRBench/gr_h.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    elif args.d == "pq_2h":
        dataset = pd.read_json("PQ/2H_fullkb.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    elif args.d == "pq_3h":
        dataset = pd.read_json("PQ/3H_fullkb.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    # dataset = dataset.select(range(20))
    # retriever = HybridTripletRetriever().to("cuda:2")
    
    if args.retriever in {"tfidf", "bm25"}:
        retriever = NPTripletRetriever(kind=args.retriever)
    else:
        retriever = TwoTowerModel(encoder = args.retriever, device = args.device).to(args.device)
    if args.retriever == 'kamr':

         state_dict = torch.load("checkpoints/pretrain.pt", map_location="cpu")

         
         load_msg = retriever.load_state_dict(state_dict, strict=True)
         print(load_msg)  # shows missing/unexpected keys

         
    elif args.retriever == 'skp':
	 state_dict = torch.load("checkpoints/pretrain_skp.pt", map_location="cpu")

         
         load_msg = retriever.load_state_dict(state_dict, strict=True)
         print(load_msg)  # shows missing/unexpected keys


    for data in tqdm(dataset, desc="Generating predictions"):
        qid = data["id"]
        question = data["question"]
        if qid in processed_results:
            continue


        
        triplet_list = [(h.strip(), r.strip(), t.strip()) for h, r, t in data["graph"]]
        if args.retriever in {"tfidf", "bm25"}:
            top_triplets = retriever.retrieve(question, triplet_list, top_k=args.top_k)
        else:
            
            # Build once per example graph:
            cache = precompute_graph_cache(retriever, triplet_list, device=args.device)
            
            if args.retriever == 'kamr': 
                top_triplets = retriever.retrieve_expand_1hop_component_keys_exclude_joint_seed_backfill(
                # top_triplets = retriever.retrieve_2hop_component_keys_exclude_joint_seed_backfill(
                    question,
                    triplet_list,
                    top_k=args.top_k,          # e.g., 50
                    seed_top=16,               # seeds
                    entity2idx=cache["entity2idx"],
                    concept_embeds=cache["concept_embeds"],  # scores are computed from q on the fly
                    w_rel=1.0,                 # (tunable) bias toward relations
                    w_ent=1.0,
                )
              elif args.retriever == 'hybrid': 
                top_triplets = retriever.hybrid_retrieve(question, triplet_list, top_k=args.top_k)
            else:
                
                top_triplets = retriever.retrieve(question, triplet_list, top_k=args.top_k)
        result = {
            "id": qid,
            "question": question,
            "prediction": [list(t) for t in top_triplets],
            "ground_paths": None,
            "input": question,
            "raw_output": [f"{h} {r} {t}" for h, r, t in top_triplets],
        }

        f.write(json.dumps(result) + "\n")
        f.flush()

    f.close()
    return prediction_file




@torch.no_grad()
def gen_prediction_gretriever(args):
    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d, args.split, args.retriever)
    os.makedirs(output_dir, exist_ok=True)
    
    


    prediction_file = os.path.join(output_dir, f"retrieved_triplets_{args.top_k}_{args.retriever}.jsonl")
    

    f, processed_results = get_output_file(prediction_file, force=args.force)

    if args.d in ["RoG-cwq", "RoG-webqsp"]:
        dataset = load_dataset(input_file, split=args.split)
    elif args.d == "gr_h":
        dataset = pd.read_json("GRBench/gr_h.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    elif args.d == "pq_2h":
        dataset = pd.read_json("PQ/2H_fullkb.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    elif args.d == "pq_3h":
        dataset = pd.read_json("PQ/3H_fullkb.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    retriever = TwoTowerModel(encoder = args.retriever, device = args.device).to(args.device)
    retriever.eval()



    for data in tqdm(dataset, desc="G-Retriever PCST"):
        qid = data["id"]
        question = data["question"]
        if qid in processed_results:
            continue

        triplets = [(h.strip(), r.strip(), t.strip()) for h, r, t in data["graph"]]
        if not triplets:
            continue

        # Extract unique entities and relations
        entities = sorted(set([h for h, _, _ in triplets] + [t for _, _, t in triplets]))
        relations = sorted(set([r for _, r, _ in triplets]))
        all_nodes = entities
        all_edges = relations

        # Encode question
        q_emb = retriever.encode_queries([question])  # (1, D)

        # Encode node and edge labels
        node_embs = retriever.encode_keys(all_nodes)  # (N_nodes, D)
        edge_embs = retriever.encode_keys(all_edges)  # (N_edges, D)

        # Cosine similarity scores
        node_sims = torch.nn.functional.cosine_similarity(q_emb, node_embs).cpu()
        edge_sims = torch.nn.functional.cosine_similarity(q_emb, edge_embs).cpu()

        # Node prizes
        topk = min(20, len(all_nodes))
        _, topk_idx = torch.topk(node_sims, topk)
        n_prizes = torch.zeros(len(all_nodes))
        n_prizes[topk_idx] = torch.arange(topk, 0, -1).float()

        # Edge prizes (with reweighting trick)
        topk_e = min(20, len(edge_sims.unique()))
        topk_vals, _ = torch.topk(edge_sims.unique(), topk_e)
        e_prizes = edge_sims.clone()
        e_prizes[e_prizes < topk_vals[-1]] = 0.0
        last = topk_e
        q = 0.90  # 90th percentile ? sparser; try 0.8 for less sparse
        Ce_init = torch.quantile(e_prizes, q).item()
        c = 0.6  # decay in prize distribution
        for k in range(topk_e):
            indices = (e_prizes == topk_vals[k])
            weight = min((topk_e - k) / indices.sum(), last)
            e_prizes[indices] = weight
            last = weight * (1 - c)
        Ce = min(Ce_init, e_prizes.max().item() * (1 - c / 2))  # keep your cap, or drop it

        # Build edge list for pcst
        edge_list = []
        edge_costs = []
        virtual_node_prizes = []
        virtual_edges = []
        virtual_costs = []
        mapping_n = {}
        mapping_e = {}
        entity_idx = {ent: idx for idx, ent in enumerate(all_nodes)}
        relation_idx = {rel: idx for idx, rel in enumerate(all_edges)}

        for i, (h, r, t) in enumerate(triplets):
            src, dst = entity_idx[h], entity_idx[t]
            prize_e = e_prizes[relation_idx[r]]
            if prize_e <= Ce:
                mapping_e[len(edge_list)] = i
                edge_list.append((src, dst))
                edge_costs.append(Ce - prize_e)
            else:
                virtual_id = len(all_nodes) + len(virtual_node_prizes)
                mapping_n[virtual_id] = i
                virtual_edges.append((src, virtual_id))
                virtual_edges.append((virtual_id, dst))
                virtual_costs.extend([0, 0])
                virtual_node_prizes.append(prize_e - Ce)

        # Combine
        all_prizes = np.concatenate([n_prizes.numpy(), np.array(virtual_node_prizes)])
        edges = edge_list + virtual_edges
        costs = edge_costs + virtual_costs
        if len(all_nodes) > 0:
            root = int(torch.argmax(node_sims).item())
        else:
            root = -1
        num_clusters = 1
        pruning = 'gw'

        vertices, selected = pcst_fast(edges, all_prizes.tolist(), costs, root, num_clusters, pruning, 0)

        # Recover triplets
        num_real_edges = len(edge_list)
        selected_nodes = [v for v in vertices if v < len(all_nodes)]
        selected_edges = [mapping_e[e] for e in selected if e < num_real_edges]
        virtual_nodes = [v for v in vertices if v >= len(all_nodes)]
        if virtual_nodes:
            virtual_edges_extra = [mapping_n[v] for v in virtual_nodes]
            selected_edges += virtual_edges_extra

        # selected_edges = list(set(selected_edges))
        # final_triplets = [triplets[i] for i in selected_edges]
        
        selected_edges = list(set(selected_edges))

        # ---- Hard cap on number of retrieved triplets ----
        # Add a new hyperparameter, e.g. in args: args.max_triplets (default: 10 or 20)
        max_triplets = getattr(args, "max_triplets", 10)
        
        if len(selected_edges) > max_triplets:
            # Score each selected edge by its relation similarity (edge_sims)
            triplet_scores = []
            for i in selected_edges:
                r = triplets[i][1]                  # relation string
                rel_idx = relation_idx[r]           # index in all_edges / edge_sims
                triplet_scores.append(edge_sims[rel_idx].item())
        
            scores_tensor = torch.tensor(triplet_scores)
            topk = min(max_triplets, len(selected_edges))
            _, keep_idx = torch.topk(scores_tensor, topk)
        
            # Keep only top-scoring edges
            keep_idx = keep_idx.tolist()
            selected_edges = [selected_edges[j] for j in keep_idx]
        
        # Now build final triplets
        final_triplets = [triplets[i] for i in selected_edges]

        
        result = {
            "id": qid,
            "question": question,
            "prediction": [list(t) for t in final_triplets],
            "ground_paths": None,
            "input": question,
            "raw_output": [f"{h} {r} {t}" for h, r, t in final_triplets],
        }

        f.write(json.dumps(result) + "\n")
        f.flush()

    f.close()
    return prediction_file

@torch.no_grad()
def gen_prediction_grag(args):
    input_file = os.path.join(args.data_path, args.d)
    output_dir = os.path.join(args.output_path, args.d, args.split, args.retriever)
    os.makedirs(output_dir, exist_ok=True)
    
    


    prediction_file = os.path.join(output_dir, f"retrieved_triplets_{args.top_k}_{args.retriever}.jsonl")
    
    f, processed_results = get_output_file(prediction_file, force=args.force)

    if args.d in ["RoG-cwq", "RoG-webqsp"]:
        dataset = load_dataset(input_file, split=args.split)
    elif args.d == "gr_h":
        dataset = pd.read_json("GRBench/gr_h.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    elif args.d == "pq_2h":
        dataset = pd.read_json("PQ/2H_fullkb.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    elif args.d == "pq_3h":
        dataset = pd.read_json("PQ/3H_fullkb.jsonl", lines=True)
        dataset = dataset.to_dict(orient="records")   # list of dicts
    # dataset = dataset.select(range(20))
    retriever = TwoTowerModel(encoder = args.retriever, device = args.device).to(args.device)
    # retriever = HybridRetriever().to("cuda:1")
    # Caches for entity/relation texts ? embedding
    concept_embedding_cache = {}
    all_concepts = set()

    # First pass: collect all unique entities and relations
    print("Collecting concepts...")
    for data in tqdm(dataset, desc="Scanning dataset"):
        graph = data["graph"]
        for h, r, t in graph:
            all_concepts.update([h.strip(), r.strip(), t.strip()])

    all_concepts = sorted(list(all_concepts))  # deterministic order
    print(f"Total unique concepts: {len(all_concepts)}")

    # Batch encode all concepts (entities + relations)
    print("Encoding concepts...")
    batch_size = 128
    all_embeddings = []
    for i in tqdm(range(0, len(all_concepts), batch_size)):
        batch = all_concepts[i:i+batch_size]
        emb = retriever.encode_keys(batch)  # (B, D)
        all_embeddings.append(emb)
    all_embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    concept_embedding_cache = dict(zip(all_concepts, all_embeddings))

    for data in tqdm(dataset, desc="Processing examples"):
        qid = data["id"]
        question = data["question"]
        if qid in processed_results:
            continue

        # Encode the question
        query_vec = retriever.encode_queries([question])  # (1, D)

        # Build 1-hop subgraphs
        graph_dict = build_1hop_dict(data["graph"])  # entity ? [(rel, neighbor)]
        subgraph_reprs = []
        subgraph_triplets = []
        center_entities = list(graph_dict.keys())

        for entity in center_entities:
            neighbors = graph_dict[entity]
            concept_set = set()
            
            # Always include center entity
            concept_set.add(entity)
            
            for r, t in neighbors:
                concept_set.add(r)
                concept_set.add(t)
            
            # Collect embeddings from cache
            embs = [concept_embedding_cache[c].unsqueeze(0) for c in concept_set if c in concept_embedding_cache]

            if not embs:
                continue  # Skip if nothing to aggregate

            pooled = torch.mean(torch.cat(embs, dim=0), dim=0, keepdim=True)  # (1, D)
            subgraph_reprs.append(pooled)
            
            # Reconstruct triplet form for saving (entity, r, t)
            subgraph_triplets.append([(entity, r, t) for r, t in neighbors])
            
            
        if not subgraph_reprs:
            print(f"No valid subgraphs for question id: {qid}")
            continue

        subgraph_reprs = torch.cat(subgraph_reprs, dim=0)  # (num_subgraphs, D)
        sims = torch.nn.functional.cosine_similarity(query_vec, subgraph_reprs)  # (num_subgraphs,)
        top_indices = torch.topk(sims, k=min(args.top_k, sims.shape[0])).indices.tolist()

        # Save top-K retrieved subgraphs
        top_subgraphs = [subgraph_triplets[i] for i in top_indices]

        result = {
            "id": qid,
            "question": question,
            "prediction": top_subgraphs,
            "ground_paths": None,
            "input": question,
            "raw_output": [[f"{h} {r} {t}" for h, r, t in top_subgraphs[i]] for i in range(len(top_subgraphs))],
        }

        f.write(json.dumps(result) + "\n")
        f.flush()

    f.close()
    return prediction_file
#def gen_prediction(args):
#    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
#    if args.lora or os.path.exists(args.model_path + "/adapter_config.json"):
#        print("Load LORA model")
#        model = AutoPeftModelForCausalLM.from_pretrained(
#            args.model_path, device_map="auto", torch_dtype=torch.bfloat16
#        )
#    else:
#        model = AutoModelForCausalLM.from_pretrained(
#            args.model_path,
#            device_map="auto",
#            torch_dtype=torch.float16,
#            use_auth_token=True,
#        )
#
#    input_file = os.path.join(args.data_path, args.d)
#    output_dir = os.path.join(args.output_path, args.d, args.model_name, args.split)
#    print("Save results to: ", output_dir)
#
#    retriever = TwoTowerModel().to("cuda:1")
#    
#    # Load dataset
#    dataset = load_dataset(input_file, split=args.split)
#
#    # Load prompt template
#    prompter = utils.InstructFormater(args.prompt_path)
#
#    def prepare_dataset(sample):
#        # Prepare input prompt
#        sample["text"] = prompter.format(
#            instruction=INSTRUCTION, message=sample["question"]
#        )
#        # Find ground-truth paths for each Q-P pair
#        graph = utils.build_graph(sample["graph"])
#        paths = utils.get_truth_paths(sample["q_entity"], sample["a_entity"], graph)
#        ground_paths = set()
#        for path in paths:
#            ground_paths.add(tuple([p[1] for p in path]))  # extract relation path
#        sample["ground_paths"] = list(ground_paths)
#        return sample
#
#    dataset = dataset.map(
#        prepare_dataset,
#        num_proc=N_CPUS,
#    )
#
#    # Predict
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#
#    prediction_file = os.path.join(
#        output_dir, f"predictions_{args.n_beam}_{args.do_sample}.jsonl"
#    )
#    f, processed_results = get_output_file(prediction_file, force=args.force)
#    for data in tqdm(dataset):
#        question = data["question"]
#        input_text = data["text"]
#        qid = data["id"]
#        if qid in processed_results:
#            continue
#        raw_output = generate_seq(
#            model,
#            input_text,
#            tokenizer,
#            max_new_tokens=args.max_new_tokens,
#            num_beam=args.n_beam,
#            do_sample=args.do_sample,
#        )
#        rel_paths = parse_prediction(raw_output["paths"])
#        if args.debug:
#            print("ID: ", qid)
#            print("Question: ", question)
#            print("Prediction: ", rel_paths)
#        # prediction = outputs[0]["generated_text"].strip()
#        data = {
#            "id": qid,
#            "question": question,
#            "prediction": rel_paths,
#            "ground_paths": data["ground_paths"],
#            "input": input_text,
#            "raw_output": raw_output,
#        }
#        f.write(json.dumps(data) + "\n")
#        f.flush()
#    f.close()
#
#    return prediction_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="rmanluo"
    )
    parser.add_argument("--d", "-d", type=str, default="RoG-webqsp")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument("--output_path", type=str, default="results/gen_rule_path")
    parser.add_argument("--retriever", type=str, default="bge")
    parser.add_argument(
        "--model_name",
        type=str,
        help="model_name for save results",
        default="Llama-2-7b-hf",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="model_name for save results",
        default="meta-llama/Llama-2-7b-chat-hf",
    )
    parser.add_argument(
        "--prompt_path", type=str, help="prompt_path", default="prompts/llama2.txt"
    )
    parser.add_argument(
        "--rel_dict",
        nargs="+",
        default=["datasets/KG/fbnet/relations.dict"],
        help="relation dictionary",
    )
    parser.add_argument(
        "--force", "-f", action="store_true", help="force to overwrite the results"
    )
    parser.add_argument(
        "--device", type=str, help="device", default="cuda:0"
    )
    parser.add_argument("--debug", action="store_true", help="Debug")
    parser.add_argument("--lora", action="store_true", help="load lora weights")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--n_beam", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", help="do sampling")

    args = parser.parse_args()


    # gen_path = gen_prediction_gretriever(args)
    if args.retriever == "grag":
        gen_path = gen_prediction_grag(args)
    elif args.retriever == "gretriever":
        gen_path = gen_prediction_gretriever(args)
    else:
        gen_path = gen_prediction(args)