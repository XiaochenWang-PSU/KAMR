import os
import random
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW

# ---------------------------
# Reproducibility
# ---------------------------
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Paths (from your generator)
# ---------------------------
DATASET = "pq_2H"  # change to websqp if you trained that one
BASE = base_path_of_kg

if "pq" not in DATASET:
    QUERIES_CSV = f"{BASE}/queries_{DATASET}.csv"                 # qid,query
    KEYS_CSV    = f"{BASE}/keys_{DATASET}.csv"                    # kid,doc  ("h | r | t")
    LABELS_CSV  = f"{BASE}/labels_{DATASET}.csv.gz"               # qid,kid,lid  (positive-only rows exist, lid=0)
    LMAP_CSV    = f"{BASE}/label_map_{DATASET}.csv"               # lid,label
    QER_CSV     = f"{BASE}/question_qer_map_{DATASET}.csv"        # qid,entity,relation
else:
    QUERIES_CSV = f"{BASE}/{DATASET}_queries.csv"                 # qid,query
    KEYS_CSV    = f"{BASE}/{DATASET}_keys.csv"                    # kid,doc  ("h | r | t")
    LABELS_CSV  = f"{BASE}/{DATASET}_labels.csv.gz"               # qid,kid,lid  (positive-only rows exist, lid=0)
    LMAP_CSV    = f"{BASE}/{DATASET}_label_map.csv"               # lid,label
    QER_CSV     = f"{BASE}/{DATASET}_question_qer_map.csv"        # qid,entity,relation

# ---------------------------
# Load data
# ---------------------------
queries_df = pd.read_csv(QUERIES_CSV)
keys_df    = pd.read_csv(KEYS_CSV)
labels_df  = pd.read_csv(LABELS_CSV, compression="gzip")# [:10000]
label_map  = pd.read_csv(LMAP_CSV)
qer_df     = pd.read_csv(QER_CSV)

# id -> text maps
qid_to_query: Dict[int, str] = dict(zip(queries_df["qid"], queries_df["query"]))
kid_to_doc:   Dict[int, str] = dict(zip(keys_df["kid"], keys_df["doc"]))

# Keep only positives for retrieval training
labels_pos = labels_df[labels_df["lid"] == 0].copy()

# Group positives per qid (one pos per question is enough; if multiple, take first)
pos_by_qid = labels_pos.sort_values(["qid"]).groupby("qid").first().reset_index()
pos_by_qid["doc_text"] = pos_by_qid["kid"].map(kid_to_doc)
pos_by_qid["query"]    = pos_by_qid["qid"].map(qid_to_query)

# Join q?(e,r) mapping
qer_map = qer_df.groupby("qid").first().reset_index()
train_df = pos_by_qid.merge(qer_map, on="qid", how="left")  # qid, kid, lid, doc_text, query, entity, relation
train_df = train_df.dropna(subset=["entity", "relation"])    # safety

# ---------------------------
# Dataset
# ---------------------------
class RetrievalDataset(Dataset):
    """
    Returns:
      - query_text
      - doc_text (full h|r|t)  [kept as value for later]
      - entity_text, relation_text (from qer_map)
    """
    def __init__(self, df: pd.DataFrame):
        self.qids      = df["qid"].tolist()
        self.queries   = df["query"].tolist()
        self.docs      = df["doc_text"].tolist()
        self.entities  = df["entity"].astype(str).tolist()
        # strip prefix like 'ns.xxx' -> 'xxx' if needed
        self.relations = [r.split('.')[-1] for r in df["relation"].astype(str).tolist()]

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        return {
            "qid": self.qids[idx],
            "query": self.queries[idx],
            "doc": self.docs[idx],          # full h|r|t (not used in loss)
            "entity": self.entities[idx],   # from QER mapping
            "relation": self.relations[idx],
        }

# ---------------------------
# Encoders
# ---------------------------
def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B,T,1)
    summed = (last_hidden_state * mask).sum(dim=1)                  # (B,H)
    count  = mask.sum(dim=1).clamp(min=1e-9)                        # (B,1)
    return summed / count

class TwoTowerModel(nn.Module):
    """
    Only needs:
      - query encoder
      - key encoder (for "entity | relation" keys)
      - entity encoder (for entity and relation separately)
    """
    def __init__(self, encoder="dragon", use_cls=True):
        super().__init__()
        self.encoder_name = encoder
        self.use_cls = use_cls  # if False, mean-pool

        if encoder == "distilbert":
            name = "distilbert/distilbert-base-uncased"
            self.query_tokenizer = AutoTokenizer.from_pretrained(name)
            self.key_tokenizer   = AutoTokenizer.from_pretrained(name)
            self.entity_tokenizer= AutoTokenizer.from_pretrained(name)
            self.query_encoder   = AutoModel.from_pretrained(name)
            self.key_encoder     = AutoModel.from_pretrained(name)
            self.entity_encoder  = AutoModel.from_pretrained(name)

        elif encoder == "mxbai":
            name = "mixedbread-ai/mxbai-rerank-base-v2"
            self.query_tokenizer = AutoTokenizer.from_pretrained(name)
            self.key_tokenizer   = AutoTokenizer.from_pretrained(name)
            self.entity_tokenizer= AutoTokenizer.from_pretrained(name)
            self.query_encoder   = AutoModel.from_pretrained(name)
            self.key_encoder     = AutoModel.from_pretrained(name)
            self.entity_encoder  = AutoModel.from_pretrained(name)

        elif encoder == "bce":
            name = "maidalun1020/bce-embedding-base_v1"
            self.query_tokenizer = AutoTokenizer.from_pretrained(name)
            self.key_tokenizer   = AutoTokenizer.from_pretrained(name)
            self.entity_tokenizer= AutoTokenizer.from_pretrained(name)
            self.query_encoder   = AutoModel.from_pretrained(name)
            self.key_encoder     = AutoModel.from_pretrained(name)
            self.entity_encoder  = AutoModel.from_pretrained(name)

        elif encoder == "twolar":
            name = "Dundalia/TWOLAR-large"
            self.query_tokenizer = AutoTokenizer.from_pretrained(name)
            self.key_tokenizer   = AutoTokenizer.from_pretrained(name)
            self.entity_tokenizer= AutoTokenizer.from_pretrained(name)
            self.query_encoder   = AutoModelForSeq2SeqLM.from_pretrained(name)
            self.key_encoder     = AutoModelForSeq2SeqLM.from_pretrained(name)
            self.entity_encoder  = AutoModelForSeq2SeqLM.from_pretrained(name)

        elif encoder == "colbert":
            name = "colbert-ir/colbertv2.0"
            self.query_tokenizer = AutoTokenizer.from_pretrained(name)
            self.key_tokenizer   = AutoTokenizer.from_pretrained(name)
            self.entity_tokenizer= AutoTokenizer.from_pretrained(name)
            self.query_encoder   = AutoModel.from_pretrained(name)
            self.key_encoder     = AutoModel.from_pretrained(name)
            self.entity_encoder  = AutoModel.from_pretrained(name)

        elif encoder == "dragon":
            qname = "facebook/dragon-plus-query-encoder"
            kname = "facebook/dragon-plus-context-encoder"
            self.query_tokenizer = AutoTokenizer.from_pretrained(qname)
            self.key_tokenizer   = AutoTokenizer.from_pretrained(kname)
            self.entity_tokenizer= self.key_tokenizer
            self.query_encoder   = AutoModel.from_pretrained(qname)
            self.key_encoder     = AutoModel.from_pretrained(kname)
            self.entity_encoder  = self.key_encoder
            # self.entity_encoder  = AutoModel.from_pretrained(kname)
            # freeze query encoder if desired
            # for param in self.query_encoder.parameters():
            #     param.requires_grad = False

        else:
            raise ValueError(f"Unsupported encoder: {encoder}")

        # Hidden size (read from one encoder config)
        hdim = getattr(self.query_encoder.config, "hidden_size", None)
        if hdim is None and hasattr(self.query_encoder.config, "d_model"):
            hdim = self.query_encoder.config.d_model
        assert hdim is not None, "Cannot determine hidden size."

        # Temperature for similarity logits
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def _encode(self, encoder, tokenizer, texts, max_len=128):
        tokens = tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_len
        ).to(next(self.parameters()).device)
        out = encoder(**tokens)
        if hasattr(out, "last_hidden_state"):
            rep = out.last_hidden_state
        else:
            rep = out[0]
        pooled = rep[:, 0, :] if self.use_cls else _mean_pool(rep, tokens["attention_mask"])
        return pooled

    def encode_queries(self, texts):
        return self._encode(self.query_encoder, self.query_tokenizer, texts)

    def encode_keys(self, texts):
        # used for "entity | relation" composite keys
        return self._encode(self.key_encoder, self.key_tokenizer, texts)

    def encode_entities(self, texts):
        # used for both entities and relations (as separate concepts)
        return self._encode(self.entity_encoder, self.entity_tokenizer, texts)

# ---------------------------
# Loss (BCE with in-batch negatives)
# ---------------------------
bce = nn.BCEWithLogitsLoss(reduction="mean")

def bce_inbatch_loss(q: torch.Tensor, k: torch.Tensor, logit_scale: torch.Tensor):
    """
    q: (B, H)  queries
    k: (B, H)  positives aligned by index; all other pairs are negatives
    logit_scale: scalar nn.Parameter (multiplicative on cosine sims)
    """
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)

    logits = logit_scale * (q @ k.t())  # (B, B)
    B = logits.size(0)
    targets = torch.eye(B, device=logits.device)  # positives on diagonal, others 0
    return bce(logits, targets)

# ---------------------------
# DataLoader
# ---------------------------
retr_ds  = RetrievalDataset(train_df)
retr_dl  = DataLoader(retr_ds, batch_size=16, shuffle=True, drop_last=True)

# ---------------------------
# Model / Optim / Scheduler
# ---------------------------
model = TwoTowerModel(encoder="dragon", use_cls=True).to(device)

encoder_lr = 2e-5
weight_decay = 0.01
no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight")

def split_groups(named_params, lr):
    decay, no_decay = [], []
    for n, p in named_params:
        if not p.requires_grad:
            continue
        (no_decay if any(k in n for k in no_decay_keys) else decay).append(p)
    return [
        {"params": decay,    "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": lr, "weight_decay": 0.0},
    ]

# only train key_encoder + entity_encoder (query encoder is frozen for dragon)
opt_groups = []
opt_groups = split_groups(model.key_encoder.named_parameters(), lr=encoder_lr)
#for enc_attr in ["key_encoder", "entity_encoder"]:
#    if hasattr(model, enc_attr) and getattr(model, enc_attr) is not None:
#        opt_groups += split_groups(getattr(model, enc_attr).named_parameters(), lr=encoder_lr)

optim = AdamW(opt_groups, betas=(0.9, 0.999), eps=1e-8)

# Scheduler: cosine with 10% warmup
epochs = 10
steps_per_epoch = len(retr_dl)  # only retrieval step now
total_steps = epochs * steps_per_epoch
num_warmup = int(0.1 * total_steps)

scheduler = get_cosine_schedule_with_warmup(
    optim, num_warmup_steps=num_warmup, num_training_steps=total_steps
)

# weights
w_pair = 0.0   # composite (entity | relation) key loss
w_ent  = 1.0   # concept loss on entity + relation

# ---------------------------
# Training loop: (entity|relation) + entity & relation concepts
# ---------------------------
for epoch in range(1, epochs + 1):
    model.train()
    tot_pair = tot_ent = tot = 0.0

    for batch in tqdm(retr_dl, desc=f"[Epoch {epoch}] er-pair + concepts"):
        q_txt = batch["query"]
        e_txt = batch["entity"]
        r_txt = batch["relation"]

        # composite key: "entity | relation"
        er_txt = [f"{e} | {r}" for e, r in zip(e_txt, r_txt)]

        # Encode
        q  = model.encode_queries(q_txt)        # (B, D)
        k_er = model.encode_keys(er_txt)        # (B, D)

        # entity and relation as separate concepts
        e = model.encode_entities(e_txt)        # (B, D)
        r = model.encode_entities(r_txt)        # (B, D)

        # composite pair loss: q ? (entity | relation)
        loss_pair = bce_inbatch_loss(q, k_er, model.logit_scale)

        # concept loss: q ? entity and q ? relation
        # stack e and r, duplicate q
        q2 = torch.cat([q, q], dim=0)           # (2B, D)
        k2 = torch.cat([e, r], dim=0)           # (2B, D)
        loss_ent = bce_inbatch_loss(q2, k2, model.logit_scale)

        loss_retr = w_pair * loss_pair + w_ent * loss_ent

        optim.zero_grad()
        loss_retr.backward()
        optim.step()
        scheduler.step()

        tot_pair += loss_pair.item()
        tot_ent  += loss_ent.item()
        tot      += loss_retr.item()

    denom = max(1, len(retr_dl))
    print(
        f"Epoch {epoch:02d} | pair(er) {tot_pair/denom:.8f} "
        f"| ent+rel {tot_ent/denom:.8f} "
        f"| total {tot/denom:.8f}"
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints/pretrain_{DATASET}_epoch{epoch}_loss_ent.pt")

# ---------------------------
# (Optional) quick retrieval eval (still using full doc_texts as values)
# ---------------------------
@torch.no_grad()
def batched_embed(model, texts, fn, batch=256):
    model.eval()
    outs = []
    for i in range(0, len(texts), batch):
        v = fn(texts[i:i+batch]).detach().cpu()
        outs.append(v)
    return torch.cat(outs, dim=0)

@torch.no_grad()
def evaluate_retrieval(model, labels_df, sample_q=500):
    """
    Quick check: use the key_encoder on full doc_text ('h | r | t') as values.
    Even though training used 'entity | relation', this gives a rough sanity check.
    """
    model.eval()
    qs = list(queries_df["query"])[:sample_q]
    qv = batched_embed(model, qs, model.encode_queries, batch=256)
    ds = list(keys_df["doc"])
    dv = batched_embed(model, ds, model.encode_keys, batch=256)
    qv = F.normalize(qv, dim=-1); dv = F.normalize(dv, dim=-1)
    sims = dv @ qv.t()  # (N, Q)

    # build qid->pos kid map
    pos_map = labels_df[labels_df.lid==0].groupby("qid").first().kid.to_dict()

    hits1 = 0
    for col, qid in enumerate(list(queries_df["qid"])[:sample_q]):
        if qid not in pos_map:
            continue
        pos_kid = pos_map[qid]
        order = torch.argsort(sims[:, col], descending=True).cpu().tolist()
        ranked_kids = [keys_df.iloc[i].kid for i in order]
        if pos_kid in ranked_kids[:1]:
            hits1 += 1
    total = min(sample_q, len(queries_df))
    print(f"Hits@1 ~ {hits1/total:.4f}")

# evaluate_retrieval(model, labels_df, sample_q=500)
