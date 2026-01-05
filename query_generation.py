import os
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Manager
import multiprocessing
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =========================
# LLM prompt helpers (your originals)
# =========================
def construct_prompt(triple, mask_choice):
    head, relation, tail = triple
    if mask_choice == 'head':
        masked = f"?? - {relation} - {tail}"
    else:  # only head or tail, never relation
        masked = f"{head} - {relation} - ??"
    return (
        "You are given a triple from a knowledge graph in the form of (head, relation, tail), but one element is missing, shown as '??'.\n"
        "Your job is to generate a **single** natural language question based on the incomplete triple, such that the missing part is what the question is asking for.\n"
        "The question should be a complete sentence ending with a question mark.\n"
        "Do NOT generate multiple questions. Stop after the first question. Do NOT include the masked answer or the triple in your output.\n"
        f"\nTriple: {masked}\nMasked Answer: {triple}\nOutput:"
    ), masked

def extract_first_question(text: str) -> str:
    """
    Robustly take the first sentence ending with '?' from a decoded continuation.
    Falls back to the first line.
    """
    text = (text or "").strip()
    # Remove any leading "assistant:" or similar headers if present
    for marker in ["assistant", "Assistant", "ASSISTANT", "### Assistant"]:
        if text.lower().startswith(marker.lower()):
            text = text[len(marker):].lstrip(":").strip()

    # Take up to the first '?'.
    if "?" in text:
        return text.split("?")[0].strip() + "?"
    # Fallback: first non-empty line
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text

def worker(gpu_id, prompts_chunk_with_indices, return_list):
    import warnings, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tqdm import tqdm
    warnings.filterwarnings("ignore", category=UserWarning)

    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
        device_map={"": device}
    )

    local_outputs = []
    batch_size = 64
    max_prompt_tokens = 3000
    max_input_len = 2048
    max_new_tokens = 300

    for i in tqdm(range(0, len(prompts_chunk_with_indices), batch_size), desc=f"GPU {gpu_id}"):
        batch = prompts_chunk_with_indices[i:i + batch_size]
        batch_indices, batch_prompts = zip(*batch)

        # Build chat-formatted strings
        input_texts = [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{p}\n"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            for p in batch_prompts
        ]

        # Prefilter by tokenized length of the *raw* user prompts
        raw_lengths = [len(tokenizer(p)["input_ids"]) for p in batch_prompts]

        # ? FIX: zip indices, texts, and lengths together
        filtered = [(idx, txt) for idx, txt, L in zip(batch_indices, input_texts, raw_lengths) if L < max_prompt_tokens]
        if not filtered:
            continue

        f_indices, f_inputs = zip(*filtered)

        enc = tokenizer(
            list(f_inputs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len
        ).to(device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )

        # Decode only the continuation
        input_len = enc.input_ids.shape[1]
        cont = gen[:, input_len:]
        decoded = tokenizer.batch_decode(cont, skip_special_tokens=True)

        for idx, text in zip(f_indices, decoded):
            q = extract_first_question(text)
            print(q)
            local_outputs.append((idx, q))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return_list.extend(local_outputs)


# =========================
# File & prompt utilities
# =========================
def read_txt_kb(path):
    """
    Reads a PathQuestion-style kb txt file. Each line is like:
    head relation tail \
    We'll take the first three whitespace-separated tokens on each valid line.
    """
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            # remove trailing backslash if present
            if ln.endswith("\\"):
                ln = ln[:-1].strip()
            parts = ln.split()
            if len(parts) < 3:
                continue
            h, r, t = parts[0], parts[1], parts[2]
            triples.append((h, r, t))
    return triples

def build_entity_index(triplet_list):
    ent2triples = defaultdict(set)
    for h, r, t in triplet_list:
        ent2triples[h].add((h, r, t))
        ent2triples[t].add((h, r, t))
    return ent2triples

# =========================
# Main generation per KB
# =========================
def process_kb_txt(kb_path, out_prefix, sample_fraction=1.0, filter_substr="is_reviewed"):
    print(f"\n=== Processing {kb_path} ===")
    triplets = read_txt_kb(kb_path)
    print(f"Loaded {len(triplets)} raw triples.")

    # Optional filter
    if filter_substr:
        filtered = [tr for tr in triplets if filter_substr.lower() not in str(tr[1]).lower()]
        print(f"Filtered out {len(triplets) - len(filtered)} by relation containing '{filter_substr}'.")
        triplets = filtered

    # Optional subsample
    if 0 < sample_fraction < 1.0:
        k = max(1, int(len(triplets) * sample_fraction))
        triplets = random.sample(triplets, k)
        print(f"Subsampled to {len(triplets)} triples.")

    # Build prompts (+ metadata for q?(e,r))
    prompts, metas = [], []
    for tr in tqdm(triplets, desc="Building prompts"):
        choice = random.choice(["head", "tail"])
        prompt, masked = construct_prompt(tr, choice)
        # record unmasked entity for (q, e, r) supervision
        h, r, t = tr
        unmasked_entity = t if choice == "head" else h

        prompts.append(prompt)
        metas.append({
            "triple": tr,
            "masked_repr": masked,
            "unmasked_entity": unmasked_entity,
            "relation": r
        })

    # Parallel generation across GPUs (falls back to 1 proc on CPU/GPU)
    num_gpus = torch.cuda.device_count()
    manager = Manager()
    return_list = manager.list()
    procs = []

    if num_gpus == 0:
        # Single CPU process
        print("No CUDA device found  generating on CPU in a single process.")
        worker(0, list(enumerate(prompts)), return_list)
    else:
        # Split prompts across all GPUs
        n = len(prompts)
        ndevs = num_gpus
        split = max(1, n // ndevs)

        chunks = []
        for i in range(ndevs):
            start = i * split
            end = (i + 1) * split
            chunk_prompts = prompts[start:end]
            chunk_indices = list(range(start, start + len(chunk_prompts)))
            if chunk_prompts:
                chunks.append(list(zip(chunk_indices, chunk_prompts)))

        # remainder to last chunk
        rem_start = ndevs * split
        if rem_start < len(prompts) and chunks:
            chunks[-1].extend(zip(range(rem_start, len(prompts)), prompts[rem_start:]))

        for gpu_id, chunk in enumerate(chunks):
            p = Process(target=worker, args=(gpu_id, chunk, return_list))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    # Reassemble
    idx2q = {idx: q for idx, q in return_list}
    questions = []
    for i, meta in enumerate(metas):
        q = idx2q.get(i, "").strip()
        if not q:
            q = f"Based on {meta['masked_repr']}, what is the missing element?"
        questions.append(q)

    # =========================
    # Save outputs
    # =========================
    # 1) Triple ? Question (direct mapping)
    t2q_rows = []
    for meta, q in zip(metas, questions):
        h, r, t = meta["triple"]
        t2q_rows.append({"head": h, "relation": r, "tail": t, "question": q})
    t2q_df = pd.DataFrame(t2q_rows)
    t2q_df.to_csv(f"{out_prefix}_triple_to_question.csv", index=False)

    # 2) IR (positive-only) pack: queries, keys, labels, label map
    ir_rows = []
    for q, meta in zip(questions, metas):
        pos_doc = " | ".join(map(str, meta["triple"]))
        ir_rows.append({"query": q, "doc": pos_doc, "label": "Positive"})
    ir_df = pd.DataFrame(ir_rows)

    ir_df['qid'], qid_uniques = pd.factorize(ir_df['query'])
    ir_df['kid'], kid_uniques = pd.factorize(ir_df['doc'])
    ir_df['lid'], lid_uniques = pd.factorize(ir_df['label'])  # single class

    queries_df = pd.DataFrame({'qid': range(len(qid_uniques)), 'query': qid_uniques})
    keys_df = pd.DataFrame({'kid': range(len(kid_uniques)), 'doc': kid_uniques})
    labels_df = ir_df[['qid', 'kid', 'lid']]
    label_map_df = pd.DataFrame({'lid': range(len(lid_uniques)), 'label': lid_uniques})

    queries_df.to_csv(f"{out_prefix}_queries.csv", index=False)
    keys_df.to_csv(f"{out_prefix}_keys.csv", index=False)
    labels_df.to_csv(f"{out_prefix}_labels.csv.gz", index=False, compression="gzip")
    label_map_df.to_csv(f"{out_prefix}_label_map.csv", index=False)

    # 3) q?(e,r) mapping for single-hop supervision
    qer_rows = []
    for q, meta in zip(questions, metas):
        qer_rows.append({
            "query": q,
            "entity": str(meta["unmasked_entity"]),
            "relation": str(meta["relation"])
        })
    qer_df = pd.DataFrame(qer_rows)
    qer_df = qer_df.merge(queries_df, on="query", how="left")[['qid', 'entity', 'relation']]
    qer_df.to_csv(f"{out_prefix}_question_qer_map.csv", index=False)

    print(f"Saved for {kb_path}:")
    print(f"  - {out_prefix}_triple_to_question.csv")
    print(f"  - {out_prefix}_queries.csv")
    print(f"  - {out_prefix}_keys.csv")
    print(f"  - {out_prefix}_labels.csv.gz")
    print(f"  - {out_prefix}_label_map.csv")
    print(f"  - {out_prefix}_question_qer_map.csv")

def main():
    # -------- Args (tweak as needed) --------
    kb_files = [
        path_of_kb # modify as needed
    ]
    sample_fraction = 1.0  # set <1.0 to subsample for smoke tests

    for path, prefix in kb_files:
        process_kb_txt(path, prefix, sample_fraction=sample_fraction)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
