# KAMR

Code for **"KAMR: Grounding Generation via Knowledge-Aligned Multi-hop Retrieval"**.

## Datasets

- **PathQuestion**: https://github.com/zmtkeke/IRN/tree/master/PathQuestion  
- **ComplexWebQuestions (CWQ)**: https://huggingface.co/datasets/rmanluo/RoG-cwq  

## Workflow

### 1) Build the pretraining dataset
```bash
python augment.py
```

### 2) Pretrain the retriever
```bash
python pretrain.py


### 3) Run retrieval inference (batch)
```bash
bash run_retr.sh


### 4) Run LLM generation (batch)
```bash
bash run_generation.sh


## Notes

Additional documentation and usage details will be released when the repository is made public.
