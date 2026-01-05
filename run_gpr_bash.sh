#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python
SCRIPT=src/qa_prediction/predict_answer.py
PROMPT_PATH=prompts/llama2_predict.txt

# Datasets and models
# DATASETS=(pq_3h pq_2h RoG-cwq)
DATASETS=("RoG-webqsp")
# DATASETS=("RoG-cwq")
# DATASETS=("RoG-webqsp")
# MODELS=("qwen3")
# MODELS=("meta-llama/Llama-2-7b-chat-hf")
# MODELS=("google/flan-t5-xl")
MODELS=('gpt-3.5-turbo')
# Root where rule files live:
# results/gen_rule_path/<DATASET>/test/<METHOD>/retrieved_triplets_50_<METHOD>.jsonl
# (Also supports legacy filename: retrieved_triplets_50.jsonl)
RULE_ROOT=results/gen_rule_path

run_one () {
  local dataset="$1"
  local model="$2"
  local rule_path="$3"

  echo "--------------------------------------------------------------------------------"
  echo "[RUN] dataset=${dataset}  model=${model}"
  echo "      rule_path=${rule_path}"
  echo "--------------------------------------------------------------------------------"

  "${PYTHON_BIN}" "${SCRIPT}" \
    --model_name "${model}" \
    -d "${dataset}" \
    --prompt_path "${PROMPT_PATH}" \
    --add_rule \
    --rule_path "${rule_path}"

    
}


allowed=("kamr")
allowed_str=" ${allowed[*]} "

for dataset in "${DATASETS[@]}"; do
  test_dir="${RULE_ROOT}/${dataset}/test"
  if [[ ! -d "${test_dir}" ]]; then
    echo "[WARN] Missing dir: ${test_dir}   skipping ${dataset}"
    continue
  fi

  # methods are subfolders under .../test/
  mapfile -t METHODS < <(find "${test_dir}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
  if [[ "${#METHODS[@]}" -eq 0 ]]; then
    echo "[WARN] No methods under ${test_dir}   skipping ${dataset}"
    continue
  fi

  for method in "${METHODS[@]}"; do
    # Prefer new pattern with method suffix; fall back to legacy filename if present
    # Only handle specific methods
    #echo "[Path]: ${test_dir}   dataset: ${dataset}"
  if [[ "$allowed_str" != *" $method "* ]]; then
      echo "[SKIP] Method=$method"
      continue
    fi
    rule_new="${test_dir}/${method}/retrieved_triplets_50_${method}.jsonl"
    rule_legacy="${test_dir}/${method}/retrieved_triplets_50.jsonl"

    if [[ -f "${rule_new}" ]]; then
      rule_path="${rule_new}"
    elif [[ -f "${rule_legacy}" ]]; then
      rule_path="${rule_legacy}"
    else
      echo "[INFO] No rule file for method=${method} in ${dataset}   skipping"
      continue
    fi

    for model in "${MODELS[@]}"; do
      run_one "${dataset}" "${model}" "${rule_path}"
    done
  done
done
