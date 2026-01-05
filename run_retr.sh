#!/usr/bin/env bash

########################
# Config
########################
# DATASETS=(pq_3h pq_2h RoG-cwq)
# RETRIEVERS=(distilbert mxbai bge jina splade bce colbert dragon kamr)
DATASETS=(pq_2h)
RETRIEVERS=(kamr)

# GPUs you want to use
GPUS=(0 1 2 3)
#GPUS=(2)
# Concurrency limit per GPU (prevents piling on a GPU during model warmup)
MAX_SLOTS_PER_GPU=1

# Thresholds for "low usage" (extra safety)
UTIL_THRESH=50   # %
MEM_THRESH=50    # % of total GPU memory
SLEEP_SECS=20    # short poll to keep things responsive; adjust if you like
IDLE_CYCLES_BEFORE_STATUS=3

PYTHON_BIN=python
BASE_CMD="src/qa_prediction/gen_rule_path.py"
COMMON_ARGS="--split test --force"

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

########################
# Sanity checks
########################
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?"
  exit 1
fi

########################
# State (Bash 4+: associative arrays)
########################
declare -A SLOTS   # gpu_id -> current running count
declare -A PID2GPU # pid -> gpu_id

for g in "${GPUS[@]}"; do
  SLOTS["$g"]=0
done

########################
# Helpers
########################

# Update SLOTS[] by reaping finished PIDs
reap_finished() {
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      gpu="${PID2GPU[$pid]}"
      # Decrement slot for that GPU if still >0
      if (( SLOTS["$gpu"] > 0 )); then
        SLOTS["$gpu"]=$(( SLOTS["$gpu"] - 1 ))
      fi
      unset "PID2GPU[$pid]"
      echo "[$(date '+%F %T')] Job $pid finished; freed GPU $gpu (slots now ${SLOTS[$gpu]}/$MAX_SLOTS_PER_GPU)."
    fi
  done
}

# Query one-shot GPU metrics for quick decisions
# Populates arrays: UTIL[], MEM_PCT[]
query_gpu_metrics() {
  UTIL=()
  MEM_PCT=()
  # Read all GPUs once
  mapfile -t lines < <(nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits)
  for line in "${lines[@]}"; do
    IFS=',' read -r idx util mu mt <<< "$line"
    idx="${idx//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    mu="${mu//[[:space:]]/}"
    mt="${mt//[[:space:]]/}"
    memp=100
    if [[ "$mt" -gt 0 ]]; then
      memp=$(( 100 * mu / mt ))
    fi
    UTIL[$idx]=$util
    MEM_PCT[$idx]=$memp
  done
}

# Pick a GPU that has an available slot and is under thresholds
pick_gpu() {
  query_gpu_metrics
  for g in "${GPUS[@]}"; do
    # Must have free slot
    if (( SLOTS["$g"] >= MAX_SLOTS_PER_GPU )); then
      continue
    fi
    # Must pass safety thresholds (soft check)
    util="${UTIL[$g]:-100}"
    memp="${MEM_PCT[$g]:-100}"
    if (( util < UTIL_THRESH && memp < MEM_THRESH )); then
      echo "$g"
      return 0
    fi
  done
  return 1
}

# Status line
print_status() {
  query_gpu_metrics
  line="GPU status |"
  for g in "${GPUS[@]}"; do
    line+=" g${g}:slots=${SLOTS[$g]}/$MAX_SLOTS_PER_GPU util=${UTIL[$g]:-NA}% mem=${MEM_PCT[$g]:-NA}% |"
  done
  echo "[$(date '+%F %T')] $line"
}

########################
# Job queue build
########################
QUEUE=()
for d in "${DATASETS[@]}"; do
  for r in "${RETRIEVERS[@]}"; do
    QUEUE+=("$d::$r")
  done
done

########################
# Scheduler
########################
idle_ticks=0
i=0
total=${#QUEUE[@]}
echo "Scheduling $total jobs..."

while (( i < total || ${#PID2GPU[@]} > 0 )); do
  # Reap any finished jobs first
  reap_finished

  # Try to schedule next queued job if any remain
  if (( i < total )); then
    if gpu_id="$(pick_gpu)"; then
      d="${QUEUE[$i]%%::*}"
      r="${QUEUE[$i]##*::}"
      ts="$(date +%Y%m%d_%H%M%S)"
      log="${LOG_DIR}/${d}_${r}_gpu${gpu_id}_${ts}.log"
      cmd="${PYTHON_BIN} ${BASE_CMD} -d ${d} --retriever ${r} ${COMMON_ARGS} --device cuda:${gpu_id}"

      echo "[$(date '+%F %T')] Launching on GPU ${gpu_id}: ${cmd}"
      # Launch and capture PID
      set +e
      ${cmd} > "${log}" 2>&1 &
      pid=$!
      set -e

      if kill -0 "$pid" 2>/dev/null; then
        PID2GPU["$pid"]="$gpu_id"
        SLOTS["$gpu_id"]=$(( SLOTS["$gpu_id"] + 1 ))
        echo "[$(date '+%F %T')] Started PID ${pid} on GPU ${gpu_id} (slots ${SLOTS[$gpu_id]}/$MAX_SLOTS_PER_GPU). Log: ${log}"
        i=$(( i + 1 ))
        idle_ticks=0
      else
        echo "[$(date '+%F %T')] Failed to start job for ${d}/${r}. See log: ${log}"
        # Do not advance i; retry on next loop
      fi
      continue
    fi
  fi

  # No scheduling happened this loop: sleep & show periodic status
  idle_ticks=$(( idle_ticks + 1 ))
  if (( idle_ticks % IDLE_CYCLES_BEFORE_STATUS == 0 )); then
    print_status
  fi
  sleep "$SLEEP_SECS"
done

echo "All jobs completed."
