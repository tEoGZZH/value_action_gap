#!/usr/bin/env bash
#SBATCH -J task2_value_action_gap
#SBATCH -o /home/zs473554/jobscripts/value_action_gap/log/llama3base_task2_vllm%j.txt
#SBATCH -t 03:00:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -p c23g
#SBATCH --gres=gpu:1
#SBATCH -A rwth1809

set -euo pipefail

PROJECT_DIR=/home/zs473554/projects/value_action_gap
VENV_DIR=/work/zs473554/envs/value_action_gap/.venv311

# ====== model config ======
# MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
MODEL_ID="/hpcwork/zs473554/checkpoints/open-instruct/llama3base/mixing/three_seven/"

OUT_DIR=${PROJECT_DIR}/results/task2
LOG_DIR=${OUT_DIR}/log
mkdir -p "${LOG_DIR}"

# ====== env + venv first ======
cd ${PROJECT_DIR}/src
source ${VENV_DIR}/bin/activate


# ====== start vLLM server in background ======
PORT=8000
MODEL_NAME="llama3"

export VLLM_LOGGING_LEVEL=DEBUG

python -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_ID} \
  --served-model-name llama3 \
  --host 127.0.0.1 \
  --port ${PORT} \
  --structured-outputs-config.backend=xgrammar \
  --structured-outputs-config.disable_any_whitespace=true \
  --structured-outputs-config.disable_fallback=true \
  > ${OUT_DIR}/log/vllm_server_${SLURM_JOB_ID}.log 2>&1 &

VLLM_PID=$!

echo "vLLM PID=${VLLM_PID}"
sleep 2

echo "===== PROCESS CHECK ====="
ps -fp ${VLLM_PID} || echo "vLLM process exited"

echo "===== PORT LISTEN CHECK ====="
ss -ltnp | grep 8000 || echo "8000 not listening yet"

echo "===== GPU CHECK ====="
nvidia-smi || true

echo "===== vLLM LOG TAIL (early) ====="
tail -n 60 ${OUT_DIR}/vllm_server_${SLURM_JOB_ID}.log || true


cleanup() { kill ${VLLM_PID} >/dev/null 2>&1 || true; }
trap cleanup EXIT

# ====== Ready Check =========
echo "===== READY CHECK ====="

READY_TIMEOUT=300   # Wait at most 5 minutes
READY_INTERVAL=2

server_url="http://127.0.0.1:${PORT}"

# helper: print last logs + process status on failure
fail_dump() {
  echo "----- vLLM process status -----"
  ps -fp "${VLLM_PID}" || true
  echo "----- port listen -----"
  ss -ltnp | grep ":${PORT}\b" || true
  echo "----- vLLM log tail -----"
  tail -n 200 "${OUT_DIR}/vllm_server_${SLURM_JOB_ID}.log" || true
}

start_ts=$(date +%s)

# 1 wait until /v1/models returns 200
while true; do
  now=$(date +%s)
  if (( now - start_ts > READY_TIMEOUT )); then
    echo "ERROR: vLLM not ready after ${READY_TIMEOUT}s (models endpoint)."
    fail_dump
    exit 1
  fi

  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "ERROR: vLLM process exited before ready."
    fail_dump
    exit 1
  fi

  if command -v curl >/dev/null 2>&1; then
    code=$(curl -s -o /dev/null -w "%{http_code}" "${server_url}/v1/models" || true)
    if [[ "${code}" == "200" ]]; then
      echo "OK: /v1/models reachable."
      break
    fi
  else
    # fallback: python http check
    python - <<PY >/dev/null 2>&1 && { echo "OK: /v1/models reachable."; break; }
import urllib.request
urllib.request.urlopen("${server_url}/v1/models", timeout=2).read()
PY
  fi

  sleep "${READY_INTERVAL}"
done

# 2 do a tiny inference to ensure model is loaded
echo "===== SMOKE TEST: chat completion ====="
if command -v curl >/dev/null 2>&1; then
  curl -s "${server_url}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "llama3",
      "messages": [{"role":"user","content":"Reply with exactly: READY"}],
      "max_tokens": 3,
      "temperature": 0
    }' | head -c 400 || true
  echo
else
  python - <<PY
import json, urllib.request
data = json.dumps({
  "model":"llama3",
  "messages":[{"role":"user","content":"Reply with exactly: READY"}],
  "max_tokens":3,
  "temperature":0
}).encode()
req = urllib.request.Request(
  "${server_url}/v1/chat/completions",
  data=data,
  headers={"Content-Type":"application/json"},
  method="POST",
)
print(urllib.request.urlopen(req, timeout=30).read()[:400])
PY
fi

echo "===== vLLM READY ====="


# ====== client env for hf_backend ======
export VLLM_BASE_URL="http://127.0.0.1:${PORT}"
export VLLM_MODEL="${MODEL_NAME}"
export VLLM_TIMEOUT_S="300"

# ====== run eval (this should NOT load transformers model if hf_backend routes correctly) ======
srun --export=ALL  python -u -m tasks.task2.eval \
  --model ${MODEL_ID} \
  --out_csv ${OUT_DIR}/llama3.1-8B-base-3_7_results.csv
