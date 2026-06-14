#!/bin/bash
MODEL=/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
PORT=8010
LOG=~/vllm_server.log

echo "[$(date)] Starting vLLM on port $PORT..." | tee $LOG

# Ensure cuda-13 runtime is on the linker path
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH

nohup ~/vllm_env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --port $PORT \
    --host 0.0.0.0 \
    --served-model-name default --enable-auto-tool-choice --tool-call-parser hermes --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    >> $LOG 2>&1 &

PID=$!
echo "[$(date)] vLLM launched with PID $PID" | tee -a $LOG
echo $PID > ~/vllm.pid
