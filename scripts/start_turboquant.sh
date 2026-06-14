#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.6; export PATH=$HOME/turboquant_env/bin:$CUDA_HOME/bin:/usr/local/bin:$PATH
MODEL=/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
PORT=8010
echo "Starting vLLM+TurboQuant (weight quant + KV cache compression) on port $PORT..."
~/turboquant_env/bin/python3 ~/tq_launcher.py \
    --model $MODEL \
    --port $PORT \
    --host 0.0.0.0 \
    --served-model-name default --enable-auto-tool-choice --tool-call-parser hermes --enforce-eager --gpu-memory-utilization 0.85 \
    --max-model-len 8192