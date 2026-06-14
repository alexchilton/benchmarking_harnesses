#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$HOME/sglang2_env/bin:$CUDA_HOME/bin:/usr/local/bin:$PATH
MODEL=/home/alex/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
PORT=8010
echo "Starting SGLang 0.5.2 on port $PORT..."
exec ~/sglang2_env/bin/python3 -m sglang.launch_server \
    --model-path $MODEL \
    --port $PORT --host 0.0.0.0 \
    --mem-fraction-static 0.85 \
    --context-length 8192 \
    --served-model-name default
