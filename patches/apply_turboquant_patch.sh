#!/bin/bash
# Re-apply the vLLM-0.23 KV-layout fix to an installed turboquant-plus-vllm 0.13.7.
# Usage: bash apply_turboquant_patch.sh /path/to/turboquant_env
set -e
ENV=${1:?usage: apply_turboquant_patch.sh <venv path>}
F="$ENV/lib/python3.12/site-packages/turboquant_vllm/vllm_patch.py"
[ -f "$F" ] || { echo "not found: $F"; exit 1; }
cp "$F" "$F.bak"
sed -i "s/key_cache, _ = kv_cache.unbind(0)/key_cache, _ = kv_cache.unbind(1)/" "$F"
sed -i "s/key_cache, value_cache = kv_cache.unbind(0)/key_cache, value_cache = kv_cache.unbind(1)/" "$F"
echo "patched $F (backup at $F.bak)"
grep -n "unbind(1)" "$F"
