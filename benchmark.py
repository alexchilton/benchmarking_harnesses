#!/usr/bin/env python3
"""
Quick benchmark: vLLM vs SGLang vs TurboQuant
Run against whichever server is currently on port 8010.
Usage: python3 benchmark.py [--label vllm|sglang|turboquant] [--kv-capacity]

Measures latency + tokens/sec AND GPU memory (peak VRAM), so memory-savers
like TurboQuant (3-bit weights / 4-bit KV cache) are actually captured.
"""
import argparse
import json
import time
import statistics
import threading
import subprocess
import urllib.request

BASE_URL = "http://localhost:8010/v1"

TESTS = {
    "basic_short": {
        "description": "Short factual answer",
        "messages": [{"role": "user", "content": "What is the capital of France? One word answer."}],
        "max_tokens": 10,
        "runs": 5,
    },
    "basic_long": {
        "description": "Longer generation",
        "messages": [{"role": "user", "content": "Explain how transformers work in neural networks. Be concise but complete."}],
        "max_tokens": 300,
        "runs": 3,
    },
    "rag_style": {
        "description": "RAG-style: digest context + answer",
        "messages": [{"role": "user", "content": (
            "Context: The Eiffel Tower was built between 1887 and 1889 as the entrance arch for the 1889 World's Fair. "
            "It was designed by Gustave Eiffel's engineering company and stands 330 metres tall. "
            "It was initially criticised by some of France's leading artists and intellectuals but has since become "
            "a global cultural icon of France. It receives about 7 million visitors per year.\n\n"
            "Question: Based only on the context above, when was the Eiffel Tower built and how tall is it?"
        )}],
        "max_tokens": 100,
        "runs": 3,
    },
    "tool_call": {
        "description": "Tool/function calling",
        "messages": [{"role": "user", "content": "What's the weather in London? Use the get_weather tool."}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["city"]
                }
            }
        }],
        "max_tokens": 100,
        "runs": 3,
    },
    "multi_turn": {
        "description": "Multi-turn conversation (agent-style)",
        "messages": [
            {"role": "user", "content": "I need to plan a trip to Japan for 5 days."},
            {"role": "assistant", "content": "I'd be happy to help plan your 5-day Japan trip! What cities are you interested in visiting?"},
            {"role": "user", "content": "Tokyo and Kyoto. What are the must-see spots in each?"},
        ],
        "max_tokens": 300,
        "runs": 3,
    },
}


def gpu_mem_used_mib():
    """Total GPU memory currently used (MiB). Only one engine runs on the GPU at a time."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return int(out.decode().splitlines()[0].strip())
    except Exception:
        return -1


class MemSampler:
    """Polls GPU VRAM in a background thread, tracking the peak."""
    def __init__(self, interval=0.1):
        self.interval = interval
        self._stop = threading.Event()
        self.peak = 0
        self._t = None

    def _run(self):
        while not self._stop.is_set():
            m = gpu_mem_used_mib()
            if m > self.peak:
                self.peak = m
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = gpu_mem_used_mib()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(timeout=1)


def call_api(messages, max_tokens, tools=None):
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if tools:
        payload["tools"] = tools

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    elapsed = time.perf_counter() - t0

    usage = result.get("usage", {})
    output_tokens = usage.get("completion_tokens", 0)
    return elapsed, output_tokens


def check_server():
    try:
        req = urllib.request.Request(f"{BASE_URL}/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = data.get("data", [])
            return models[0]["id"] if models else "unknown"
    except Exception:
        return None


def run_benchmarks(label):
    print(f"\n{'=' * 64}")
    print(f"Benchmark: {label.upper()}")
    print(f"{'=' * 64}")

    model_id = check_server()
    if model_id is None:
        print("ERROR: No server responding on port 8010. Is it started?")
        return {}

    steady_vram = gpu_mem_used_mib()
    print(f"Model: {model_id}")
    print(f"Steady-state VRAM (model loaded, idle): {steady_vram} MiB\n")
    results = {"_steady_state_vram_mib": steady_vram}
    overall_peak = steady_vram

    for test_name, test in TESTS.items():
        print(f"  [{test_name}] {test['description']}")
        latencies = []
        tokens_per_sec = []
        test_peak = 0

        for i in range(test["runs"]):
            try:
                with MemSampler() as ms:
                    elapsed, out_tokens = call_api(
                        test["messages"],
                        test["max_tokens"],
                        test.get("tools"),
                    )
                test_peak = max(test_peak, ms.peak)
                latencies.append(elapsed)
                if out_tokens > 0:
                    tokens_per_sec.append(out_tokens / elapsed)
                print(f"    run {i + 1}: {elapsed:.2f}s, {out_tokens} tokens", end="")
                if out_tokens > 0:
                    print(f" ({out_tokens / elapsed:.1f} tok/s)", end="")
                print()
            except Exception as e:
                print(f"    run {i + 1}: ERROR - {e}")

        overall_peak = max(overall_peak, test_peak)
        if latencies:
            avg = statistics.mean(latencies)
            med = statistics.median(latencies)
            avg_tps = statistics.mean(tokens_per_sec) if tokens_per_sec else 0
            print(f"    -> avg: {avg:.2f}s | median: {med:.2f}s | avg tok/s: {avg_tps:.1f} | peak VRAM: {test_peak} MiB")
            results[test_name] = {
                "avg_latency": avg,
                "median_latency": med,
                "avg_tok_per_sec": avg_tps,
                "peak_vram_mib": test_peak,
            }

    results["_overall_peak_vram_mib"] = overall_peak
    print(f"\n  Overall peak VRAM: {overall_peak} MiB (steady-state {steady_vram} MiB)")
    print("  NOTE: engines launched with mem-fraction/gpu-mem-util 0.85 preallocate a KV pool,")
    print("  so steady-state VRAM is dominated by that pool. For weight/KV-quant savings the")
    print("  meaningful test is max-context-before-OOM at fixed util (run with --kv-capacity).")
    return results


def kv_capacity_probe():
    """Find the largest single-prompt token budget the server accepts before erroring.
    A KV-cache compressor (e.g. TurboQuant 4-bit KV) should fit a larger budget."""
    print(f"\n{'-' * 64}\nKV capacity probe (largest max_tokens before failure)\n{'-' * 64}")
    lo, hi, best = 256, 32768, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            call_api([{"role": "user", "content": "Count slowly: 1, 2, 3,"}], mid)
            best = mid
            lo = mid + 1
            print(f"    max_tokens={mid}: OK")
        except Exception as e:
            print(f"    max_tokens={mid}: fail ({str(e)[:50]})")
            hi = mid - 1
    print(f"  -> largest accepted max_tokens: {best}")
    return best


def save_results(label, results):
    fname = f"bench_{label}_{int(time.time())}.json"
    with open(fname, "w") as f:
        json.dump({"label": label, "results": results}, f, indent=2)
    print(f"\nResults saved to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="unknown", help="Label for this run: vllm, sglang, or turboquant")
    parser.add_argument("--kv-capacity", action="store_true", help="Also run the KV-cache capacity probe")
    args = parser.parse_args()

    results = run_benchmarks(args.label)
    if results and args.kv_capacity:
        results["_kv_capacity_max_tokens"] = kv_capacity_probe()
    if results:
        save_results(args.label, results)
