"""
TurboQuant+vLLM launcher.
Applies TurboQuant weight and KV cache patches before starting the vLLM server.
Patches are applied at IMPORT time so that vLLM spawn-workers (which re-import this
module) also get them; the server launch itself is guarded by __main__.
"""
import sys


def _force_uva_available():
    """WSL fix: vLLM disables UVA via a blanket 'pin_memory=False on WSL' heuristic,
    so is_uva_available() returns False and the V2 model runner (which TurboQuant
    requires) refuses to allocate its staged-write buffers. But UVA / mapped pinned
    memory actually works on this WSL2 (verified: host write is visible on the device
    view). Override the gate so the V2 runner can initialize. Runs at module import
    in both the API server process and the spawned EngineCore worker.
    """
    try:
        import vllm.utils.platform_utils as _pu
        try:
            _pu.is_uva_available.cache_clear()
        except Exception:
            pass
        _pu.is_uva_available = lambda: True
        # Patch any module that already imported the symbol by value.
        for _mod in list(sys.modules.values()):
            if getattr(_mod, "is_uva_available", None) is not None:
                try:
                    _mod.is_uva_available = lambda: True
                except Exception:
                    pass
        print("WSL UVA gate overridden (is_uva_available -> True).")
    except Exception as e:
        print(f"WARN: could not override UVA gate: {e}")


_force_uva_available()

import turboquant_vllm

print("Applying TurboQuant weight quantization (3-bit)...")
turboquant_vllm.enable_weight_quantization(bits=3, group_size=128)

print("Applying TurboQuant KV cache compression (K=4bit, V=4bit)...")
turboquant_vllm.patch_vllm_attention(k_bits=4, v_bits=4)

print("TurboQuant patches applied.")

if __name__ == "__main__":
    import asyncio
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.entrypoints.openai.api_server import run_server

    print("Starting vLLM server...")
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(sys.argv[1:])
    asyncio.run(run_server(args))
