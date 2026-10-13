"""Rank installed apps for a free-text request with a text-only d1 decision model.

Settings (environment):
    DECISION_SEARCHBAR_BACKEND   llama (default: GGUF via llama-server), torch, or onnx
    DECISION_SEARCHBAR_DEVICE    cuda (default when nvidia-smi exists) or cpu
    DECISION_SEARCHBAR_LLAMA     the llama-server binary (default: llama-server on PATH)
    DECISION_SEARCHBAR_GGUF      a local d1 GGUF (default: LiquidAI/d1-omni-600M-GGUF:Q8_0, fetched by llama-server)
    DECISION_SEARCHBAR_MODEL     text-only checkpoint dir from `strip` (torch weights; config and tokenizer for onnx)
    DECISION_SEARCHBAR_ONNX      ONNX text graph for the onnx backend
"""

import os
import shutil
import sys


def setting(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"DECISION_SEARCHBAR_{name}", default)


def required(name: str, backend: str) -> str:
    if value := setting(name):
        return value
    raise RuntimeError(f"the {backend} backend needs DECISION_SEARCHBAR_{name} (see the README)")


def open_model():
    """The configured model (for llama, a llama-server started for this thread); call `.ready()` before use."""
    from .model import LlamaD1, load

    backend = setting("BACKEND", "llama")
    device = setting("DEVICE", "cuda" if shutil.which("nvidia-smi") else "cpu")
    if backend == "llama":
        return LlamaD1.start(setting("LLAMA", "llama-server"), setting("GGUF"), gpu=device == "cuda")
    return load(required("MODEL", backend), backend, device,
                required("ONNX", backend) if backend == "onnx" else None)


def main() -> None:
    args = sys.argv[1:]
    if args[:1] == ["rank"]:
        import time

        from .apps import installed_apps
        from .rank import rank

        model, apps, query = open_model(), installed_apps(), " ".join(args[1:])
        model.ready()
        rank(model, "warm up", apps)
        start = time.perf_counter()
        ranked = rank(model, query, apps)
        print(f"{(time.perf_counter() - start) * 1000:.0f} ms on {model.device}")
        for app, score in ranked[:10]:
            print(f"{score:7.3f}  {app.name}  ({app.id})")
    elif args[:1] == ["strip"] and len(args) == 3:
        from .torch_backend import strip

        strip(args[1], args[2])
    elif not args:
        from .ui import run

        run()
    else:
        sys.exit("usage: decision-searchbar [rank QUERY... | strip OMNI_DIR TEXT_DIR]")
