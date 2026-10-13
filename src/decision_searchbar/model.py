"""Text-only d1: the shared trunk and decision head of d1-omni-600M, without the vision and audio encoders.

    model = load(path, backend="torch", device="cuda")  # or backend="onnx"
    model.probabilities(state, [question, ...])  # each question's distribution over its options

Text calls in d1-omni never touch the media encoders, so dropping them changes no text answer. Backends only turn
token ids and option-marker positions into option logits; prompting and calibration are shared here.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from .d1.prompt import QTYPES, as_question, encode, temperature_key


class Tok:
    """The three tokenizer calls `prompt.encode` makes, on a bare `tokenizers.Tokenizer`."""

    def __init__(self, path: Path, bos: str):
        self.t = Tokenizer.from_file(str(path))
        self.bos_token_id = self.t.token_to_id(bos)

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": self.t.encode(text, add_special_tokens=add_special_tokens).ids}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.t.token_to_id(token)


class D1:
    def __init__(self, path: Path):
        self.config = json.loads((path / "config.json").read_text())
        bos = json.loads((path / "tokenizer_config.json").read_text())["bos_token"]
        self.tok = Tok(path / "tokenizer.json", bos)

    def probabilities(self, state: Any, questions: Sequence, per_option: int = 24) -> list[list[float]]:
        """`per_option`: option-text tokens per option (d1's default is 24)."""
        questions = [as_question(q) for q in questions]
        rows = [encode(self.tok, "" if state is None else state, q, self.config["max_length"], per_option=per_option)
                for q in questions]
        length, k = max(len(ids) for ids, _ in rows), max(len(m) for _, m in rows)
        batch = {name: np.zeros((len(rows), n), np.int64) for name, n in
                 (("input_ids", length), ("attention_mask", length), ("marker_pos", k), ("marker_mask", k))}
        for i, (ids, markers) in enumerate(rows):
            batch["input_ids"][i, :len(ids)], batch["attention_mask"][i, :len(ids)] = ids, 1
            batch["marker_pos"][i, :len(markers)], batch["marker_mask"][i, :len(markers)] = markers, 1
        batch["qtype"] = np.array([QTYPES[q.type] for q in questions], np.int64)
        temps, out = self.config["temperatures"], []
        for q, z in zip(questions, self.logits(batch)):
            z = z[:q.options].astype(np.float64) / temps.get(temperature_key(q), temps.get(q.type, 1.0))
            p = np.exp(z - z.max())
            p = (p / p.sum()).tolist()
            out.append(p[::-1] if q.type == "noul" else p)  # the model reads a noul as [false, true]
        return out

    def ready(self) -> None:
        pass

    def logits(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class OnnxD1(D1):
    """onnx-community/d1-omni-600M-ONNX's text graph (`model_quantized.onnx`: 8-bit, `model_q4.onnx`: 4-bit)."""

    def __init__(self, path: Path, onnx_file: Path, device: str = "cpu"):
        super().__init__(path)
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_file), options, providers=providers)
        self.device = self.session.get_providers()[0]

    def logits(self, batch):
        return self.session.run(["logits"], {**batch, "prefix_len": np.zeros(len(batch["qtype"]), np.int64)})[0]


PR_SET_PDEATHSIG = 1
GGUF = "LiquidAI/d1-omni-600M-GGUF:Q8_0"


class LlamaD1:
    """A llama.cpp `llama-server` running a d1 GGUF (without its mmproj, so text only), via `/v1/systemone`.

    The server does its own prompting and calibration; `per_option` is d1's default of 24 there.
    """

    def __init__(self, url: str, process=None):
        import httpx

        self.url = url.rstrip("/") + "/v1/systemone"
        self.http = httpx.Client(timeout=30)
        self.device, self.process = url, process

    @classmethod
    def start(cls, server: str | Path, gguf: str | Path | None = None, gpu: bool = True) -> LlamaD1:
        """Spawn `llama-server` on a free local port; it dies with the spawning thread (so spawn from the main one).
        Without `gguf`, the server fetches `GGUF` from Hugging Face (cached after the first run).
        Call `ready()` before the first question."""
        import atexit
        import ctypes
        import signal
        import socket
        import subprocess

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        # a whole question is read in one batch, so -b/-ub must hold the longest (the final round, ~2k tokens)
        args = [str(server), *(["-m", str(gguf)] if gguf else ["-hf", GGUF]), "--no-mmproj",
                "--host", "127.0.0.1", "--port", str(port), "-b", "8192", "-ub", "8192", "-c", "65536", "-np", "4", "-ngl", "99" if gpu else "0", *([] if gpu else ["--device", "none"])]
        libc = ctypes.CDLL(None, use_errno=True)
        process = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   preexec_fn=lambda: libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM))
        atexit.register(process.terminate)
        return cls(f"http://127.0.0.1:{port}", process)

    def ready(self, timeout: float = 600) -> None:  # the first run downloads the GGUF
        import time

        import httpx

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.process.returncode}")
            try:
                if self.http.get(self.device + "/health", timeout=1).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(0.1)
        raise RuntimeError(f"llama-server not healthy after {timeout:.0f} s")

    def probabilities(self, state: Any, questions: Sequence, per_option: int = 24) -> list[list[float]]:
        named = {f"q{i}": q for i, q in enumerate(questions)}
        r = self.http.post(self.url, json={"state": state, "questions": named})
        r.raise_for_status()
        answers = r.json()["answers"]
        out = []
        for name, q in named.items():
            a = answers[name]
            out.append([a["noul"], 1 - a["noul"]] if q["type"] == "noul" else
                       [a["probabilities"][k] for k in (q["criteria"] if q["type"] == "choice" else
                                                        map(str, range(len(q["criteria"]))))])
        return out


def load(path: str | Path, backend: str = "torch", device: str = "cpu", onnx_file: str | Path | None = None,
         url: str = "http://127.0.0.1:8091"):
    """`path`: a text-only checkpoint dir (config, tokenizer, and for torch the weights); `url` for llama."""
    if backend == "llama":
        return LlamaD1(url)
    path = Path(path)
    if backend == "torch":
        from .torch_backend import TorchD1

        return TorchD1(path, device)
    if backend == "onnx":
        return OnnxD1(path, Path(onnx_file), device)
    raise ValueError(f"unknown backend {backend!r}")
