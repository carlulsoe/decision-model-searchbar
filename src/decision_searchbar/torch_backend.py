"""The PyTorch backend: the vendored trunk and head on a text-only checkpoint written by `strip`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .d1.encoder import DecisionHead, Trunk
from .model import D1

KEEP = ("encoder.", "head.")


class Net(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.encoder = Trunk(config["text_config"])
        self.head = DecisionHead(config["text_config"]["hidden_size"], config["head_layers"])


class TorchD1(D1):
    def __init__(self, path: Path, device: str = "cpu", dtype: torch.dtype | None = None):
        super().__init__(path)
        self.net = Net(self.config)
        weights = {k: v for k, v in load_file(path / "model.safetensors").items() if k.startswith(KEEP)}
        self.net.load_state_dict(weights, strict=True)
        dtype = dtype or (torch.float32 if device == "cpu" else torch.float16)  # the card advises against bf16
        self.net.to(device=device, dtype=dtype).eval()
        self.device = device

    @torch.no_grad()
    def logits(self, batch):
        t = {k: torch.from_numpy(v).to(self.device) for k, v in batch.items()}
        pad = t["attention_mask"].bool()
        enc = self.net.encoder
        h = enc(enc.embed_tokens(t["input_ids"]), pad, torch.zeros_like(t["qtype"]))
        z = self.net.head(h, pad, t["marker_pos"], t["marker_mask"].bool(), t["qtype"])
        if self.device == "cuda":
            torch.cuda.synchronize()
        return z.float().cpu().numpy()


def strip(src: str | Path, dst: str | Path) -> None:
    """Write a text-only checkpoint: trunk and head weights (kept in float32), config without media sections."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    weights = {k: v.contiguous() for k, v in load_file(src / "model.safetensors").items() if k.startswith(KEEP)}
    save_file(weights, dst / "model.safetensors")
    config = json.loads((src / "config.json").read_text())
    for key in ("vision_config", "audio_config", "image_text_length", "audio_text_length", "projector_hidden_size",
                "auto_map", "architectures"):
        config.pop(key, None)
    config["model_type"] = "d1_text"
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        shutil.copy(src / name, dst / name)
