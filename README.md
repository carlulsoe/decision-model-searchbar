# decision-searchbar

Type what you want to do ("how to start my terminal") and get your installed apps ranked by
[LiquidAI d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M), used text-only. A standalone GTK launcher
for GNOME (or any Linux desktop with GTK 4 and libadwaita); it leaves the normal overview search alone.

```sh
uv run decision-searchbar                      # launcher window: results 300 ms after you stop typing, Enter opens
uv run decision-searchbar rank how to start my terminal
```

## Setup

You need:

- **Python 3.14+ with PyGObject, GTK 4 and libadwaita from your distribution** (Fedora:
  `python3-gobject gtk4 libadwaita`). PyGObject comes from the system Python, so create the venv with
  `uv venv --python /usr/bin/python3 --system-site-packages`, then `uv sync`.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`** on your `PATH`, recent enough to serve
  d1 models at `/v1/systemone` (tested with release b11559; use a CUDA build for an NVIDIA GPU). On the first start
  it downloads `LiquidAI/d1-omni-600M-GGUF` Q8_0 (0.41 GB) into the Hugging Face cache, without the `mmproj` file.

The app starts its own `llama-server` on a free local port and stops it when it exits.

Settings are environment variables (see `src/decision_searchbar/__init__.py`):

| Variable | Default |
|---|---|
| `DECISION_SEARCHBAR_BACKEND` | `llama`; or `torch`, `onnx` (below) |
| `DECISION_SEARCHBAR_DEVICE` | `cuda` if `nvidia-smi` exists, else `cpu` |
| `DECISION_SEARCHBAR_LLAMA` | `llama-server` |
| `DECISION_SEARCHBAR_GGUF` | unset: the Hugging Face GGUF above; or a local `.gguf` path |
| `DECISION_SEARCHBAR_MODEL` | required for `torch` and `onnx`: a text-only checkpoint dir |
| `DECISION_SEARCHBAR_ONNX` | required for `onnx`: the ONNX text graph |

To bind it to a key, make a GNOME custom shortcut that runs `uv run --project /path/to/this/repo decision-searchbar`.

## How it works

- **Apps**: every `Gio.AppInfo` that `should_show()`, the same set GNOME's overview search shows. Each app is
  offered to the model as `name: generic name. description. keywords` (cut to d1's 24 tokens per option).
- **Ranking** (`rank.py`): d1 `choice` questions in a tournament. The apps are split into groups of at most 8, one
  choice question per group, and each group's winner goes to a final choice question averaged over 4 shuffled
  option orders. One choice over all ~90 apps at once was worse and depended heavily on option order (see below).
- **Text only**: d1-omni's text calls never touch its vision (94M) or audio (112M) encoders. The default backend
  runs LiquidAI's GGUF without its `mmproj` file; the torch backend runs a stripped checkpoint.

## Backends

| Backend | Weights | GPU (RTX 3060 Ti) | CPU (24 threads) |
|---|---|---|---|
| `llama` (default) | `LiquidAI/d1-omni-600M-GGUF` Q8_0, 0.41 GB | ~150 ms | 4.2 s |
| `torch` | stripped checkpoint, fp16 (fp32 on CPU) | ~120 ms | 5.1 s |
| `onnx` | `onnx-community/d1-omni-600M-ONNX` q8 / q4 | 216 / 240 ms | 5.6 / 3.5 s |

All give the same accuracy within noise. The CPU is too slow for interactive use.

The `torch` and `onnx` backends are optional extras (`uv sync --extra torch`, `--extra onnx`). Both read the
config and tokenizer from a text-only checkpoint, made from a download of `LiquidAI/d1-omni-600M`:

```sh
uvx --from huggingface_hub hf download LiquidAI/d1-omni-600M --local-dir d1-omni-600M
uv run --extra torch decision-searchbar strip d1-omni-600M d1-text-380M   # trunk + head only, fp32, 1.5 GB
uv run --extra torch --with "transformers>=5.15" python scripts/parity.py d1-omni-600M d1-text-380M  # matches
```

## Benchmark

`bench/cases.json`: 50 requests with the apps that count as right. They name the apps installed on the author's
machine, so edit them for yours. `bench/run.py` reports top-1, top-3, MRR and latency, and with `--shuffles N`
reruns with the app list shuffled.

```sh
uv run python bench/run.py --configs tournament:4,choice --shuffles 2 --baseline --misses
```

| Method | Top-1 | Top-3 |
|---|---|---|
| keyword overlap (baseline) | 42% | 60% |
| one choice over all apps | 38–62% (order-dependent) | 60–80% |
| one choice, averaged over 8 orders | 44–58% | 72–80% |
| one yes/no (`noul`) question per app | 32–40% | 54–56% |
| **tournament, groups of 8, final over 4 orders** (default, ~120 ms) | **82–86%** | **92–94%** |
| Bradley-Terry over 4 random groups-of-8 rounds (~310 ms) | 74–82% | 90–94% |
| Bradley-Terry, 2 rounds then 4 among the top 16 (~220 ms) | 86–88% | 94% |
| Bradley-Terry, 3 rounds then 4 among the top 16 (~300 ms) | 84–92% | 94–96% |

With 50 cases, one case is 2 points, so differences of a few points are noise. Latencies are torch fp16 on the GPU;
the llama backend adds roughly 30%. The Bradley-Terry variants (one strength per app fitted to all the group
choices) were not worth twice the latency and are no longer in the code. Most remaining misses are apps with poor
metadata: GNOME Settings has no description, so "connect to wifi" doesn't find it.

## License

MIT (`LICENSE`), except `src/decision_searchbar/d1/`, which is vendored from LiquidAI/d1-omni-600M under the
[LFM Open License v1.0](src/decision_searchbar/d1/LICENSE). The model weights are under the same LFM license,
which does not permit commercial use by organisations with $10M or more in annual revenue.
