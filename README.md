# floudsonnx

Ollama-style ONNX model store and runtime — pull, cache, and load ORT sessions.

Behaves like Ollama but for ONNX models: pull models from HuggingFace, cache them
locally under `~/.flouds/`, load them into `onnxruntime.InferenceSession` or
`ORTModelForSeq2SeqLM`, and serve them to any downstream service with a single call.

## Install

```bash
pip install floudsonnx                  # core — load pre-exported models
pip install floudsonnx[export]          # + auto-export via flouds-model-exporter
pip install floudsonnx[seq2seq]         # + ORTModelForSeq2SeqLM (s2s / llm)
pip install floudsonnx[server]          # + optional FastAPI HTTP server
pip install floudsonnx[all]             # everything
```

> **Note:** Do not install `sentence-transformers`. Always pass
> `library="transformers"` when pulling `fe` / `sc` / `ranker` models.

## Quick start

```python
from floudsonnx import create_model
import numpy as np

# Pull (auto-export if missing) + load session
model = create_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    model_for="fe",
    library="transformers",
    normalize_embeddings=True,
)

# Tokenize
enc = model.tokenizer(["Hello world"], return_tensors="np", padding=True, truncation=True)

# Build input feed from session's expected inputs (handles token_type_ids automatically)
inputs = {i.name for i in model.session.get_inputs()}
feed = {n: enc[n].astype(np.int64) for n in inputs if n in enc}

# Run
outputs = model.run(None, feed)
print(outputs[0].shape)  # (1, seq_len, 384)
```

## Model types

| `model_for` | Task | Session type | Example model |
|---|---|---|---|
| `fe` | feature-extraction | `ort.InferenceSession` | `sentence-transformers/all-MiniLM-L6-v2` |
| `sc` | text-classification | `ort.InferenceSession` | `distilbert-base-uncased-finetuned-sst-2-english` |
| `ranker` | sequence-classification | `ort.InferenceSession` | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| `s2s` | seq2seq-lm | `ORTModelForSeq2SeqLM` | `google/flan-t5-small` |
| `llm` | text-generation-with-past | `ORTModelForSeq2SeqLM` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

## CLI

```bash
# Pull a model (auto-exports if not on disk)
floudsonnx pull sentence-transformers/all-MiniLM-L6-v2 \
    --for fe --task feature-extraction --library transformers --normalize-embeddings

# List locally stored models
floudsonnx list

# Inspect manifest
floudsonnx info sentence-transformers/all-MiniLM-L6-v2

# Remove from store
floudsonnx remove sentence-transformers/all-MiniLM-L6-v2

# Hot-reload session from disk
floudsonnx reload sentence-transformers/all-MiniLM-L6-v2

# Session cache stats
floudsonnx stats

# Start optional HTTP server
floudsonnx serve --host 0.0.0.0 --port 19720
```

## Python API

```python
from floudsonnx import FloudsOnnxClient, FloudsOnnxSettings

# Custom settings
client = FloudsOnnxClient(FloudsOnnxSettings(
    home_dir="/data/flouds",
    session_provider="CPUExecutionProvider",
))

# pull — export only (no session load)
manifest = client.pull("BAAI/bge-base-en-v1.5", model_for="fe",
                        library="transformers", normalize_embeddings=True)

# create_model — pull + load (idempotent)
model = client.create_model("BAAI/bge-base-en-v1.5", model_for="fe")

# load_model — load only (must already be on disk)
model = client.load_model("BAAI/bge-base-en-v1.5", model_for="fe")

# reload — hot-reload from disk without restarting
model = client.reload("BAAI/bge-base-en-v1.5", model_for="fe")

# unload — evict from RAM, keep files on disk
client.unload("BAAI/bge-base-en-v1.5", model_for="fe")

# remove — delete from disk + evict from RAM
client.remove("BAAI/bge-base-en-v1.5", model_for="fe")

# list
for m in client.list():
    print(m.model_name, m.model_for, m.session_strategy)

# S2S / LLM
s2s = client.create_model("google/flan-t5-small", model_for="s2s")
out = s2s.seq2seq_model.generate(input_ids=enc["input_ids"], max_new_tokens=64)
```

## Environment variables (`FLOUDSONNX_` prefix)

| Variable | Default | Description |
|---|---|---|
| `HOME_DIR` | `~/.flouds` | Model store root |
| `ONNX_PATH` | `<home>/models` | Override models directory |
| `SESSION_PROVIDER` | `CPUExecutionProvider` | ORT execution provider |
| `ENCODER_CACHE_MAX` | `5` | Max cached encoder sessions |
| `DECODER_CACHE_MAX` | `5` | Max cached decoder sessions |
| `SEQ2SEQ_CACHE_MAX` | `3` | Max cached seq2seq models |
| `EXPORT_OPTIMIZE` | `false` | Enable ONNX graph optimization |
| `EXPORT_OPSET` | *(auto)* | Force specific ONNX opset |
| `EXPORT_DEVICE` | `cpu` | Export device |
| `EXPORT_LIBRARY` | *(auto)* | Force exporter library (e.g. `transformers`) |
| `EXPORT_NORMALIZE_EMBEDDINGS` | `false` | Normalize embeddings on export |
| `EXPORT_USE_SUBPROCESS` | `false` | Run export in subprocess |
| `EXPORT_USE_FALLBACK_IF_FAILED` | `false` | Try legacy fallback on export failure |
| `EXPORT_MERGE` | `false` | Merge encoder/decoder into single file |
| `EXPORT_HF_TOKEN` | *(none)* | HuggingFace API token |
| `SERVER_HOST` | `127.0.0.1` | HTTP server host |
| `SERVER_PORT` | `19720` | HTTP server port |

## Local store layout

```
~/.flouds/
└── models/
    ├── fe/
    │   └── all-MiniLM-L6-v2/
    │       ├── model.onnx
    │       ├── model_optimized.onnx
    │       ├── tokenizer.json
    │       └── manifest.json
    ├── s2s/
    ├── sc/
    ├── ranker/
    └── llm/
```

## Development

```bash
pip install -r requirements-dev.txt
pip install -e ".[export,seq2seq]"
pre-commit install
pytest tests/unit/ -v
```

## Manual tests

```bash
python manual_test.py --normalize-embeddings          # fe model
python test_all_models.py --type fe s2s ranker        # multiple types
python test_all_models.py --type llm                  # LLM (~2GB download)
python test_all_models.py --no-export                 # load existing only
```

## License

Apache-2.0 © 2026 Goutam Malakar
