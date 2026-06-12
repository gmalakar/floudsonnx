# Examples

Quick-start scripts for each model type. All require `pip install floudsonnx[export]`.

| Script | Model type | Model |
|---|---|---|
| `test_fe.py` | Feature extraction | `sentence-transformers/all-MiniLM-L6-v2` |
| `test_s2s.py` | Seq2seq | `google/flan-t5-small` |
| `test_ranker.py` | Cross-encoder reranker | `cross-encoder/ms-marco-MiniLM-L-12-v2` |
| `test_llm.py` | LLM generation | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

## Usage

```bash
python examples/test_fe.py
python examples/test_s2s.py
python examples/test_ranker.py
python examples/test_llm.py    # ~2GB download
```

> Always use `library="transformers"` for `fe/sc/ranker` models.
> Do **not** install `sentence-transformers`.
