# 🔮 Transformer Explainer

**See data flow through a real transformer, step by step.**

Most explanations of transformers are static diagrams. This is different — it's an interactive tool that runs a *real* (tiny) transformer and lets you inspect every intermediate value: embeddings, attention weights, feed-forward activations, and predictions.

![Screenshot placeholder](screenshot.png)

## Why?

Understanding transformers by reading papers and blog posts is hard. Understanding them by watching actual data flow through actual matrix multiplications? That clicks.

## Quick Start

```bash
pip install -r requirements.txt
python run.py
```

Open [http://localhost:8070](http://localhost:8070) in your browser.

## What You'll See

- **Type any text** and watch it get tokenized, embedded, and processed
- **Click any component** in the flow diagram to inspect its internals
- **Attention heatmaps** showing which characters attend to which
- **Embedding visualizations** showing how text becomes vectors
- **Feed-forward activations** showing the "thinking" layer
- **Next-token predictions** with probability distributions
- **Auto-regressive generation** — watch the model write text character by character

## Architecture

```
transformer/
  model.py       # Transformer from scratch (NumPy only — no PyTorch!)
  trainer.py     # Training loop + weight save/load
api.py           # FastAPI backend
static/          # Vanilla JS frontend
  index.html
  app.js
  style.css
weights/         # Pre-trained tiny model
  model.json
```

### The Model

- **Pure NumPy** — every operation is transparent, no black-box frameworks
- Character-level tokenizer
- 64-dimensional embeddings, 2 attention heads, 2 transformer blocks
- ~50K parameters — runs instantly
- Pre-trained on simple English sentences

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/process` | Run text through transformer, return all intermediate states |
| `POST /api/generate` | Auto-regressive generation with step-by-step states |
| `GET /api/model-info` | Model architecture info |
| `GET /api/vocab` | Vocabulary mapping |

## What You'll Learn

1. How text becomes numbers (tokenization)
2. How numbers become vectors (embeddings)
3. Why position matters (positional encoding)
4. How attention works — really (Q, K, V matrices, scaled dot-product)
5. What multi-head attention looks like in practice
6. How residual connections preserve information
7. What feed-forward layers actually compute
8. How softmax produces predictions
9. How auto-regressive generation works step by step

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — Harvard NLP
- [3Blue1Brown: Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)

## License

MIT
