"""
Minimal Transformer implemented from scratch with NumPy.
Every component exposes .forward() and .explain() for full transparency.
"""

import numpy as np
import json
import os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Character-level tokenizer with special tokens."""

    def __init__(self, vocab=None):
        if vocab is None:
            vocab = {}
        self.char2id = dict(vocab)
        self.id2char = {v: k for k, v in self.char2id.items()}

    def fit(self, text):
        chars = sorted(set(text))
        self.char2id = {"<pad>": 0, "<unk>": 1}
        for i, c in enumerate(chars, start=2):
            self.char2id[c] = i
        self.id2char = {v: k for k, v in self.char2id.items()}
        return self

    @property
    def vocab_size(self):
        return len(self.char2id)

    def encode(self, text):
        return [self.char2id.get(c, 1) for c in text]

    def decode(self, ids):
        return "".join(self.id2char.get(i, "?") for i in ids)

    def forward(self, text):
        return np.array(self.encode(text))

    def explain(self, text):
        ids = self.encode(text)
        return {
            "input_text": text,
            "tokens": list(text),
            "token_ids": ids,
            "vocab_size": self.vocab_size,
        }

    def to_dict(self):
        return {"char2id": self.char2id}

    @classmethod
    def from_dict(cls, d):
        return cls(vocab=d["char2id"])

# ---------------------------------------------------------------------------
# Embedding + Positional Encoding
# ---------------------------------------------------------------------------

class Embedding:
    def __init__(self, vocab_size, d_model, max_len=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        scale = np.sqrt(2.0 / d_model)
        self.token_emb = np.random.randn(vocab_size, d_model) * scale
        self.pe = self._sinusoidal_pe(max_len, d_model)

    @staticmethod
    def _sinusoidal_pe(max_len, d_model):
        pe = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe

    def forward(self, token_ids):
        seq_len = len(token_ids)
        tok = self.token_emb[token_ids]  # (seq, d)
        return tok + self.pe[:seq_len]

    def explain(self, token_ids):
        seq_len = len(token_ids)
        tok = self.token_emb[token_ids]
        pe = self.pe[:seq_len]
        combined = tok + pe
        return {
            "token_embeddings": tok.tolist(),
            "positional_encoding": pe.tolist(),
            "combined": combined.tolist(),
        }

    def to_dict(self):
        return {"token_emb": self.token_emb.tolist(), "d_model": self.d_model, "vocab_size": self.vocab_size, "max_len": self.max_len}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.vocab_size = d["vocab_size"]
        obj.d_model = d["d_model"]
        obj.max_len = d["max_len"]
        obj.token_emb = np.array(d["token_emb"])
        obj.pe = Embedding._sinusoidal_pe(obj.max_len, obj.d_model)
        return obj

# ---------------------------------------------------------------------------
# Layer Norm
# ---------------------------------------------------------------------------

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta

    def explain(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normed = self.gamma * (x - mean) / np.sqrt(var + self.eps) + self.beta
        return {"input": x.tolist(), "mean": mean.squeeze(-1).tolist(), "var": var.squeeze(-1).tolist(), "output": normed.tolist()}

    def to_dict(self):
        return {"d_model": self.d_model, "gamma": self.gamma.tolist(), "beta": self.beta.tolist()}

    @classmethod
    def from_dict(cls, d):
        obj = cls(d["d_model"])
        obj.gamma = np.array(d["gamma"])
        obj.beta = np.array(d["beta"])
        return obj

# ---------------------------------------------------------------------------
# Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    def __init__(self, d_model, n_heads):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

    def forward(self, x, mask=None):
        return self._compute(x, mask)[0]

    def _compute(self, x, mask=None):
        seq_len = x.shape[0]
        Q = x @ self.W_q  # (seq, d)
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape to (n_heads, seq, d_k)
        Q_h = Q.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        K_h = K.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)
        V_h = V.reshape(seq_len, self.n_heads, self.d_k).transpose(1, 0, 2)

        scores = Q_h @ K_h.transpose(0, 2, 1) / np.sqrt(self.d_k)  # (heads, seq, seq)
        if mask is not None:
            scores = scores + mask  # mask has -inf for blocked positions
        attn_weights = softmax(scores, axis=-1)
        attn_out = attn_weights @ V_h  # (heads, seq, d_k)

        # Concat and project
        concat = attn_out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        output = concat @ self.W_o
        return output, {"Q": Q, "K": K, "V": V, "Q_heads": Q_h, "K_heads": K_h, "V_heads": V_h,
                        "scores": scores, "attn_weights": attn_weights, "attn_out": attn_out, "output": output}

    def explain(self, x, mask=None):
        _, info = self._compute(x, mask)
        return {k: v.tolist() for k, v in info.items()}

    def to_dict(self):
        return {"d_model": self.d_model, "n_heads": self.n_heads,
                "W_q": self.W_q.tolist(), "W_k": self.W_k.tolist(),
                "W_v": self.W_v.tolist(), "W_o": self.W_o.tolist()}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.d_model = d["d_model"]
        obj.n_heads = d["n_heads"]
        obj.d_k = obj.d_model // obj.n_heads
        obj.W_q = np.array(d["W_q"])
        obj.W_k = np.array(d["W_k"])
        obj.W_v = np.array(d["W_v"])
        obj.W_o = np.array(d["W_o"])
        return obj

# ---------------------------------------------------------------------------
# Feed Forward
# ---------------------------------------------------------------------------

class FeedForward:
    def __init__(self, d_model, d_ff=None):
        self.d_model = d_model
        self.d_ff = d_ff or d_model * 4
        scale1 = np.sqrt(2.0 / d_model)
        scale2 = np.sqrt(2.0 / self.d_ff)
        self.W1 = np.random.randn(d_model, self.d_ff) * scale1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, d_model) * scale2
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        hidden = relu(x @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2

    def explain(self, x):
        hidden = relu(x @ self.W1 + self.b1)
        pre_act = x @ self.W1 + self.b1
        output = hidden @ self.W2 + self.b2
        return {"pre_activation": pre_act.tolist(), "post_activation": hidden.tolist(), "output": output.tolist()}

    def to_dict(self):
        return {"d_model": self.d_model, "d_ff": self.d_ff,
                "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist()}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.d_model = d["d_model"]
        obj.d_ff = d["d_ff"]
        obj.W1 = np.array(d["W1"])
        obj.b1 = np.array(d["b1"])
        obj.W2 = np.array(d["W2"])
        obj.b2 = np.array(d["b2"])
        return obj

# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff=None):
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out = self.attn.forward(x, mask)
        x2 = self.ln1.forward(x + attn_out)
        ff_out = self.ff.forward(x2)
        return self.ln2.forward(x2 + ff_out)

    def explain(self, x, mask=None):
        attn_out, attn_info = self.attn._compute(x, mask)
        residual1 = x + attn_out
        ln1_out = self.ln1.forward(residual1)
        ff_info = self.ff.explain(ln1_out)
        ff_out_arr = np.array(ff_info["output"])
        residual2 = ln1_out + ff_out_arr
        ln2_out = self.ln2.forward(residual2)
        return {
            "attention": {k: v.tolist() for k, v in attn_info.items()},
            "residual_after_attn": residual1.tolist(),
            "after_ln1": ln1_out.tolist(),
            "feed_forward": ff_info,
            "residual_after_ff": residual2.tolist(),
            "after_ln2": ln2_out.tolist(),
            "output": ln2_out.tolist(),
        }

    def to_dict(self):
        return {"attn": self.attn.to_dict(), "ln1": self.ln1.to_dict(),
                "ff": self.ff.to_dict(), "ln2": self.ln2.to_dict()}

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.attn = MultiHeadAttention.from_dict(d["attn"])
        obj.ln1 = LayerNorm.from_dict(d["ln1"])
        obj.ff = FeedForward.from_dict(d["ff"])
        obj.ln2 = LayerNorm.from_dict(d["ln2"])
        return obj

# ---------------------------------------------------------------------------
# Transformer (full model)
# ---------------------------------------------------------------------------

class Transformer:
    def __init__(self, vocab_size, d_model=64, n_heads=2, n_layers=2, d_ff=None, max_len=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or d_model * 4
        self.max_len = max_len

        self.embedding = Embedding(vocab_size, d_model, max_len)
        self.blocks = [TransformerBlock(d_model, n_heads, self.d_ff) for _ in range(n_layers)]
        scale = np.sqrt(2.0 / d_model)
        self.W_out = np.random.randn(d_model, vocab_size) * scale
        self.b_out = np.zeros(vocab_size)

    def _causal_mask(self, seq_len):
        mask = np.full((seq_len, seq_len), -1e9)
        mask = np.triu(mask, k=1)
        return mask[None, :, :]  # (1, seq, seq)

    def forward(self, token_ids):
        x = self.embedding.forward(token_ids)
        mask = self._causal_mask(len(token_ids))
        for block in self.blocks:
            x = block.forward(x, mask)
        logits = x @ self.W_out + self.b_out
        return logits

    def predict_next(self, token_ids):
        logits = self.forward(token_ids)
        probs = softmax(logits[-1])
        return probs

    def explain(self, token_ids):
        emb_info = self.embedding.explain(token_ids)
        x = self.embedding.forward(token_ids)
        mask = self._causal_mask(len(token_ids))
        block_infos = []
        for i, block in enumerate(self.blocks):
            info = block.explain(x, mask)
            block_infos.append(info)
            x = np.array(info["output"])
        logits = x @ self.W_out + self.b_out
        probs = softmax(logits[-1])
        top_k = 10
        top_ids = np.argsort(probs)[::-1][:top_k]
        return {
            "embedding": emb_info,
            "blocks": block_infos,
            "final_logits": logits[-1].tolist(),
            "probabilities": probs.tolist(),
            "top_predictions": [{"id": int(i), "prob": float(probs[i])} for i in top_ids],
        }

    def to_dict(self):
        return {
            "vocab_size": self.vocab_size, "d_model": self.d_model,
            "n_heads": self.n_heads, "n_layers": self.n_layers,
            "d_ff": self.d_ff, "max_len": self.max_len,
            "embedding": self.embedding.to_dict(),
            "blocks": [b.to_dict() for b in self.blocks],
            "W_out": self.W_out.tolist(), "b_out": self.b_out.tolist(),
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls.__new__(cls)
        obj.vocab_size = d["vocab_size"]
        obj.d_model = d["d_model"]
        obj.n_heads = d["n_heads"]
        obj.n_layers = d["n_layers"]
        obj.d_ff = d["d_ff"]
        obj.max_len = d["max_len"]
        obj.embedding = Embedding.from_dict(d["embedding"])
        obj.blocks = [TransformerBlock.from_dict(b) for b in d["blocks"]]
        obj.W_out = np.array(d["W_out"])
        obj.b_out = np.array(d["b_out"])
        return obj

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))
