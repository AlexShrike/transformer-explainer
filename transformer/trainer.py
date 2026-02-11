"""
Training loop for the tiny word-level transformer.
Uses analytical gradients for the output layer and evolution strategy for inner layers.
"""

import numpy as np
import json
import os
import time
from .model import Transformer, Tokenizer, softmax


def cross_entropy_loss(logits, targets):
    probs = softmax(logits, axis=-1)
    seq_len = len(targets)
    loss = 0.0
    for t in range(seq_len):
        loss -= np.log(probs[t, targets[t]] + 1e-9)
    return loss / seq_len


def compute_loss(model, token_ids_list):
    total = 0.0
    for ids in token_ids_list:
        inp = ids[:-1]
        tgt = ids[1:]
        logits = model.forward(inp)
        total += cross_entropy_loss(logits, tgt)
    return total / len(token_ids_list)


def get_all_params(model):
    params = []
    params.append(("embedding.token_emb", model.embedding.token_emb))
    params.append(("W_out", model.W_out))
    params.append(("b_out", model.b_out))
    for i, block in enumerate(model.blocks):
        prefix = f"block_{i}"
        params.append((f"{prefix}.attn.W_q", block.attn.W_q))
        params.append((f"{prefix}.attn.W_k", block.attn.W_k))
        params.append((f"{prefix}.attn.W_v", block.attn.W_v))
        params.append((f"{prefix}.attn.W_o", block.attn.W_o))
        params.append((f"{prefix}.ff.W1", block.ff.W1))
        params.append((f"{prefix}.ff.b1", block.ff.b1))
        params.append((f"{prefix}.ff.W2", block.ff.W2))
        params.append((f"{prefix}.ff.b2", block.ff.b2))
        params.append((f"{prefix}.ln1.gamma", block.ln1.gamma))
        params.append((f"{prefix}.ln1.beta", block.ln1.beta))
        params.append((f"{prefix}.ln2.gamma", block.ln2.gamma))
        params.append((f"{prefix}.ln2.beta", block.ln2.beta))
    return params


def train_fast(model, tokenizer, corpus, epochs=200, lr=0.005, verbose=True):
    all_ids = tokenizer.encode(corpus)
    seq_len = 8
    sequences = []
    for i in range(0, len(all_ids) - seq_len, 2):
        sequences.append(np.array(all_ids[i:i + seq_len]))

    if verbose:
        print(f"Fast training: {len(sequences)} sequences, {epochs} epochs")

    for epoch in range(epochs):
        np.random.shuffle(sequences)
        batch = sequences[:min(16, len(sequences))]

        for ids in batch:
            inp = ids[:-1]
            tgt = ids[1:]

            # Forward pass, saving intermediates for each block
            x = model.embedding.forward(inp)
            mask = model._causal_mask(len(inp))
            block_inputs = [x.copy()]
            for block in model.blocks:
                x = block.forward(x, mask)
                block_inputs.append(x.copy())

            logits = x @ model.W_out + model.b_out
            probs = softmax(logits, axis=-1)

            dlogits = probs.copy()
            for t in range(len(tgt)):
                dlogits[t, tgt[t]] -= 1.0
            dlogits /= len(tgt)

            # Update output layer
            model.W_out -= lr * (x.T @ dlogits)
            model.b_out -= lr * dlogits.sum(axis=0)

            # Backprop gradient to hidden
            dx = dlogits @ model.W_out.T

            # Update embeddings
            for t in range(len(inp)):
                model.embedding.token_emb[inp[t]] -= lr * 0.1 * dx[t]

        # Evolution strategy - perturb a few random params each epoch
        if epoch % 3 == 0:
            eval_batch = sequences[:min(8, len(sequences))]
            current_loss = compute_loss(model, eval_batch)

            all_params = []
            for block in model.blocks:
                all_params.extend([
                    block.attn.W_q, block.attn.W_k, block.attn.W_v, block.attn.W_o,
                    block.ff.W1, block.ff.W2,
                ])

            # Only perturb 4 random params per step for speed
            np.random.shuffle(all_params)
            for param in all_params[:4]:
                noise = np.random.randn(*param.shape) * 0.015
                param += noise
                new_loss = compute_loss(model, eval_batch)
                if new_loss > current_loss:
                    param -= noise
                else:
                    current_loss = new_loss

        if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
            loss = compute_loss(model, sequences[:16])
            sample = sequences[0]
            probs = model.predict_next(sample[:-1])
            pred = np.argmax(probs)
            actual = sample[-1]
            print(f"Epoch {epoch:3d} — loss: {loss:.4f} — pred: '{tokenizer.decode([pred])}' actual: '{tokenizer.decode([actual])}'")

    return model


def save_model(model, tokenizer, path):
    data = {
        "model": model.to_dict(),
        "tokenizer": tokenizer.to_dict(),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved to {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def load_model(path):
    with open(path) as f:
        data = json.load(f)
    model = Transformer.from_dict(data["model"])
    tokenizer = Tokenizer.from_dict(data["tokenizer"])
    return model, tokenizer


def pretrain_default(save_path="weights/model.json"):
    """Pre-train a tiny word-level model on a rich corpus."""
    # Highly repetitive corpus with clear patterns for a tiny model to learn
    sentences = [
        # "the" → noun patterns (heavily repeated)
        "the cat sat on the mat",
        "the cat sat on the mat",
        "the dog ran in the park",
        "the dog ran in the park",
        "the bird sat on the tree",
        "the bird flew over the tree",
        "the sun is bright",
        "the book is on the table",
        "the cat is happy",
        "the dog is good",
        "the cat ran to the park",
        "the dog sat on the mat",
        "the bird is small",
        "the cat and the dog",
        "the sun is warm",
        "the book is good",
        # "is" → adjective/state patterns
        "the cat is happy",
        "the dog is good",
        "the bird is small",
        "the sun is bright",
        "the book is good",
        "it is a good day",
        "it is a nice day",
        "he is a good friend",
        "she is a good teacher",
        "the food is good",
        "the day is nice",
        "the weather is nice",
        "it is warm today",
        "it is cold today",
        # "to the" patterns
        "went to the store",
        "went to the park",
        "went to the beach",
        "ran to the park",
        "ran to the store",
        "go to the park",
        "go to the store",
        "go to the beach",
        # "in the" patterns
        "the cat sat in the sun",
        "the dog ran in the park",
        "the bird is in the tree",
        "the book is in the bag",
        "sat in the sun",
        "ran in the park",
        # "on the" patterns
        "the cat sat on the mat",
        "the book is on the table",
        "sat on the mat",
        "sat on the table",
        "the bird sat on the tree",
        # "a" → noun patterns
        "is a good day",
        "is a nice day",
        "is a good friend",
        "is a small cat",
        "is a big dog",
        "have a good day",
        "have a nice day",
        # More repetition of key patterns
        "the cat sat on the mat",
        "the dog ran in the park",
        "the bird flew over the tree",
        "the cat is happy",
        "the dog is good",
        "it is a good day",
        "went to the store",
        "went to the park",
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat is happy",
        "the dog is good",
        "it is a nice day",
        "the sun is bright",
        "the book is on the table",
        "the cat sat in the sun",
    ]
    corpus = ". ".join(sentences) + "."

    tokenizer = Tokenizer()
    tokenizer.fit(corpus)

    print(f"Vocabulary: {tokenizer.vocab_size} words")
    print(f"Sample words: {list(tokenizer.word2id.keys())[:20]}")

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
    )

    model = train_fast(model, tokenizer, corpus, epochs=800, lr=0.008, verbose=True)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    save_model(model, tokenizer, save_path)

    # Test predictions
    test_cases = ["the cat", "is a", "went to", "the", "in the"]
    print("\n=== Test Predictions ===")
    for test in test_cases:
        ids = tokenizer.encode(test)
        if not ids:
            print(f"  '{test}' → (empty encoding)")
            continue
        probs = model.predict_next(np.array(ids))
        top5 = np.argsort(probs)[::-1][:5]
        preds = [f"'{tokenizer.decode([i])}' ({probs[i]:.3f})" for i in top5]
        print(f"  '{test}' → {', '.join(preds)}")

    return model, tokenizer


if __name__ == "__main__":
    pretrain_default()
