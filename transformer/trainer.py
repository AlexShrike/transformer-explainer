"""
Simple training loop for the tiny transformer.
Uses basic SGD with gradient estimation via finite differences (numerical gradients)
since we want pure numpy with no autograd.

For practical training speed we use a simple cross-entropy loss and parameter-wise
numerical gradient computation. This is slow but transparent and only needs to run once
on a tiny model.
"""

import numpy as np
import json
import os
import time
from .model import Transformer, Tokenizer, softmax


def cross_entropy_loss(logits, targets):
    """Cross-entropy loss for a sequence. logits: (seq, vocab), targets: (seq,)"""
    probs = softmax(logits, axis=-1)
    seq_len = len(targets)
    loss = 0.0
    for t in range(seq_len):
        loss -= np.log(probs[t, targets[t]] + 1e-9)
    return loss / seq_len


def compute_loss(model, token_ids_list):
    """Average loss over multiple sequences."""
    total = 0.0
    for ids in token_ids_list:
        inp = ids[:-1]
        tgt = ids[1:]
        logits = model.forward(inp)
        total += cross_entropy_loss(logits, tgt)
    return total / len(token_ids_list)


def perturb_param(arr, idx, eps):
    """Get flat index into array, perturb by eps."""
    flat = arr.ravel()
    old = flat[idx]
    flat[idx] = old + eps
    return old


def restore_param(arr, idx, old_val):
    flat = arr.ravel()
    flat[idx] = old_val


def numerical_gradient(model, data, param_array, eps=1e-4):
    """Compute numerical gradient for a single parameter array."""
    grad = np.zeros_like(param_array.ravel())
    for i in range(len(grad)):
        old = perturb_param(param_array, i, eps)
        loss_plus = compute_loss(model, data)
        restore_param(param_array, i, old)

        old = perturb_param(param_array, i, -eps)
        loss_minus = compute_loss(model, data)
        restore_param(param_array, i, old)

        grad[i] = (loss_plus - loss_minus) / (2 * eps)
    return grad.reshape(param_array.shape)


def get_all_params(model):
    """Return list of (name, array_reference) for all trainable parameters."""
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


def train(model, tokenizer, corpus, epochs=50, lr=0.01, batch_size=4, verbose=True):
    """
    Train using simple SGD with analytical gradients approximated via
    forward-mode finite differences. Works for tiny models only.
    
    For speed, we use a simplified approach: only update output layer and
    attention weights with numerical gradients on small batches.
    """
    # Prepare data: split corpus into sequences
    all_ids = tokenizer.encode(corpus)
    seq_len = 16  # short sequences for tiny model
    sequences = []
    for i in range(0, len(all_ids) - seq_len, seq_len // 2):
        sequences.append(np.array(all_ids[i:i + seq_len]))
    
    if not sequences:
        print("Corpus too short!")
        return

    if verbose:
        print(f"Training on {len(sequences)} sequences of length {seq_len}")
        print(f"Vocab size: {tokenizer.vocab_size}, Params: ~{sum(p.size for _, p in get_all_params(model))}")

    params = get_all_params(model)
    
    for epoch in range(epochs):
        np.random.shuffle(sequences)
        batch = sequences[:batch_size]
        
        loss_before = compute_loss(model, batch)
        
        # Update each parameter
        for name, param in params:
            grad = numerical_gradient(model, batch, param, eps=1e-3)
            param -= lr * grad
        
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            loss_after = compute_loss(model, sequences[:8])
            print(f"Epoch {epoch}/{epochs} — loss: {loss_after:.4f}")
    
    return model


def train_fast(model, tokenizer, corpus, epochs=200, lr=0.005, verbose=True):
    """
    Faster training using analytical gradient for the output layer 
    and stochastic parameter perturbation for other layers.
    """
    all_ids = tokenizer.encode(corpus)
    seq_len = 12
    sequences = []
    for i in range(0, len(all_ids) - seq_len, 4):
        sequences.append(np.array(all_ids[i:i + seq_len]))
    
    if verbose:
        print(f"Fast training: {len(sequences)} sequences, {epochs} epochs")

    for epoch in range(epochs):
        np.random.shuffle(sequences)
        batch = sequences[:8]
        
        # For each sequence in batch, compute gradient for W_out analytically
        for ids in batch:
            inp = ids[:-1]
            tgt = ids[1:]
            
            # Forward pass
            x = model.embedding.forward(inp)
            mask = model._causal_mask(len(inp))
            for block in model.blocks:
                x = block.forward(x, mask)
            
            logits = x @ model.W_out + model.b_out
            probs = softmax(logits, axis=-1)
            
            # Gradient of cross-entropy w.r.t. logits
            dlogits = probs.copy()
            for t in range(len(tgt)):
                dlogits[t, tgt[t]] -= 1.0
            dlogits /= len(tgt)
            
            # Gradient for W_out and b_out
            model.W_out -= lr * (x.T @ dlogits)
            model.b_out -= lr * dlogits.sum(axis=0)
            
            # Gradient for last hidden state -> backprop through blocks approximately
            dx = dlogits @ model.W_out.T
            
            # Simple gradient for embedding
            for t in range(len(inp)):
                model.embedding.token_emb[inp[t]] -= lr * 0.1 * dx[t]
        
        # Occasionally perturb other params with evolution strategy
        if epoch % 3 == 0:
            params_to_evolve = []
            for block in model.blocks:
                params_to_evolve.extend([block.attn.W_q, block.attn.W_k, block.attn.W_v, block.attn.W_o,
                                         block.ff.W1, block.ff.W2])
            
            current_loss = compute_loss(model, batch)
            for param in params_to_evolve:
                noise = np.random.randn(*param.shape) * 0.01
                param += noise
                new_loss = compute_loss(model, batch)
                if new_loss > current_loss:
                    param -= noise  # revert
                else:
                    current_loss = new_loss
        
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            loss = compute_loss(model, sequences[:16])
            # Show a sample prediction
            sample = sequences[0]
            probs = model.predict_next(sample[:-1])
            pred = np.argmax(probs)
            actual = sample[-1]
            print(f"Epoch {epoch:3d} — loss: {loss:.4f} — pred: '{tokenizer.decode([pred])}' actual: '{tokenizer.decode([actual])}'")
    
    return model


def save_model(model, tokenizer, path):
    """Save model and tokenizer together."""
    data = {
        "model": model.to_dict(),
        "tokenizer": tokenizer.to_dict(),
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved to {path} ({os.path.getsize(path) / 1024:.0f} KB)")


def load_model(path):
    """Load model and tokenizer."""
    with open(path) as f:
        data = json.load(f)
    model = Transformer.from_dict(data["model"])
    tokenizer = Tokenizer.from_dict(data["tokenizer"])
    return model, tokenizer


def pretrain_default(save_path="weights/model.json"):
    """Pre-train a tiny model on a simple corpus and save weights."""
    corpus = """the cat sat on the mat. the dog ran in the park. a bird flew over the tree.
the cat and the dog are friends. the bird sat on the tree. the dog chased the cat.
the cat ran to the mat. the bird flew to the park. the dog sat on the mat.
a cat is on the mat. a dog is in the park. a bird is on the tree.
the cat sat on the mat and the dog ran to the park.
the bird flew over the tree and sat on the mat.
the cat and the bird are on the mat. the dog ran to the tree.
the dog chased the bird over the park. the cat sat and watched.
a cat sat on a mat. a dog ran in a park. a bird flew over a tree.
the mat is where the cat sat. the park is where the dog ran.
"""

    tokenizer = Tokenizer()
    tokenizer.fit(corpus)
    
    print(f"Vocabulary: {tokenizer.vocab_size} chars: {list(tokenizer.char2id.keys())}")
    
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
    )
    
    model = train_fast(model, tokenizer, corpus, epochs=300, lr=0.005, verbose=True)
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    save_model(model, tokenizer, save_path)
    
    # Test
    test = "the cat "
    ids = tokenizer.encode(test)
    probs = model.predict_next(np.array(ids))
    top5 = np.argsort(probs)[::-1][:5]
    print(f"\nTest: '{test}' → top predictions:")
    for i in top5:
        print(f"  '{tokenizer.decode([i])}' ({probs[i]:.3f})")
    
    return model, tokenizer


if __name__ == "__main__":
    pretrain_default()
