"""FastAPI backend for the Transformer Explainer."""

import os
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from transformer.model import Transformer, Tokenizer, softmax
from transformer.trainer import load_model, pretrain_default

app = FastAPI(title="Transformer Explainer")

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "model.json")

# Load or train model
if os.path.exists(WEIGHTS_PATH):
    model, tokenizer = load_model(WEIGHTS_PATH)
    print(f"Loaded model from {WEIGHTS_PATH}")
else:
    print("No pre-trained weights found. Training...")
    model, tokenizer = pretrain_default(WEIGHTS_PATH)


class ProcessRequest(BaseModel):
    text: str

class GenerateRequest(BaseModel):
    text: str
    max_tokens: int = 20


@app.get("/api/vocab")
def get_vocab():
    return {"vocab": tokenizer.char2id, "vocab_size": tokenizer.vocab_size}


@app.get("/api/model-info")
def get_model_info():
    return {
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "n_heads": model.n_heads,
        "n_layers": model.n_layers,
        "d_ff": model.d_ff,
        "max_len": model.max_len,
    }


@app.post("/api/process")
def process_text(req: ProcessRequest):
    text = req.text
    tok_info = tokenizer.explain(text)
    token_ids = np.array(tokenizer.encode(text))
    
    if len(token_ids) == 0:
        return {"error": "Empty input"}
    
    explain = model.explain(token_ids)
    
    # Add token labels
    explain["tokens"] = tok_info["tokens"]
    explain["token_ids"] = tok_info["token_ids"]
    explain["vocab_size"] = tok_info["vocab_size"]
    
    # Map prediction IDs to chars
    for pred in explain["top_predictions"]:
        pred["char"] = tokenizer.decode([pred["id"]])
    
    return explain


@app.post("/api/generate")
def generate(req: GenerateRequest):
    text = req.text
    ids = tokenizer.encode(text)
    steps = []
    
    for _ in range(req.max_tokens):
        token_ids = np.array(ids)
        explain = model.explain(token_ids)
        probs = np.array(explain["probabilities"])
        next_id = int(np.argmax(probs))
        next_char = tokenizer.decode([next_id])
        
        # Slim down step info for generation (just attention weights and predictions)
        step = {
            "input_text": tokenizer.decode(ids),
            "predicted_char": next_char,
            "predicted_id": next_id,
            "top_predictions": [{"id": int(i), "char": tokenizer.decode([i]), "prob": float(probs[i])}
                                for i in np.argsort(probs)[::-1][:5]],
            "attention_weights": [[head.tolist() if not isinstance(head, list) else head
                                   for head in block["attention"]["attn_weights"]]
                                  for block in explain["blocks"]],
        }
        steps.append(step)
        ids.append(next_id)
        
        if next_char == "." or next_char == "\n":
            break
    
    return {
        "input": req.text,
        "output": tokenizer.decode(ids),
        "steps": steps,
    }


# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
