
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Sequence

import numpy as np


def _slugify(s: str) -> str:
    s = s.lower().replace("/", "_").replace("-", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:64]


def _texts_fingerprint(texts: Sequence[str]) -> str:
    h = hashlib.blake2b(digest_size=8)
    for t in texts:
        h.update(b"\x00")
        h.update((t if isinstance(t, str) else "").encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _cache_filename(
    cache_dir: Path,
    model_name: str,
    max_length: int,
    pooling: str,
    texts: Sequence[str],
) -> Path:
    fp = _texts_fingerprint(texts)
    name = f"emb_{_slugify(model_name)}_{max_length}_{pooling}_{fp}.npy"
    return Path(cache_dir) / name


def get_or_compute_embeddings(
    texts: Sequence[str],
    cache_dir: str | Path,
    model_name: str = "yikuan8/Clinical-Longformer",
    max_length: int = 1024,
    batch_size: int = 32,
    pooling: str = "mean",
) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_filename(cache_dir, model_name, max_length, pooling, texts)

    if cache_path.exists():
        emb = np.load(cache_path)
        if emb.shape[0] == len(texts):
            print(f"  [text_embedding_cache] hit: {cache_path.name} {emb.shape}")
            return emb
        print(f"  [text_embedding_cache] size mismatch ({emb.shape[0]} vs {len(texts)}); recomputing")

    print(f"  [text_embedding_cache] miss: computing {len(texts)} embeddings via {model_name}")
    emb = _compute_embeddings(texts, model_name, max_length, batch_size, pooling)
    np.save(cache_path, emb)
    print(f"  [text_embedding_cache] saved: {cache_path.name} {emb.shape} ({emb.nbytes / 1e6:.0f} MB)")
    return emb


def pool_hidden_state(hidden, attention_mask, pooling: str):
    import torch

    if pooling == "cls":
        return hidden[:, 0, :]
    if pooling != "mean":
        raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")

    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    return torch.where(counts > 0, summed / counts.clamp(min=1),
                       torch.zeros_like(summed))


def _compute_embeddings(
    texts: Sequence[str],
    model_name: str,
    max_length: int,
    batch_size: int,
    pooling: str,
) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel
    from torch.amp import autocast

    if pooling not in ("mean", "cls"):
        raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [text_embedding_cache] loading {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    n = len(texts)
    clean = [t if isinstance(t, str) else "" for t in texts]


    order = sorted(range(n), key=lambda i: len(clean[i]))
    pooled_by_index = [None] * n

    with torch.no_grad():
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            batch = [clean[i] for i in idx]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            ).to(device)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                hidden = model(**inputs).last_hidden_state
                pooled = pool_hidden_state(hidden, inputs["attention_mask"],
                                           pooling)
            arr = pooled.float().cpu().numpy()
            for row, i in enumerate(idx):
                pooled_by_index[i] = arr[row]
            if (start // batch_size) % 50 == 0:
                print(f"    [text_embedding_cache] {start + len(idx)}/{n} done")

    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


    return np.vstack(pooled_by_index).astype(np.float32)
