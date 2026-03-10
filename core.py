"""core.py — Model loading, math, clustering, search. No UI imports."""

import numpy as np

_CACHE = {}

# ── Layer 0: Atomic helpers ──────────────────────────────────

def cosim(a, b):
    return np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)

def nrm(v):
    return float(np.linalg.norm(v))

def topk(vals, k=10):
    return np.argsort(vals)[-k:][::-1]

def csim_batch(X, v):
    """Cosine sim of every row in X against v."""
    return X @ v / (np.linalg.norm(X, axis=1) * nrm(v) + 1e-9)

# ── Layer 1: Model access ───────────────────────────────────

def load(name="gpt2"):
    if name not in _CACHE:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModel.from_pretrained(name, output_attentions=True)
        mdl.eval()
        _CACHE[name] = (mdl, tok)
    return _CACHE[name]

def vocab(tok):
    return {i: tok.decode([i]) for i in range(tok.vocab_size)}

def param(model, key):
    d = dict(model.named_parameters())
    return d[key].detach().cpu().numpy() if key in d else None

def layer_names(model):
    return [n for n, _ in model.named_parameters()]

def embed_matrix(model):
    for k in ["wte.weight", "embed_tokens.weight", "embeddings.word_embeddings.weight"]:
        w = param(model, k)
        if w is not None: return w
    return next((p.detach().cpu().numpy() for n, p in model.named_parameters()
                 if "embed" in n.lower() and p.dim() == 2), None)

def get_heads(W, nh):
    """Reshape weight (d, cols) → (n_heads, d_head, cols/n_heads_factor)."""
    return W.reshape(W.shape[0], nh, -1).transpose(1, 0, 2)

def cfg(model, *keys):
    """Read model config attribute, trying multiple names."""
    for k in keys:
        v = getattr(model.config, k, None)
        if v is not None: return v
    return 12

def n_heads(model):  return cfg(model, "n_head", "num_attention_heads")
def n_layers(model): return cfg(model, "n_layer", "num_hidden_layers")

# ── Layer 2: Tracing & activations ──────────────────────────

def _forward(model, tok, text):
    import torch
    inp = tok(text, return_tensors="pt")
    with torch.no_grad():
        return model(**inp, output_hidden_states=True, output_attentions=True)

def activations(model, tok, text):
    out = _forward(model, tok, text)
    return {i: h.squeeze(0).cpu().numpy() for i, h in enumerate(out.hidden_states)}

def attn_maps(model, tok, text):
    out = _forward(model, tok, text)
    return {i: a.squeeze(0).cpu().numpy() for i, a in enumerate(out.attentions)}

def token_ids(tok, text):
    return tok.encode(text)

def token_path(acts, tidx):
    """Single token's vector across all layers → (n_layers, d)."""
    return np.stack([acts[i][tidx] for i in sorted(acts)])

def delta_path(path):
    return np.diff(path, axis=0)

# ── Layer 3: Space operations ────────────────────────────────

def reduce(X, n=2, method="pca"):
    if method == "pca":
        from sklearn.decomposition import PCA
        p = PCA(n_components=n).fit(X)
        return {"coords": p.transform(X), "method": method,
                "info": {"variance": p.explained_variance_ratio_.tolist()}}
    if method == "umap":
        from umap import UMAP
        return {"coords": UMAP(n_components=n).fit_transform(X),
                "method": method, "info": {}}
    if method == "tsne":
        from sklearn.manifold import TSNE
        return {"coords": TSNE(n_components=n, perplexity=min(30, len(X)-1))
                .fit_transform(X), "method": method, "info": {}}
    raise ValueError(method)

def cluster(X, method="kmeans", k=12):
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=min(k, len(X)), n_init=10, random_state=42).fit_predict(X)
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

def neighbors(X, idx, k=20):
    sims = csim_batch(X, X[idx])
    top = topk(sims, k + 1)
    return top[top != idx][:k], sims[top[top != idx][:k]]

def search(X, voc, query, tok):
    ids = tok.encode(query)
    if not ids: return []
    sims = csim_batch(X, X[ids[0]])
    top = topk(sims, 50)
    return [(int(i), float(sims[i]), voc.get(i, "?")) for i in top]

def cluster_examples(labels, voc, cid, n=10):
    idxs = np.where(labels == cid)[0]
    sel = idxs[:n] if len(idxs) <= n else np.random.choice(idxs, n, replace=False)
    return [voc.get(int(i), "?").strip() or "·" for i in sel]

def cluster_summary(labels, voc, n=10):
    return {int(c): cluster_examples(labels, voc, c, n)
            for c in np.unique(labels) if c >= 0}

def manifold_stats(X, labels):
    out = {}
    for c in np.unique(labels):
        if c < 0: continue
        pts = X[labels == c]
        mu = pts.mean(0)
        out[int(c)] = {"mean": mu, "spread": float(np.mean(np.linalg.norm(pts - mu, axis=1))),
                        "size": len(pts)}
    return out

# ── Layer 4: Comparison / pattern finding ────────────────────

def compare_tokens(X, ids):
    V = X[ids]
    n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return (V @ V.T) / (n @ n.T)

def shared_dims(X, ids, k=20):
    """Dims where selected tokens agree (low variance, high mean magnitude)."""
    V = X[ids]
    ratio = np.abs(V.mean(0)) / (V.std(0) + 1e-9)
    return topk(ratio, k)

def dim_profile(X, idx):
    return X[idx]

def outlier_dims(vec, X, k=20):
    z = np.abs(vec - X.mean(0)) / (X.std(0) + 1e-9)
    return topk(z, k)

# ── Layer 5: Available models ────────────────────────────────

MODELS = {
    "gpt2":        "GPT-2 (117M)",
    "gpt2-medium": "GPT-2 Medium (345M)",
    "gpt2-large":  "GPT-2 Large (774M)",
    "distilgpt2":  "DistilGPT-2 (82M)",
    "EleutherAI/pythia-70m": "Pythia 70M",
    "EleutherAI/pythia-160m": "Pythia 160M",
}

# ── TeX strings (for temml rendering) ────────────────────────

def tex_overview(X):
    V, d = X.shape
    return f"\\mathbf{{E}} \\in \\mathbb{{R}}^{{{V} \\times {d}}}"

def tex_pca(info):
    v = info.get("variance", [])
    return ", ".join(f"\\sigma_{{{i+1}}}^2={x:.1%}" for i, x in enumerate(v[:3]))

def tex_token(vec):
    return f"\\|\\mathbf{{v}}\\| = {nrm(vec):.2f},\\; d = {len(vec)}"

def tex_cluster(stats, cid):
    s = stats.get(cid, {})
    return f"n={s.get('size','?')},\\; \\bar{{d}}_{{intra}}={s.get('spread',0):.3f}"

def tex_delta():
    return "\\Delta_l = \\mathbf{h}^{(l)} - \\mathbf{h}^{(l-1)}"
