#!/usr/bin/env python3
"""LLM Vector Space Explorer — single-file. Run: python main.py"""

## Let LLM try to estimate directions based on arrows to put on to maybe find meaningful correlations

# ══════════════════════════════════════════════════════════════
# SECTION 0: THREADING FIX — must be before ANY other imports
# ══════════════════════════════════════════════════════════════

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_VERBOSE"] = "0"

MODELS = {
    "gpt2": "GPT-2 (117M)",
    "gpt2-medium": "GPT-2 Medium (345M)",
    "distilgpt2": "DistilGPT-2 (82M)",
    "EleutherAI/pythia-70m": "Pythia 70M",
    "EleutherAI/pythia-160m": "Pythia 160M",
}

# ══════════════════════════════════════════════════════════════
# SECTION 1: VENV BOOTSTRAP
# ══════════════════════════════════════════════════════════════

import sys, platform, subprocess, shutil
from pathlib import Path
from core import *
from views import *

VENV = Path.home() / ".llm_explorer_venv"
IS_WIN = platform.system() == "Windows"
BIN = VENV / ("Scripts" if IS_WIN else "bin")
PY = BIN / ("python.exe" if IS_WIN else "python")
PIP = BIN / ("pip.exe" if IS_WIN else "pip")

DEPS = [
    "dash", "plotly", "numpy", "scikit-learn",
    "torch --index-url https://download.pytorch.org/whl/cpu",
    "transformers", "umap-learn", "rich",
]

def in_venv():
    return sys.prefix == str(VENV)

def run_cmd(cmd, **kw):
    print(f"  -> {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd, **kw)

def pip_install(spec):
    run_cmd([str(PIP), "install", "-q"] + spec.split())

def create_venv():
    import venv as _venv
    print(f"Creating venv at {VENV}")
    _venv.create(str(VENV), with_pip=True)
    run_cmd([str(PY), "-m", "pip", "install", "-q", "--upgrade", "pip"])

def install_deps():
    print("Installing dependencies...")
    for dep in DEPS:
        try:
            pip_install(dep)
        except subprocess.CalledProcessError:
            print(f"  Warning: failed to install {dep}")

def ensure_venv():
    if not PY.exists():
        if VENV.exists():
            shutil.rmtree(VENV)
        create_venv()
        install_deps()
    else:
        rc = subprocess.call(
            [str(PY), "-c", "import dash, torch, transformers, sklearn, rich"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if rc != 0:
            install_deps()

def relaunch():
    print(f"Launching via {PY}\n")
    try:
        r = subprocess.run([str(PY), str(Path(__file__).resolve())], env={**os.environ})
        sys.exit(r.returncode)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)

if not in_venv():
    ensure_venv()
    relaunch()
    sys.exit(0)

# ══════════════════════════════════════════════════════════════
# SECTION 2: CORE — pure logic, no UI
# ══════════════════════════════════════════════════════════════

import numpy as np
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

con = Console()

_CACHE = {}
MAX_TOKENS = 5000
LLM_LABEL_MAX_EXAMPLES = 500  # max examples to send to LLM for labeling

# ── Layer 0: Atomic ──────────────────────────────────────────

def cosim(a, b):
    return np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)

def nrm(v):
    return float(np.linalg.norm(v))

def topk(vals, k=10):
    k = min(k, len(vals))
    return np.argsort(vals)[-k:][::-1]

def csim_batch(X, v):
    v_norm = nrm(v)
    if v_norm < 1e-9:
        return np.zeros(len(X))
    n = np.linalg.norm(X, axis=1) * v_norm + 1e-9
    return X @ v / n

# ── Layer 2: Tracing ─────────────────────────────────────────

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
    return np.stack([acts[i][tidx] for i in sorted(acts)])

def delta_path(path):
    return np.diff(path, axis=0)

# ── Layer 3: Space operations ────────────────────────────────

def reduce(X, n=2, method="pca", dim_indices=None):
    """Reduce dimensionality. If dim_indices is provided, only use those dimensions."""
    if dim_indices is not None and len(dim_indices) > 0:
        X = X[:, dim_indices]
    if len(X) < 4:
        pad = np.zeros((4, X.shape[1]))
        pad[:len(X)] = X
        X = pad
    if method == "pca":
        from sklearn.decomposition import PCA
        nc = min(n, X.shape[0], X.shape[1])
        p = PCA(n_components=nc).fit(X)
        return {"coords": p.transform(X), "method": method,
                "info": {"variance": p.explained_variance_ratio_.tolist()}}
    if method == "umap":
        from umap import UMAP
        nn = min(15, len(X) - 1)
        return {"coords": UMAP(n_components=n, n_neighbors=nn).fit_transform(X),
                "method": method, "info": {}}
    if method == "tsne":
        from sklearn.manifold import TSNE
        pp = min(30, len(X) - 1)
        return {"coords": TSNE(n_components=n, perplexity=max(pp, 2))
                .fit_transform(X), "method": method, "info": {}}
    raise ValueError(method)

def cluster(X, method="kmeans", k=12):
    k = min(k, len(X) - 1, 30)
    if k < 2:
        return np.zeros(len(X), dtype=int)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, n_init=5, random_state=42, max_iter=100).fit_predict(X)
    from sklearn.cluster import DBSCAN
    return DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

def neighbors(X, idx, k=20):
    sims = csim_batch(X, X[idx])
    top = topk(sims, k + 1)
    top = top[top != idx][:k]
    return top, sims[top]

def search(X, voc, query, tok):
    ids = tok.encode(query)
    if not ids:
        return []
    sims = csim_batch(X, X[ids[0]])
    top = topk(sims, 50)
    return [(int(i), float(sims[i]), voc.get(i, "?")) for i in top]

def cluster_examples(labels, voc, cid, n=10):
    idxs = np.where(labels == cid)[0]
    if len(idxs) == 0:
        return ["(empty)"]
    sel = idxs[:n] if len(idxs) <= n else np.random.default_rng(42).choice(idxs, n, replace=False)
    return [voc.get(int(i), "?").strip() or "." for i in sel]

def cluster_summary(labels, voc, n=10):
    return {int(c): cluster_examples(labels, voc, c, n) for c in np.unique(labels) if c >= 0}

def manifold_stats(X, labels):
    out = {}
    for c in np.unique(labels):
        if c < 0:
            continue
        pts = X[labels == c]
        mu = pts.mean(0)
        out[int(c)] = {"mean": mu, "spread": float(np.mean(np.linalg.norm(pts - mu, axis=1))),
                        "size": len(pts)}
    return out

# ── LLM-based cluster labeling ───────────────────────────────

def llm_label_cluster(model, tok, examples, max_examples=100):
    """
    Use the loaded LLM to generate a short label for a cluster.
    We carefully limit examples to fit within the model's context window,
    and use a tightly constrained prompt + generation to get a usable label.
    """
    import torch

    # Determine model's max context length
    max_ctx = getattr(model.config, 'n_positions', None) \
           or getattr(model.config, 'max_position_embeddings', 1024)

    # Clean up examples
    examples = [e.strip() for e in examples if e.strip() and e.strip() != "." and e.strip() != "?"]
    if not examples:
        return "(empty cluster)"

    # Deduplicate while preserving order
    seen = set()
    unique_examples = []
    for e in examples:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique_examples.append(e)
    examples = unique_examples

    # Subsample if too many
    if len(examples) > max_examples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(examples), max_examples, replace=False)
        examples = [examples[i] for i in sorted(indices)]

    # Build prompt, but we must ensure it fits in context.
    # Reserve tokens for generation (10) and some safety margin (50).
    max_prompt_tokens = max_ctx - 60

    # Start with a tight prompt format that steers GPT-2 better
    prefix = 'Word category: "'
    suffix_after_examples = '"\nCategory name:'

    # Incrementally add examples until we'd exceed the token budget
    selected = []
    for ex in examples:
        candidate = prefix + ", ".join(selected + [ex]) + suffix_after_examples
        n_tokens = len(tok.encode(candidate))
        if n_tokens > max_prompt_tokens:
            break
        selected.append(ex)

    if not selected:
        # Even one example is too long? Just use first 3 chars
        selected = [examples[0][:20]]

    prompt = prefix + ", ".join(selected) + suffix_after_examples

    input_ids = tok(prompt, return_tensors="pt")["input_ids"]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=12,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
            repetition_penalty=1.3,
        )

    generated = out[0][input_ids.shape[1]:]
    label = tok.decode(generated, skip_special_tokens=True).strip()

    # Clean up: take first line, first sentence, strip quotes/punctuation
    label = label.split("\n")[0].split(".")[0].split('"')[0].strip()
    label = label.strip(":-–— \t")

    if len(label) > 50:
        label = label[:47] + "…"

    # If the model produced garbage or empty, fall back to showing top examples
    if not label or len(label) < 2 or label.lower().startswith("the ") and len(label) > 40:
        # Fallback: pick 3 representative examples as the label
        label = " / ".join(selected[:4])
        if len(label) > 50:
            label = label[:47] + "…"

    return label


def llm_label_all_clusters(model, tok, labels, voc, sidx, max_examples=100):
    """
    Generate LLM-based labels for all clusters.
    Returns dict: cluster_id -> label string
    """
    cluster_labels = {}
    unique_clusters = sorted([c for c in np.unique(labels) if c >= 0])

    for cid in unique_clusters:
        mask = labels == cid
        # Map local indices back to global vocab indices
        gids = sidx[mask] if sidx is not None else np.where(mask)[0]
        examples = [voc.get(int(i), "").strip() for i in gids]
        examples = [e for e in examples if e and e != "?" and e != "." and len(e) > 0]

        n_total = len(examples)

        try:
            label = llm_label_cluster(model, tok, examples, max_examples)
            cluster_labels[cid] = f"{label} ({n_total})"
            con.print(f"  [green]C{cid}[/green]: {label} ({n_total} tokens)")
        except Exception as e:
            # Fallback: just show a few examples
            fallback = " / ".join(examples[:4]) if examples else "?"
            if len(fallback) > 50:
                fallback = fallback[:47] + "…"
            cluster_labels[cid] = f"{fallback} ({n_total})"
            con.print(f"  [yellow]C{cid}[/yellow]: fallback — {e}")

    return cluster_labels

def heuristic_label_cluster(examples, max_label_len=50):
    """
    Generate a descriptive label for a cluster using heuristic analysis
    instead of relying on GPT-2 generation.
    """
    examples = [e.strip() for e in examples if e.strip() and e not in (".", "?", "")]
    if not examples:
        return "(empty cluster)"

    # Deduplicate
    seen = set()
    unique = []
    for e in examples:
        low = e.lower().strip()
        if low not in seen:
            seen.add(low)
            unique.append(e)
    examples = unique

    # --- Strategy 1: Check for common prefix/suffix patterns ---
    cleaned = [e.strip().lower() for e in examples if len(e.strip()) > 1]

    if cleaned:
        # Check common prefix (at least 2 chars)
        prefix = os.path.commonprefix(cleaned)
        if len(prefix) >= 3:
            return f"prefix: '{prefix}…'"

        # Check common suffix
        reversed_strs = [s[::-1] for s in cleaned]
        suffix = os.path.commonprefix(reversed_strs)[::-1]
        if len(suffix) >= 3:
            return f"suffix: '…{suffix}'"

    # --- Strategy 2: Character-type analysis ---
    has_alpha = sum(1 for e in examples if any(c.isalpha() for c in e))
    has_digit = sum(1 for e in examples if any(c.isdigit() for c in e))
    has_upper = sum(1 for e in examples if e.strip() and e.strip()[0].isupper())
    has_punct = sum(1 for e in examples if all(not c.isalnum() for c in e.strip()))
    avg_len = np.mean([len(e.strip()) for e in examples]) if examples else 0
    n = len(examples)

    type_tags = []
    if has_punct / max(n, 1) > 0.5:
        type_tags.append("punctuation/symbols")
    if has_digit / max(n, 1) > 0.5:
        type_tags.append("numeric")
    if has_upper / max(n, 1) > 0.7:
        type_tags.append("capitalized")
    if avg_len < 2.5:
        type_tags.append("short fragments")
    elif avg_len > 8:
        type_tags.append("long tokens")

    # --- Strategy 3: Check if tokens are mostly real words ---
    # Tokens starting with space are typically full words in GPT-2
    space_prefixed = sum(1 for e in examples if e.startswith(" ") or e.startswith("Ġ"))
    if space_prefixed / max(n, 1) > 0.6:
        type_tags.append("whole words")
    elif space_prefixed / max(n, 1) < 0.2:
        type_tags.append("subwords/continuations")

    # --- Strategy 4: Pick most representative examples ---
    # Sort by frequency-like heuristic: prefer readable tokens
    readable = [e for e in examples if len(e.strip()) >= 2 and any(c.isalpha() for c in e)]
    representatives = (readable or examples)[:5]
    rep_str = ", ".join(r.strip() for r in representatives)
    if len(rep_str) > 35:
        rep_str = rep_str[:32] + "…"

    if type_tags:
        label = f"{' + '.join(type_tags[:2])}: {rep_str}"
    else:
        label = rep_str

    if len(label) > max_label_len:
        label = label[:max_label_len - 1] + "…"

    return label


def heuristic_label_all_clusters(labels, voc, sidx, max_examples=500):
    """Generate heuristic labels for all clusters."""
    cluster_labels = {}
    unique_clusters = sorted([c for c in np.unique(labels) if c >= 0])

    for cid in unique_clusters:
        mask = labels == cid
        gids = sidx[mask] if sidx is not None else np.where(mask)[0]
        examples = [voc.get(int(i), "").strip() for i in gids]
        examples = [e for e in examples if e and e != "?" and len(e) > 0]
        n_total = len(examples)

        # Subsample for analysis
        if len(examples) > max_examples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(examples), max_examples, replace=False)
            examples = [examples[i] for i in sorted(idx)]

        label = heuristic_label_cluster(examples)
        cluster_labels[cid] = f"{label} ({n_total})"
        con.print(f"  [green]C{cid}[/green]: {label} ({n_total} tokens)")

    return cluster_labels

# ── Layer 4: Comparison ──────────────────────────────────────

def compare_tokens(X, ids):
    V = X[ids]
    n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return (V @ V.T) / (n @ n.T)

def shared_dims(X, ids, k=20):
    V = X[ids]
    ratio = np.abs(V.mean(0)) / (V.std(0) + 1e-9)
    return topk(ratio, k)

def outlier_dims(vec, X, k=20):
    z = np.abs(vec - X.mean(0)) / (X.std(0) + 1e-9)
    return topk(z, k)

# ── TeX strings ──────────────────────────────────────────────

def tex_overview(E, n_sampled=None):
    V, d = E.shape
    s = f"\\mathbf{{E}} \\in \\mathbb{{R}}^{{{V} \\times {d}}}"
    if n_sampled and n_sampled < V:
        s += f"\\quad (\\text{{showing }} {n_sampled})"
    return s

def tex_pca(info):
    v = info.get("variance", [])
    if not v:
        return ""
    return ", ".join(f"\\sigma_{{{i+1}}}^2={x:.1%}" for i, x in enumerate(v[:3]))

def tex_token(vec):
    return f"\\|\\mathbf{{v}}\\| = {nrm(vec):.2f},\\; d = {len(vec)}"

def tex_cluster(stats, cid):
    s = stats.get(cid, {})
    return f"n={s.get('size', '?')},\\; \\bar{{d}}_{{intra}}={s.get('spread', 0):.3f}"

def tex_delta():
    return "\\Delta_l = \\mathbf{h}^{(l)} - \\mathbf{h}^{(l-1)}"

# ══════════════════════════════════════════════════════════════
# SECTION 3: DASH APP
# ══════════════════════════════════════════════════════════════

import json
import dash
from dash import dcc, html, Input, Output, State, callback, no_update, ctx
import plotly.graph_objects as go
import plotly.express as px

DEFAULT_MODEL = "gpt2"
DEFAULT_K = 12
DEFAULT_VIZ = "scatter"
DEFAULT_REDUCE = "pca"

def empty_fig(msg="No data yet"):
    return go.Figure().update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(size=16, color="#555"),
                          xref="paper", yref="paper", x=0.5, y=0.5)])

# ── Navigation ────────────────────────────────────────────────

def nav_push(stack, entry):
    # Prevent infinite loops: don't push if same level+id as current
    cur = nav_current(stack)
    if (cur.get("level") == entry.get("level") and
        cur.get("id") == entry.get("id")):
        return stack
    # Also limit max depth to prevent runaway navigation
    if len(stack) >= 20:
        return stack
    return stack + [entry]

def nav_pop(stack):
    return stack[:max(len(stack) - 1, 1)]

def nav_current(stack):
    return stack[-1] if stack else {"level": "model", "id": DEFAULT_MODEL, "label": "GPT-2"}

ICONS = {"model": "🏠", "cluster": "🔵", "token": "🔤",
         "trace": "📈", "attention": "👁", "compare": "⚖️"}

def nav_crumbs(stack):
    spans = []
    for i, s in enumerate(stack):
        icon = ICONS.get(s.get("level", ""), "")
        last = i == len(stack) - 1
        spans.append(html.Span(
            f"{icon} {s['label']}",
            id={"type": "crumb", "index": i},
            style={"cursor": "pointer", "padding": "4px 10px", "margin": "0 2px",
                   "borderRadius": "6px", "fontSize": "13px",
                   "background": "#4a9eff" if last else "#2a2a3a",
                   "color": "#fff", "fontWeight": "bold" if last else "normal",
                   "display": "inline-block"}
        ))
        if not last:
            spans.append(html.Span(" > ", style={"color": "#555"}))
    return spans

def level_label(stack):
    cur = nav_current(stack)
    return f"Depth {len(stack)} - {cur.get('level', 'model').title()} view"

def init_stack(mn):
    return [{"level": "model", "id": mn, "label": MODELS.get(mn, mn)}]

# ── Figures ───────────────────────────────────────────────────

def _lay(fig, title="", h=550):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_dark", height=h,
        paper_bgcolor="rgba(13,13,26,0.8)", plot_bgcolor="rgba(26,26,46,0.6)",
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0.3)"),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=12))
    return fig

def fig_scatter(coords, labels, texts, title="", cluster_labels=None):
    fig = go.Figure()
    for c in sorted(np.unique(labels)):
        mask = labels == c
        if cluster_labels and c in cluster_labels:
            name = cluster_labels[c]
        else:
            name = f"Cluster {c}" if c >= 0 else "Noise"
        idxs = np.where(mask)[0]
        fig.add_trace(go.Scattergl(
            x=coords[mask, 0], y=coords[mask, 1], mode="markers",
            marker=dict(size=5, opacity=0.65),
            text=[texts[i] for i in idxs],
            hovertemplate="%{text}<extra>" + name + "</extra>",
            name=name, customdata=idxs.tolist()))
    return _lay(fig, title)

def fig_scatter_3d(coords, labels, texts, title="", cluster_labels=None):
    fig = go.Figure()
    for c in sorted(np.unique(labels)):
        mask = labels == c
        if cluster_labels and c in cluster_labels:
            name = cluster_labels[c]
        else:
            name = f"Cluster {c}" if c >= 0 else "Noise"
        idxs = np.where(mask)[0]
        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=coords[mask, 2],
            mode="markers",
            marker=dict(size=3, opacity=0.65),
            text=[texts[i] for i in idxs],
            hovertemplate="%{text}<extra>" + name + "</extra>",
            name=name,
            customdata=idxs.tolist()))
    return _lay(fig, title, h=650)

def fig_heatmap(matrix, xl, yl, title=""):
    fig = go.Figure(go.Heatmap(
        z=matrix, x=xl, y=yl, colorscale="Viridis",
        hovertemplate="x=%{x}<br>y=%{y}<br>val=%{z:.3f}<extra></extra>"))
    return _lay(fig, title, 500).update_layout(margin=dict(l=80, b=80))

def fig_bars(vec, title="Raw Dimensions", top_k=50):
    idx = topk(np.abs(vec), top_k)
    fig = go.Figure(go.Bar(
        x=[f"d{i}" for i in idx], y=vec[idx],
        marker_color=["#4a9eff" if v > 0 else "#ff6b6b" for v in vec[idx]]))
    return _lay(fig, title, 300).update_layout(margin=dict(l=30, r=10, t=35, b=30))

def fig_radar(vec, top_k=12):
    idx = topk(np.abs(vec), top_k)
    fig = go.Figure(go.Scatterpolar(
        r=np.abs(vec[idx]), theta=[f"d{i}" for i in idx], fill="toself",
        marker=dict(color="#4a9eff"), line=dict(color="#4a9eff")))
    return _lay(fig, h=350).update_layout(
        polar=dict(bgcolor="#1a1a2e", radialaxis=dict(visible=True, color="#555")),
        margin=dict(l=40, r=40, t=30, b=30))

def fig_parallel(X, labels, k=15):
    dims = topk(X.std(axis=0), k)
    df = {f"d{d}": X[:, d] for d in dims}
    df["cluster"] = labels
    fig = px.parallel_coordinates(df, color="cluster",
        dimensions=[f"d{d}" for d in dims], color_continuous_scale="Viridis")
    return _lay(fig, h=450).update_layout(margin=dict(l=40, r=40))

def fig_path(points, labels):
    r = reduce(points, n=2, method="pca")
    c = r["coords"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=c[:, 0], y=c[:, 1], mode="lines",
        line=dict(color="#444", dash="dot", width=1),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=c[:, 0], y=c[:, 1], mode="markers+text",
        text=labels, textposition="top center", textfont=dict(size=9, color="#ccc"),
        marker=dict(
            size=12, symbol="arrow", angleref="previous",
            color=list(range(len(c))), colorscale="Plasma",
            showscale=True, colorbar=dict(title="Layer", thickness=10),
            line=dict(width=1, color="#fff")),
        hovertemplate="%{text}<extra>Layer %{marker.color}</extra>",
        showlegend=False))
    fig.add_trace(go.Scatter(
        x=[c[0, 0]], y=[c[0, 1]], mode="markers",
        marker=dict(size=16, color="#00ff88", symbol="star", line=dict(width=2, color="#000")),
        name="Start", hovertemplate=f"{labels[0]}<extra>Start</extra>"))
    fig.add_trace(go.Scatter(
        x=[c[-1, 0]], y=[c[-1, 1]], mode="markers",
        marker=dict(size=16, color="#ff4444", symbol="diamond", line=dict(width=2, color="#000")),
        name="End", hovertemplate=f"{labels[-1]}<extra>End</extra>"))
    return _lay(fig, "Token trajectory through layers", 500)

def fig_attn(am, tokens, head=0, title=""):
    m = am[head] if am.ndim == 3 else am
    return fig_heatmap(m, tokens, tokens, title or f"Attention Head {head}")

# ── Parse dimension range string ─────────────────────────────

def parse_dim_range(dim_str, max_dim):
    """
    Parse a dimension selection string like '0-100,200-300,500' into a list of indices.
    Supports: individual dims (5), ranges (0-100), comma-separated combos.
    Returns None if empty/invalid (meaning use all dims).
    """
    if not dim_str or not dim_str.strip():
        return None
    indices = set()
    parts = dim_str.replace(" ", "").split(",")
    for part in parts:
        if "-" in part:
            try:
                lo, hi = part.split("-", 1)
                lo, hi = int(lo), int(hi)
                lo = max(0, lo)
                hi = min(hi, max_dim - 1)
                indices.update(range(lo, hi + 1))
            except ValueError:
                continue
        else:
            try:
                d = int(part)
                if 0 <= d < max_dim:
                    indices.add(d)
            except ValueError:
                continue
    if not indices:
        return None
    return sorted(indices)

# ── Layout ────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.index_string = '''<!DOCTYPE html>
<html><head>{%metas%}<title>LLM Vector Space Explorer</title>{%css%}<style>
* { box-sizing: border-box; }
body { background: #0d0d1a; color: #eee; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; }
.sidebar {
    width: 280px; padding: 16px; background: #12122a;
    height: 100vh; position: fixed; top: 0; left: 0;
    overflow-y: auto; overflow-x: hidden;
    border-right: 1px solid #1f1f3a;
    scrollbar-width: thin;
    scrollbar-color: #3a3a5a #12122a;
}
.sidebar::-webkit-scrollbar { width: 6px; }
.sidebar::-webkit-scrollbar-track { background: #12122a; }
.sidebar::-webkit-scrollbar-thumb { background: #3a3a5a; border-radius: 3px; }
.sidebar::-webkit-scrollbar-thumb:hover { background: #4a4a6a; }
.sb-section {
    background: #1a1a2e; border-radius: 8px; padding: 10px 12px;
    margin-bottom: 10px; border: 1px solid #252545;
}
.sb-section label { font-size: 11px; color: #8888aa; margin-top: 6px; display: block;
    text-transform: uppercase; letter-spacing: 0.5px; }
.main { margin-left: 300px; padding: 16px 20px; padding-bottom: 40px; }
input, select {
    background: #1e1e30; color: #f0f0f0; border: 1px solid #3a3a5a;
    border-radius: 6px; padding: 7px 10px; width: 100%; font-size: 13px;
}
input:focus { border-color: #4a9eff; outline: none; box-shadow: 0 0 0 2px rgba(74,158,255,0.2); }
input::placeholder { color: #666; }
.Select-control, .Select-menu-outer { background: #1e1e30 !important; }
.Select-value-label, .Select-input input { color: #f0f0f0 !important; }
.Select-placeholder { color: #666 !important; }
.VirtualizedSelectOption { background: #1e1e30; color: #f0f0f0; }
.VirtualizedSelectFocusedOption { background: #3a3a5a !important; }
button {
    background: #4a9eff; color: #fff; border: none; border-radius: 6px;
    padding: 8px 16px; cursor: pointer; width: 100%; margin: 5px 0;
    font-size: 13px; font-weight: 600;
}
button:hover { background: #3a8eef; }
.btn-secondary { background: #2a2a3a; border: 1px solid #3a3a5a; }
.btn-secondary:hover { background: #3a3a5a; }
.crumb-bar {
    background: #12122a; border-radius: 8px; padding: 8px 12px;
    margin-bottom: 12px; border: 1px solid #1f1f3a;
    display: flex; align-items: center; gap: 2px; flex-wrap: wrap;
}
.math-panel {
    background: linear-gradient(135deg, #1a1a2e, #15152a);
    border-radius: 8px; padding: 12px 16px; margin: 8px 0;
    min-height: 30px; border: 1px solid #252545; font-size: 15px;
}
.info-panel {
    background: #12122a; border-radius: 8px; padding: 12px 16px;
    margin: 8px 0; border: 1px solid #1f1f3a; font-size: 13px;
    color: #aab; line-height: 1.6; max-height: 150px; overflow-y: auto;
}
.plot-card {
    background: #12122a; border-radius: 10px; padding: 6px;
    border: 1px solid #1f1f3a; margin-bottom: 10px;
}
.status-bar {
    position: fixed; bottom: 0; left: 280px; right: 0;
    background: #0a0a18; border-top: 1px solid #1f1f3a;
    padding: 4px 16px; font-size: 11px; color: #555;
    display: flex; justify-content: space-between; z-index: 100;
}
</style></head><body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body></html>'''

# ── Layout (continued) ───────────────────────────────────────

app.layout = html.Div([
    # ── Stores ──
    dcc.Store(id="nav-stack", data=json.dumps(init_stack(DEFAULT_MODEL))),
    dcc.Store(id="cache-key", data=""),
    dcc.Store(id="cluster-labels-store", data="{}"),

    # ── Sidebar ──
    html.Div([
        html.H3("🔬 LLM Explorer", style={"margin": "0 0 12px", "fontSize": "18px",
                                            "background": "linear-gradient(90deg,#4a9eff,#a855f7)",
                                            "WebkitBackgroundClip": "text",
                                            "WebkitTextFillColor": "transparent"}),

        html.Div([
            html.Label("Model"),
            dcc.Dropdown(id="model-sel",
                         options=[{"label": v, "value": k} for k, v in MODELS.items()],
                         value=DEFAULT_MODEL, clearable=False,
                         style={"background": "#1e1e30", "color": "#f0f0f0"}),
        ], className="sb-section"),

        html.Div([
            html.Label("Reduction"),
            dcc.Dropdown(id="reduce-sel",
                         options=[{"label": m.upper(), "value": m} for m in ["pca", "umap", "tsne"]],
                         value=DEFAULT_REDUCE, clearable=False,
                         style={"background": "#1e1e30"}),
            html.Label("Clusters (k)"),
            dcc.Slider(id="k-slider", min=2, max=30, step=1, value=DEFAULT_K,
                       marks={i: str(i) for i in range(2, 31, 4)},
                       tooltip={"placement": "bottom"}),
            html.Label("Viz Type"),
            dcc.Dropdown(id="viz-sel",
                         options=[{"label": l, "value": v} for v, l in
                                  [("scatter", "2D Scatter"), ("scatter3d", "3D Scatter"),
                                   ("heatmap", "Heatmap"), ("parallel", "Parallel Coords")]],
                         value=DEFAULT_VIZ, clearable=False,
                         style={"background": "#1e1e30"}),
        ], className="sb-section"),

        html.Div([
            html.Label("Dimension Selection"),
            html.P("e.g. 0-100,200-300,500", style={"fontSize": "10px", "color": "#666", "margin": "2px 0"}),
            dcc.Input(id="dim-range-input", type="text", placeholder="all (leave empty)",
                      debounce=True, style={"marginBottom": "4px"}),
            html.Div(id="dim-range-info", style={"fontSize": "11px", "color": "#8888aa"}),
        ], className="sb-section"),

        html.Div([
            html.Label("LLM Cluster Labels"),
            html.P("Max examples per cluster:", style={"fontSize": "10px", "color": "#666", "margin": "2px 0"}),
            dcc.Input(id="llm-label-max", type="number", value=LLM_LABEL_MAX_EXAMPLES,
                      min=10, max=2000, step=10, style={"marginBottom": "4px"}),
            html.Button("Generate LLM Labels", id="btn-llm-labels", className="btn-secondary"),
            html.Div(id="llm-label-status", style={"fontSize": "11px", "color": "#8888aa", "marginTop": "4px"}),
        ], className="sb-section"),

        html.Div([
            html.Label("Search"),
            dcc.Input(id="search-box", type="text", placeholder="token…", debounce=True),
        ], className="sb-section"),

        html.Div([
            html.Label("Trace / Compare"),
            dcc.Input(id="trace-input", type="text", placeholder="sentence…", debounce=True),
            dcc.Input(id="compare-input", type="text", placeholder="word1, word2, …", debounce=True),
        ], className="sb-section"),

        html.Div([
            html.Button("🔄 Reload", id="btn-reload"),
            html.Button("⬅ Back", id="btn-back", className="btn-secondary"),
        ], className="sb-section"),

    ], className="sidebar"),

    # ── Main ──
    html.Div([
        html.Div(id="crumb-bar", className="crumb-bar"),
        html.Div(id="math-panel", className="math-panel"),
        html.Div(id="info-panel", className="info-panel"),
        html.Div(dcc.Graph(id="main-plot", figure=empty_fig(), config={"scrollZoom": True}),
                 className="plot-card"),
        html.Div(dcc.Graph(id="detail-plot", figure=empty_fig("Detail view"), config={"scrollZoom": True}),
                 className="plot-card"),
        html.Div(dcc.Graph(id="aux-plot", figure=empty_fig("Auxiliary"), config={"scrollZoom": True}),
                 className="plot-card"),
        html.Div(id="status-bar", className="status-bar"),
    ], className="main"),
])

# ══════════════════════════════════════════════════════════════
# SECTION 4: CALLBACKS
# ══════════════════════════════════════════════════════════════

def _load(mn):
    """Load model, tokenizer, embeddings — cached."""
    if mn in _CACHE:
        return _CACHE[mn]
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    con.print(f"[cyan]Loading {mn}…[/cyan]")
    tok = AutoTokenizer.from_pretrained(mn)
    model = AutoModelForCausalLM.from_pretrained(mn, output_hidden_states=True, output_attentions=True)
    model.eval()
    E = model.get_input_embeddings().weight.detach().cpu().numpy()
    voc = {i: tok.decode([i]) for i in range(len(E))}
    con.print(f"[green]✓ {mn}: {E.shape[0]} tokens × {E.shape[1]} dims[/green]")
    _CACHE[mn] = (model, tok, E, voc)
    return model, tok, E, voc


def _compute(mn, method, k, dim_str, cluster_labels_json):
    """Compute reduction + clustering on sampled tokens."""
    model, tok, E, voc = _load(mn)
    V, d = E.shape
    n = min(V, MAX_TOKENS)
    rng = np.random.default_rng(42)
    sidx = rng.choice(V, n, replace=False) if n < V else np.arange(V)
    sidx.sort()
    X = E[sidx]

    # Parse dimension selection
    dim_indices = parse_dim_range(dim_str, d)

    # Reduce — for 3D scatter, request 3 components
    n_components = 3 if True else 2  # always compute 3 so we can switch
    r = reduce(X, n=3, method=method, dim_indices=dim_indices)
    coords3 = r["coords"][:n, :3] if r["coords"].shape[1] >= 3 else np.column_stack(
        [r["coords"][:n], np.zeros(n)])
    coords2 = coords3[:, :2]

    # Cluster on the original (optionally dim-selected) space
    X_for_cluster = X[:, dim_indices] if dim_indices else X
    labels = cluster(X_for_cluster, k=k)
    texts = [voc.get(int(i), "?") for i in sidx]

    # Load existing cluster labels
    try:
        cl = json.loads(cluster_labels_json) if cluster_labels_json else {}
        cl = {int(k2): v2 for k2, v2 in cl.items()}
    except Exception:
        cl = {}

    return model, tok, E, voc, sidx, X, coords2, coords3, labels, texts, r, dim_indices, cl


# ── Dim range info callback ──────────────────────────────────

@app.callback(
    Output("dim-range-info", "children"),
    Input("dim-range-input", "value"),
    State("model-sel", "value"),
)
def update_dim_info(dim_str, mn):
    if not mn:
        return ""
    try:
        _, _, E, _ = _load(mn)
        d = E.shape[1]
        indices = parse_dim_range(dim_str, d)
        if indices is None:
            return f"Using all {d} dimensions"
        return f"Using {len(indices)} of {d} dimensions"
    except Exception:
        return ""


# ── LLM label generation callback ────────────────────────────

@app.callback(
    Output("cluster-labels-store", "data"),
    Output("llm-label-status", "children"),
    Input("btn-llm-labels", "n_clicks"),
    State("model-sel", "value"),
    State("reduce-sel", "value"),
    State("k-slider", "value"),
    State("dim-range-input", "value"),
    State("llm-label-max", "value"),
    State("cluster-labels-store", "data"),
    prevent_initial_call=True,
)
def generate_llm_labels(n_clicks, mn, method, k, dim_str, max_ex, existing_labels):
    if not n_clicks or not mn:
        return no_update, no_update

    max_ex = max_ex or LLM_LABEL_MAX_EXAMPLES

    try:
        model, tok, E, voc, sidx, X, *_ = _compute(mn, method, k, dim_str, existing_labels)

        # Re-cluster to get labels
        dim_indices = parse_dim_range(dim_str, E.shape[1])
        X_for_cluster = X[:, dim_indices] if dim_indices else X
        labels = cluster(X_for_cluster, k=k)

        con.print(f"[cyan]Generating LLM labels for {len(np.unique(labels))} clusters…[/cyan]")
        cl = heuristic_label_all_clusters(labels, voc, sidx, max_examples=max_ex)
        con.print(f"[green]✓ Generated {len(cl)} cluster labels[/green]")

        return json.dumps({str(k2): v2 for k2, v2 in cl.items()}), f"✓ {len(cl)} labels generated"

    except Exception as e:
        con.print(f"[red]LLM labeling error: {e}[/red]")
        return no_update, f"Error: {str(e)[:80]}"


# ── Master callback ──────────────────────────────────────────

@app.callback(
    Output("main-plot", "figure"),
    Output("detail-plot", "figure"),
    Output("aux-plot", "figure"),
    Output("crumb-bar", "children"),
    Output("math-panel", "children"),
    Output("info-panel", "children"),
    Output("nav-stack", "data"),
    Output("status-bar", "children"),
    # Inputs
    Input("btn-reload", "n_clicks"),
    Input("btn-back", "n_clicks"),
    Input("main-plot", "clickData"),
    Input("search-box", "value"),
    Input("trace-input", "value"),
    Input("compare-input", "value"),
    Input("model-sel", "value"),
    Input("reduce-sel", "value"),
    Input("k-slider", "value"),
    Input("viz-sel", "value"),
    Input("dim-range-input", "value"),
    Input("cluster-labels-store", "data"),
    # State
    State("nav-stack", "data"),
)
def master_cb(reload_n, back_n, click, search_q, trace_txt, compare_txt,
              mn, method, k, viz, dim_str, cluster_labels_json, nav_json):
    tid = ctx.triggered_id or "btn-reload"
    stack = json.loads(nav_json) if nav_json else init_stack(mn)

    # ── Back button ──
    if tid == "btn-back":
        stack = nav_pop(stack)
        # If we popped back to model level, just fall through to overview

    # ── Model change → reset stack ──
    if tid == "model-sel":
        stack = init_stack(mn)

    cur = nav_current(stack)

    # ── Load & compute ──
    try:
        model, tok, E, voc, sidx, X, coords2, coords3, labels, texts, r, dim_indices, cl = \
            _compute(mn, method, k, dim_str, cluster_labels_json)
    except Exception as e:
        msg = f"Error: {e}"
        return (empty_fig(msg), empty_fig(), empty_fig(),
                nav_crumbs(stack), msg, "", json.dumps(stack),
                [html.Span(msg)])

    V, d = E.shape
    n = len(sidx)
    stats = manifold_stats(X[:, dim_indices] if dim_indices else X, labels)

    dim_info = f" | dims: {len(dim_indices)}/{d}" if dim_indices else f" | dims: {d}"
    status = [
        html.Span(f"Model: {mn}"),
        html.Span(f"Tokens: {n}/{V}{dim_info}"),
        html.Span(f"Clusters: {k} | Method: {method}"),
        html.Span(level_label(stack)),
    ]

    # ── Handle click → navigate into cluster or token ──
    if tid == "main-plot" and click:
        pt = click["points"][0]
        curve = pt.get("curveNumber", 0)
        pidx = pt.get("pointIndex", pt.get("pointNumber", 0))

        if cur.get("level") == "model":
            # Clicking on overview → go to cluster
            # Determine which cluster was clicked
            unique_sorted = sorted(np.unique(labels))
            if curve < len(unique_sorted):
                cid = unique_sorted[curve]
                clabel = cl.get(cid, f"Cluster {cid}")
                entry = {"level": "cluster", "id": int(cid), "label": f"🔵 {clabel}"}
                stack = nav_push(stack, entry)
            # else ignore

        elif cur.get("level") == "cluster":
            # Clicking inside a cluster → go to token
            cid = cur["id"]
            mask = labels == cid
            cluster_indices = np.where(mask)[0]
            if pidx < len(cluster_indices):
                local_idx = cluster_indices[pidx]
                global_idx = int(sidx[local_idx])
                tok_text = voc.get(global_idx, "?").strip() or f"id:{global_idx}"
                entry = {"level": "token", "id": global_idx, "label": f"🔤 {tok_text}"}
                stack = nav_push(stack, entry)

        # For token level clicks, don't navigate further (prevent infinite loops)

    cur = nav_current(stack)

    # ══════════════════════════════════════════════════════════
    # VIEW: Model overview
    # ══════════════════════════════════════════════════════════
    if cur.get("level") == "model":
        # Search handling
        if tid == "search-box" and search_q:
            results = search(E, voc, search_q, tok)
            info_lines = [f"**Search: '{search_q}'** — {len(results)} results"]
            for idx_r, sim, word in results[:20]:
                info_lines.append(f"`{word.strip()}` (id={idx_r}, sim={sim:.3f})")
            info = html.Div([html.P(l) for l in info_lines])
        else:
            summ = cluster_summary(labels, voc, 6)
            info_lines = []
            for cid_s, exs in sorted(summ.items()):
                clabel = cl.get(cid_s, f"C{cid_s}")
                info_lines.append(f"**{clabel}**: {', '.join(exs)}")
            info = html.Div([dcc.Markdown(l) for l in info_lines])

        # Main plot
        coords = coords3 if viz == "scatter3d" else coords2
        title = f"Token Embedding Space ({method.upper()}, k={k})"
        if viz == "scatter3d":
            main = fig_scatter_3d(coords, labels, texts, title, cluster_labels=cl)
        elif viz == "heatmap":
            sim_mat = compare_tokens(X, list(range(min(100, n))))
            tl = texts[:100]
            main = fig_heatmap(sim_mat, tl, tl, "Pairwise Similarity (first 100)")
        elif viz == "parallel":
            main = fig_parallel(X, labels)
        else:
            main = fig_scatter(coords, labels, texts, title, cluster_labels=cl)

        # Detail: variance / PCA info
        pca_tex = tex_pca(r.get("info", {}))
        detail = fig_bars(E.std(axis=0), "Dimension Std Dev across Vocabulary")

        # Aux: norm distribution
        norms = np.linalg.norm(X, axis=1)
        aux = go.Figure(go.Histogram(x=norms, nbinsx=60, marker_color="#4a9eff"))
        aux = _lay(aux, "Token Norm Distribution", 300)

        math = dcc.Markdown(f"$${tex_overview(E, n)}$$  \n$${pca_tex}$$", mathjax=True)

        return (main, detail, aux, nav_crumbs(stack), math, info,
                json.dumps(stack), status)

    # ══════════════════════════════════════════════════════════
    # VIEW: Cluster detail
    # ══════════════════════════════════════════════════════════
    if cur.get("level") == "cluster":
        cid = cur["id"]
        mask = labels == cid
        c_coords2 = coords2[mask]
        c_coords3 = coords3[mask]
        c_texts = [texts[i] for i in np.where(mask)[0]]
        c_labels = np.zeros(int(mask.sum()), dtype=int)
        c_X = X[mask]

        clabel = cl.get(cid, f"Cluster {cid}")
        title = f"Cluster: {clabel} ({mask.sum()} tokens)"

        if viz == "scatter3d":
            main = fig_scatter_3d(c_coords3, c_labels, c_texts, title)
        else:
            main = fig_scatter(c_coords2, c_labels, c_texts, title)

        # Detail: centroid dimensions
        centroid = c_X.mean(axis=0)
        detail = fig_bars(centroid, f"Centroid of {clabel}", top_k=50)

        # Aux: radar of top dims
        aux = fig_radar(centroid)

        s = stats.get(cid, {})
        math = dcc.Markdown(f"$${tex_cluster(stats, cid)}$$", mathjax=True)

        examples = cluster_examples(labels, voc, cid, 20)
        info = html.Div([
            html.P(f"**{clabel}** — {s.get('size', '?')} tokens, spread={s.get('spread', 0):.3f}"),
            html.P(f"Examples: {', '.join(examples)}"),
        ])

        return (main, detail, aux, nav_crumbs(stack), math, info,
                json.dumps(stack), status)

    # ══════════════════════════════════════════════════════════
    # VIEW: Token detail
    # ══════════════════════════════════════════════════════════
    if cur.get("level") == "token":
        gid = cur["id"]
        vec = E[gid]
        tok_text = voc.get(gid, "?").strip()

        # Main: bar chart of dimensions
        main = fig_bars(vec, f"Embedding of '{tok_text}' (id={gid})")

        # Detail: neighbors
        nb_idx, nb_sims = neighbors(E, gid, 20)
        nb_texts = [voc.get(int(i), "?").strip() for i in nb_idx]
        detail = go.Figure(go.Bar(
            x=nb_texts, y=nb_sims.tolist(),
            marker_color="#4a9eff"))
        detail = _lay(detail, f"Nearest Neighbors of '{tok_text}'", 350)

        # Aux: radar
        aux = fig_radar(vec)

        math = dcc.Markdown(f"$${tex_token(vec)}$$", mathjax=True)

        od = outlier_dims(vec, E, 10)
        info = html.Div([
            html.P(f"**{tok_text}** (id={gid})"),
            html.P(f"Norm: {nrm(vec):.3f}"),
            html.P(f"Top outlier dims: {', '.join(f'd{i}' for i in od)}"),
            html.P(f"Neighbors: {', '.join(nb_texts[:10])}"),
        ])

        return (main, detail, aux, nav_crumbs(stack), math, info,
                json.dumps(stack), status)

    # ══════════════════════════════════════════════════════════
    # VIEW: Trace
    # ══════════════════════════════════════════════════════════
    if cur.get("level") == "trace" or (tid == "trace-input" and trace_txt):
        txt = trace_txt or cur.get("id", "Hello world")
        if tid == "trace-input" and trace_txt:
            entry = {"level": "trace", "id": txt, "label": f"📈 {txt[:20]}…"}
            stack = nav_push(stack, entry)

        acts = activations(model, tok, txt)
        tids = token_ids(tok, txt)
        tok_strs = [tok.decode([t]) for t in tids]

        if len(tids) > 0:
            path = token_path(acts, 0)
            main = fig_path(path, [f"L{i}" for i in range(len(path))])

            dp = delta_path(path)
            delta_norms = np.linalg.norm(dp, axis=1)
            detail = go.Figure(go.Bar(
                x=[f"L{i}→L{i+1}" for i in range(len(delta_norms))],
                y=delta_norms.tolist(), marker_color="#a855f7"))
            detail = _lay(detail, "Layer-to-layer change (‖Δ‖)", 300)
        else:
            main = empty_fig("No tokens")
            detail = empty_fig()

        # Attention
        if len(tids) > 1:
            am = attn_maps(model, tok, txt)
            aux = fig_attn(am[0], tok_strs, head=0, title="Layer 0, Head 0 Attention")
        else:
            aux = empty_fig("Need 2+ tokens for attention")

        math = dcc.Markdown(f"$${tex_delta()}$$", mathjax=True)
        info = html.Div([
            html.P(f"**Tracing:** {txt}"),
            html.P(f"Tokens: {' | '.join(tok_strs)}"),
        ])

        return (main, detail, aux, nav_crumbs(stack), math, info,
                json.dumps(stack), status)

    # ══════════════════════════════════════════════════════════
    # VIEW: Compare
    # ══════════════════════════════════════════════════════════
    if cur.get("level") == "compare" or (tid == "compare-input" and compare_txt):
        txt = compare_txt or cur.get("id", "king, queen")
        if tid == "compare-input" and compare_txt:
            entry = {"level": "compare", "id": txt, "label": f"⚖️ {txt[:20]}…"}
            stack = nav_push(stack, entry)

        words = [w.strip() for w in txt.split(",") if w.strip()]
        ids_list = []
        word_labels = []
        for w in words:
            enc = tok.encode(w)
            if enc:
                ids_list.append(enc[0])
                word_labels.append(voc.get(enc[0], w).strip())

        if len(ids_list) >= 2:
            sim_mat = compare_tokens(E, ids_list)
            main = fig_heatmap(sim_mat, word_labels, word_labels, "Pairwise Cosine Similarity")

            sd = shared_dims(E, ids_list, 20)
            vals = E[ids_list].mean(axis=0)
            detail = fig_bars(vals, "Mean embedding of compared tokens")

            # Individual norms
            norms_c = [nrm(E[i]) for i in ids_list]
            aux = go.Figure(go.Bar(x=word_labels, y=norms_c, marker_color="#4a9eff"))
            aux = _lay(aux, "Token Norms", 300)
        else:
            main = empty_fig("Need 2+ comma-separated tokens")
            detail = empty_fig()
            aux = empty_fig()

        math = ""
        info = html.Div([html.P(f"**Comparing:** {', '.join(word_labels)}")])

        return (main, detail, aux, nav_crumbs(stack), math, info,
                json.dumps(stack), status)

    # ── Fallback ──
    return (empty_fig(), empty_fig(), empty_fig(),
            nav_crumbs(stack), "", "", json.dumps(stack), status)


# ══════════════════════════════════════════════════════════════
# SECTION 5: MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    con.print(Panel.fit(
        "[bold cyan]LLM Vector Space Explorer[/bold cyan]\n"
        "[dim]Navigate to http://localhost:8050[/dim]",
        border_style="cyan"))
    app.run(debug=False, port=8050)
