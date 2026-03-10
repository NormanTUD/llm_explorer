#!/usr/bin/env python3
"""LLM Vector Space Explorer — single-file launcher. Run: python main.py"""

# ══════════════════════════════════════════════════════════════
# SECTION 1: VENV BOOTSTRAP
# ══════════════════════════════════════════════════════════════

import sys, os, platform, subprocess, shutil
from pathlib import Path

VENV = Path.home() / ".llm_explorer_venv"
IS_WIN = platform.system() == "Windows"
BIN = VENV / ("Scripts" if IS_WIN else "bin")
PY = BIN / ("python.exe" if IS_WIN else "python")
PIP = BIN / ("pip.exe" if IS_WIN else "pip")

DEPS = [
    "dash", "plotly", "numpy", "scikit-learn",
    "torch --index-url https://download.pytorch.org/whl/cpu",
    "transformers", "umap-learn",
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
            [str(PY), "-c", "import dash, torch, transformers, umap"],
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

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("transformers").setLevel(logging.ERROR)

_CACHE = {}

# ── Layer 0: Atomic ──────────────────────────────────────────

def cosim(a, b):
    return np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9)

def nrm(v):
    return float(np.linalg.norm(v))

def topk(vals, k=10):
    return np.argsort(vals)[-k:][::-1]

def csim_batch(X, v):
    return X @ v / (np.linalg.norm(X, axis=1) * nrm(v) + 1e-9)

# ── Layer 1: Model access ───────────────────────────────────

MODELS = {
    "gpt2": "GPT-2 (117M)",
    "gpt2-medium": "GPT-2 Medium (345M)",
    "gpt2-large": "GPT-2 Large (774M)",
    "distilgpt2": "DistilGPT-2 (82M)",
    "EleutherAI/pythia-70m": "Pythia 70M",
    "EleutherAI/pythia-160m": "Pythia 160M",
}

def load(name="gpt2"):
    if name not in _CACHE:
        from transformers import AutoModel, AutoTokenizer
        print(f"  Loading {name} (cached at ~/.cache/huggingface/) ...")
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModel.from_pretrained(name, output_attentions=True)
        mdl.eval()
        _CACHE[name] = (mdl, tok)
        print(f"  Done: {name}")
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
        if w is not None:
            return w
    return next((p.detach().cpu().numpy() for n, p in model.named_parameters()
                 if "embed" in n.lower() and p.dim() == 2), None)

def get_heads(W, nh):
    return W.reshape(W.shape[0], nh, -1).transpose(1, 0, 2)

def cfg(model, *keys):
    for k in keys:
        v = getattr(model.config, k, None)
        if v is not None:
            return v
    return 12

def n_heads(model):  return cfg(model, "n_head", "num_attention_heads")
def n_layers(model): return cfg(model, "n_layer", "num_hidden_layers")

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
        return {"coords": TSNE(n_components=n, perplexity=min(30, len(X) - 1))
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
    if not ids:
        return []
    sims = csim_batch(X, X[ids[0]])
    top = topk(sims, 50)
    return [(int(i), float(sims[i]), voc.get(i, "?")) for i in top]

def cluster_examples(labels, voc, cid, n=10):
    idxs = np.where(labels == cid)[0]
    sel = idxs[:n] if len(idxs) <= n else np.random.choice(idxs, n, replace=False)
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

# ── Layer 4: Comparison ──────────────────────────────────────

def compare_tokens(X, ids):
    V = X[ids]
    n = np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
    return (V @ V.T) / (n @ n.T)

def shared_dims(X, ids, k=20):
    V = X[ids]
    ratio = np.abs(V.mean(0)) / (V.std(0) + 1e-9)
    return topk(ratio, k)

def dim_profile(X, idx):
    return X[idx]

def outlier_dims(vec, X, k=20):
    z = np.abs(vec - X.mean(0)) / (X.std(0) + 1e-9)
    return topk(z, k)

# ── TeX strings ──────────────────────────────────────────────

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
    return f"n={s.get('size', '?')},\\; \\bar{{d}}_{{intra}}={s.get('spread', 0):.3f}"

def tex_delta():
    return "\\Delta_l = \\mathbf{h}^{(l)} - \\mathbf{h}^{(l-1)}"

# ══════════════════════════════════════════════════════════════
# SECTION 3: DASH APP
# ══════════════════════════════════════════════════════════════

import json
import dash
from dash import dcc, html, Input, Output, State, callback, no_update, ctx, ALL
import plotly.graph_objects as go
import plotly.express as px

DEFAULT_MODEL = "gpt2"
DEFAULT_K = 12
DEFAULT_VIZ = "scatter"
DEFAULT_REDUCE = "pca"
EMPTY_FIG = go.Figure().update_layout(
    template="plotly_dark", height=300,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    annotations=[dict(text="No data yet", showarrow=False,
                       font=dict(size=16, color="#555"), xref="paper", yref="paper", x=0.5, y=0.5)]
)

# ── Navigation ────────────────────────────────────────────────

def nav_push(stack, entry):
    return stack + [entry]

def nav_pop(stack, to=-1):
    return stack[:to] if to > 0 else stack[:max(len(stack) - 1, 1)]

def nav_current(stack):
    return stack[-1] if stack else {"level": "model", "id": DEFAULT_MODEL, "label": "GPT-2"}

LEVEL_ICONS = {"model": "🏠", "cluster": "🔵", "token": "🔤",
               "trace": "📈", "attention": "👁", "compare": "⚖️"}

def nav_crumbs(stack):
    spans = []
    for i, s in enumerate(stack):
        icon = LEVEL_ICONS.get(s.get("level", ""), "")
        is_last = i == len(stack) - 1
        spans.append(html.Span(
            f"{icon} {s['label']}",
            id={"type": "crumb", "index": i},
            style={"cursor": "pointer", "padding": "4px 10px", "margin": "0 2px",
                   "borderRadius": "6px", "fontSize": "13px",
                   "background": "#4a9eff" if is_last else "#2a2a3a",
                   "color": "#fff", "fontWeight": "bold" if is_last else "normal",
                   "display": "inline-block"}
        ))
        if not is_last:
            spans.append(html.Span(" > ", style={"color": "#555"}))
    return spans

def level_label(stack):
    cur = nav_current(stack)
    return f"Depth {len(stack)} - {cur.get('level', 'model').title()} view"

def init_stack(model_name):
    return [{"level": "model", "id": model_name, "label": MODELS.get(model_name, model_name)}]

# ── Figures ───────────────────────────────────────────────────

def _layout(fig, title="", h=550):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_dark", height=h,
        paper_bgcolor="rgba(13,13,26,0.8)", plot_bgcolor="rgba(26,26,46,0.6)",
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0.3)"),
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=12),
    )
    return fig

def fig_scatter(coords, labels, texts, title=""):
    fig = go.Figure()
    for c in sorted(np.unique(labels)):
        mask = labels == c
        name = f"Cluster {c}" if c >= 0 else "Noise"
        fig.add_trace(go.Scattergl(
            x=coords[mask, 0], y=coords[mask, 1], mode="markers",
            marker=dict(size=5, opacity=0.65),
            text=[texts[i] for i in np.where(mask)[0]],
            hovertemplate="%{text}<extra>" + name + "</extra>",
            name=name, customdata=np.where(mask)[0],
        ))
    return _layout(fig, title)

def fig_heatmap(matrix, xlabels, ylabels, title=""):
    fig = go.Figure(go.Heatmap(
        z=matrix, x=xlabels, y=ylabels, colorscale="Viridis",
        hovertemplate="x=%{x}<br>y=%{y}<br>val=%{z:.3f}<extra></extra>"
    ))
    return _layout(fig, title, 500).update_layout(margin=dict(l=80, b=80))

def fig_bars(vec, title="Raw Dimensions", top_k=50):
    idx = topk(np.abs(vec), top_k)
    fig = go.Figure(go.Bar(
        x=[f"d{i}" for i in idx], y=vec[idx],
        marker_color=["#4a9eff" if v > 0 else "#ff6b6b" for v in vec[idx]]
    ))
    return _layout(fig, title, 300).update_layout(margin=dict(l=30, r=10, t=35, b=30))

def fig_radar(vec, dim_labels=None, top_k=12):
    idx = topk(np.abs(vec), top_k)
    labs = [dim_labels[i] if dim_labels else f"d{i}" for i in idx]
    fig = go.Figure(go.Scatterpolar(
        r=np.abs(vec[idx]), theta=labs, fill="toself",
        marker=dict(color="#4a9eff"), line=dict(color="#4a9eff")
    ))
    return _layout(fig, h=350).update_layout(
        polar=dict(bgcolor="#1a1a2e", radialaxis=dict(visible=True, color="#555")),
        margin=dict(l=40, r=40, t=30, b=30))

def fig_parallel(X, labels, dims=None, k=15):
    if dims is None:
        dims = topk(X.std(axis=0), k)
    df_cols = {f"d{d}": X[:, d] for d in dims}
    df_cols["cluster"] = labels
    fig = px.parallel_coordinates(
        df_cols, color="cluster", dimensions=[f"d{d}" for d in dims],
        color_continuous_scale="Viridis"
    )
    return _layout(fig, h=450).update_layout(margin=dict(l=40, r=40))

def fig_path(points, layer_labels):
    r = reduce(points, n=2, method="pca")
    c = r["coords"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=c[:, 0], y=c[:, 1], mode="lines+markers+text",
        text=layer_labels, textposition="top center", textfont=dict(size=10),
        marker=dict(size=9, color=list(range(len(c))), colorscale="Plasma",
                    showscale=True, colorbar=dict(title="Layer", thickness=10)),
        line=dict(color="#666", dash="dot", width=1)
    ))
    return _layout(fig, "Token path through layers", 450)

def fig_attn(attn_matrix, tokens, head=0, title=""):
    m = attn_matrix[head] if attn_matrix.ndim == 3 else attn_matrix
    return fig_heatmap(m, tokens, tokens, title=title or f"Attention Head {head}")

# ── Render engine ─────────────────────────────────────────────

def render(stack, viz, red, k, model_name, trace_text="",
           attn_layer=0, attn_head=0, trace_pos=0):
    cur = nav_current(stack)
    mdl, tok = load(model_name)
    E = embed_matrix(mdl)
    voc = vocab(tok)
    tex = tex_overview(E)
    figs = {"main": EMPTY_FIG, "detail": EMPTY_FIG, "bars": EMPTY_FIG}
    info = ""

    if cur["level"] == "model":
        r = reduce(E, method=red)
        labs = cluster(E, k=k)
        summary = cluster_summary(labs, voc)
        hover = [", ".join(summary.get(int(labs[i]), ["?"]))[:60] for i in range(len(E))]
        figs["main"] = fig_scatter(r["coords"], labs, hover,
                                    f"Embedding Space -- {model_name}")
        tex += " \\quad " + tex_pca(r.get("info", {}))
        st = manifold_stats(E, labs)
        lines = [f"**C{cid}** ({st[cid]['size']} tok, spread {st[cid]['spread']:.2f}): "
                 f"{', '.join(summary.get(cid, []))}" for cid in sorted(summary)]
        info = "  \n".join(lines[:15])
        if viz == "parallel":
            figs["detail"] = fig_parallel(E, labs)

    elif cur["level"] == "cluster":
        cid = cur["id"]
        labs = cluster(E, k=k)
        mask = np.where(labs == cid)[0]
        sub = E[mask]
        r = reduce(sub, method=red)
        hover = [voc.get(int(mask[i]), "?") for i in range(len(sub))]
        figs["main"] = fig_scatter(r["coords"], np.zeros(len(sub), dtype=int), hover,
                                    f"Cluster {cid} -- {len(mask)} tokens")
        st = manifold_stats(E, labs)
        tex = tex_cluster(st, cid)
        examples = cluster_examples(labs, voc, cid, 20)
        info = f"**Cluster {cid}** -- {len(mask)} tokens, spread {st[cid]['spread']:.3f}  \n"
        info += f"Examples: {', '.join(examples)}"
        if viz == "parallel":
            figs["detail"] = fig_parallel(sub, np.zeros(len(sub), dtype=int))

    elif cur["level"] == "token":
        tid = cur["id"]
        vec = E[tid]
        nbr_idx, nbr_sim = neighbors(E, tid, 30)
        sub = E[np.concatenate([[tid], nbr_idx])]
        r = reduce(sub, method=red)
        hover = [voc.get(int(i), "?") for i in np.concatenate([[tid], nbr_idx])]
        labs = np.array([1] + [0] * len(nbr_idx))
        figs["main"] = fig_scatter(r["coords"], labs, hover,
                                    f"Neighbors of '{voc.get(tid, '?').strip()}'")
        figs["bars"] = fig_bars(vec) if viz in ("scatter", "bars") else fig_radar(vec)
        tex = tex_token(vec)
        od = outlier_dims(vec, E)
        tex += f" \\quad \\text{{outlier dims: }}{list(od[:5])}"
        nlist = [f"{voc.get(int(nbr_idx[i]), '?').strip()} ({nbr_sim[i]:.3f})"
                 for i in range(min(15, len(nbr_idx)))]
        info = f"**'{voc.get(tid, '?').strip()}'** -- top neighbors:  \n" + ", ".join(nlist)

    elif cur["level"] == "trace":
        acts = activations(mdl, tok, trace_text)
        tids = token_ids(tok, trace_text)
        tidx = min(trace_pos, len(tids) - 1)
        path = token_path(acts, tidx)
        layer_labels = [f"L{i}" for i in range(len(path))]
        figs["main"] = fig_path(path, layer_labels)
        figs["bars"] = fig_bars(delta_path(path).mean(0), "Mean delta across layers")
        tex = tex_delta()
        tokens_str = [voc.get(t, "?").strip() for t in tids]
        info = f"Tracing **'{tokens_str[tidx]}'** (pos {tidx}) through {len(path)} layers  \n"
        info += f"Tokens: {' | '.join(f'`{t}`' for t in tokens_str)}"

    elif cur["level"] == "attention":
        am = attn_maps(mdl, tok, trace_text)
        tids = token_ids(tok, trace_text)
        tokens = [voc.get(t, "?").strip() for t in tids]
        layer = min(attn_layer, len(am) - 1)
        head_idx = min(attn_head, am[layer].shape[0] - 1)
        figs["main"] = fig_attn(am[layer], tokens, head_idx,
                                 f"Layer {layer} Head {head_idx}")
        tex = f"\\text{{Layer }}={layer},\\; \\text{{Head }}={head_idx}"
        info = f"**Attention** -- Layer {layer}/{len(am)-1}, Head {head_idx}  \n"
        info += f"Tokens: {' -> '.join(tokens)}"

    elif cur["level"] == "compare":
        ids = cur.get("ids", [])
        if len(ids) >= 2:
            sim_mat = compare_tokens(E, ids)
            labels = [voc.get(int(i), "?").strip() for i in ids]
            figs["main"] = fig_heatmap(sim_mat, labels, labels, "Token Similarity")
            sd = shared_dims(E, ids)
            figs["bars"] = fig_bars(E[ids].mean(0), f"Shared profile -- top dims: {list(sd[:5])}")
            tex = f"\\text{{Comparing }} {len(ids)} \\text{{ tokens}}"
            info = f"**Comparing:** {', '.join(labels)}  \n"
            info += f"Shared dims (high agreement): {list(sd[:10])}"

    return figs, tex, info

# ── Layout ────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.index_string = '''<!DOCTYPE html>
<html><head>{%metas%}<title>LLM Vector Space Explorer</title>{%css%}<style>
* { box-sizing: border-box; }
body { background: #0d0d1a; color: #eee; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; }
.sidebar {
    width: 260px; padding: 16px; background: #12122a;
    min-height: 100vh; position: fixed; overflow-y: auto;
    border-right: 1px solid #1f1f3a;
}
.sb-section {
    background: #1a1a2e; border-radius: 8px; padding: 10px 12px;
    margin-bottom: 10px; border: 1px solid #252545;
}
.sb-section label { font-size: 11px; color: #8888aa; margin-top: 6px; display: block;
    text-transform: uppercase; letter-spacing: 0.5px; }
.main { margin-left: 280px; padding: 16px 20px; }
input, select {
    background: #2a2a3a; color: #eee; border: 1px solid #3a3a5a;
    border-radius: 6px; padding: 7px 10px; width: 100%; font-size: 13px;
}
input:focus { border-color: #4a9eff; outline: none; }
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
    position: fixed; bottom: 0; left: 260px; right: 0;
    background: #0a0a18; border-top: 1px solid #1f1f3a;
    padding: 4px 16px; font-size: 11px; color: #555;
    display: flex; justify-content: space-between; z-index: 100;
}
</style></head><body>{%app_entry%}{%config%}{%scripts%}{%renderer%}</body></html>'''

def _section(title, children):
    return html.Div(className="sb-section", children=[
        html.Div(title, style={"fontSize": "12px", "fontWeight": "700",
                                "color": "#6a6a9a", "marginBottom": "6px"}),
        *children
    ])

sidebar = html.Div(className="sidebar", children=[
    html.H3("🔬 LLM Explorer", style={"margin": "0 0 4px", "fontSize": "18px"}),
    html.Div("Vector Space Interpretability", style={"fontSize": "10px", "color": "#555",
              "marginBottom": "14px", "letterSpacing": "1px", "textTransform": "uppercase"}),

    _section("Model", [
        html.Label("Select model"),
        dcc.Dropdown(id="model-sel",
            options=[{"label": v, "value": k} for k, v in MODELS.items()],
            value=DEFAULT_MODEL, clearable=False,
            style={"background": "#2a2a3a", "color": "#000", "fontSize": "12px"}),
    ]),

    _section("Visualization", [
        html.Label("Plot type"),
        dcc.RadioItems(id="viz-type",
            options=[{"label": v, "value": k} for k, v in
                     {"scatter": "Scatter", "heatmap": "Heatmap",
                      "parallel": "Parallel", "radar": "Radar", "bars": "Bars"}.items()],
            value=DEFAULT_VIZ, style={"fontSize": "12px"},
            labelStyle={"display": "block", "padding": "2px 0"}),
        html.Label("Reduction"),
        dcc.RadioItems(id="reduce-method",
            options=[{"label": v, "value": k} for k, v in
                     {"pca": "PCA", "umap": "UMAP", "tsne": "t-SNE"}.items()],
            value=DEFAULT_REDUCE, style={"fontSize": "12px"},
            labelStyle={"display": "inline-block", "marginRight": "12px"}),
        html.Label("Clusters (k)"),
        dcc.Slider(id="k-slider", min=3, max=30, step=1, value=DEFAULT_K,
                   marks={i: str(i) for i in [3, 8, 12, 20, 30]},
                   tooltip={"placement": "bottom"}),
    ]),

    _section("Search", [
        html.Label("Find token in embedding space"),
        dcc.Input(id="search-box", placeholder="type a word...", debounce=True),
        html.Button("Search", id="search-btn", n_clicks=0),
        html.Div(id="search-results", style={"fontSize": "11px", "color": "#8888aa",
                  "marginTop": "4px", "maxHeight": "80px", "overflowY": "auto"}),
    ]),

    _section("Trace & Attention", [
        html.Label("Input text"),
        dcc.Input(id="trace-box", placeholder="The cat sat on", debounce=True),
        html.Div(style={"display": "flex", "gap": "4px"}, children=[
            html.Button("Trace", id="trace-btn", n_clicks=0, style={"flex": "1"}),
            html.Button("Attention", id="attn-btn", n_clicks=0, style={"flex": "1"}),
        ]),
        html.Label("Token position"),
        dcc.Slider(id="trace-pos", min=0, max=20, step=1, value=0,
                   marks=None, tooltip={"placement": "bottom"}),
        html.Label("Layer"),
        dcc.Slider(id="attn-layer", min=0, max=11, step=1, value=0,
                   marks=None, tooltip={"placement": "bottom"}),
        html.Label("Head"),
        dcc.Slider(id="attn-head", min=0, max=11, step=1, value=0,
                   marks=None, tooltip={"placement": "bottom"}),
    ]),

    _section("Compare", [
        html.Label("Comma-separated tokens"),
        dcc.Input(id="compare-box", placeholder="king,queen,man,woman", debounce=True),
        html.Button("Compare", id="compare-btn", n_clicks=0),
    ]),

    html.Button("Back", id="back-btn", n_clicks=0, className="btn-secondary",
                style={"marginTop": "8px"}),
])

main_area = html.Div(className="main", children=[
    html.Div(className="crumb-bar", children=[
        html.Div(id="breadcrumbs", style={"display": "flex", "alignItems": "center",
                                            "gap": "2px", "flexWrap": "wrap"}),
        html.Span(id="depth-tag", style={"marginLeft": "auto", "fontSize": "11px",
                                          "color": "#666", "background": "#1a1a2e",
                                          "padding": "2px 8px", "borderRadius": "4px"}),
    ]),
    dcc.Markdown(id="math-panel", className="math-panel", mathjax=True,
                 style={"minHeight": "30px"}),
    html.Div(id="info-panel", className="info-panel", style={"display": "none"}),
    html.Div(className="plot-card", children=[
        dcc.Loading(type="circle", color="#4a9eff", children=[
            dcc.Graph(id="main-plot", config={"scrollZoom": True}),
        ])
    ]),
    html.Div(style={"display": "flex", "gap": "10px"}, children=[
        html.Div(className="plot-card", style={"flex": "1"}, children=[
            dcc.Loading(type="dot", color="#4a9eff", children=[
                dcc.Graph(id="detail-plot"),
            ])
        ]),
        html.Div(className="plot-card", style={"flex": "1"}, children=[
            dcc.Loading(type="dot", color="#4a9eff", children=[
                dcc.Graph(id="bars-plot"),
            ])
        ]),
    ]),
    dcc.Store(id="nav-store", data=json.dumps(init_stack(DEFAULT_MODEL))),
    html.Div(className="status-bar", children=[
        html.Span(id="status-left", children="Ready"),
        html.Span(id="status-right", children="LLM Vector Space Explorer v0.1"),
    ]),
])

app.layout = html.Div([sidebar, main_area])

# ── Callback ──────────────────────────────────────────────────

@callback(
    Output("nav-store", "data"),
    Output("main-plot", "figure"),
    Output("detail-plot", "figure"),
    Output("bars-plot", "figure"),
    Output("breadcrumbs", "children"),
    Output("depth-tag", "children"),
    Output("math-panel", "children"),
    Output("info-panel", "children"),
    Output("info-panel", "style"),
    Output("search-results", "children"),
    Output("status-left", "children"),
    Output("attn-layer", "max"),
    Output("attn-head", "max"),
    Output("trace-pos", "max"),
    # Triggers
    Input("main-plot", "clickData"),
    Input("back-btn", "n_clicks"),
    Input("search-btn", "n_clicks"),
    Input("trace-btn", "n_clicks"),
    Input("attn-btn", "n_clicks"),
    Input("compare-btn", "n_clicks"),
    Input("model-sel", "value"),
    Input("viz-type", "value"),
    Input("reduce-method", "value"),
    Input("k-slider", "value"),
    Input("attn-layer", "value"),
    Input("attn-head", "value"),
    Input("trace-pos", "value"),
    # State
    State("nav-store", "data"),
    State("search-box", "value"),
    State("trace-box", "value"),
    State("compare-box", "value"),
    prevent_initial_call=False,
)
def master_callback(click, back_n, search_n, trace_n, attn_n, compare_n,
                    model_name, viz, red, k, attn_layer_val, attn_head_val, trace_pos_val,
                    nav_json, search_q, trace_txt, compare_txt):

    stack = json.loads(nav_json) if nav_json else init_stack(model_name or DEFAULT_MODEL)
    triggered = ctx.triggered_id if ctx.triggered_id else "init"
    search_res = no_update
    mn = model_name or DEFAULT_MODEL

    mdl, tok = load(mn)
    E = embed_matrix(mdl)
    voc = vocab(tok)

    max_layer = n_layers(mdl) - 1
    max_head = n_heads(mdl) - 1
    max_trace = 20

    # ── Navigation triggers ──
    if triggered == "model-sel":
        stack = init_stack(mn)

    elif triggered == "back-btn":
        stack = nav_pop(stack)

    elif triggered == "main-plot" and click:
        pt = click["points"][0]
        cur = nav_current(stack)
        if cur["level"] == "model":
            labs = cluster(E, k=k)
            idx = pt.get("customdata", pt.get("pointIndex", 0))
            cid = int(labs[int(idx)])
            ex = cluster_examples(labs, voc, cid, 5)
            stack = nav_push(stack, {"level": "cluster", "id": cid,
                "label": f"C{cid}: {', '.join(ex)}"})
        elif cur["level"] == "cluster":
            cid = cur["id"]
            labs = cluster(E, k=k)
            mask = np.where(labs == cid)[0]
            li = int(pt.get("customdata", pt.get("pointIndex", 0)))
            tid = int(mask[li]) if li < len(mask) else int(mask[0])
            stack = nav_push(stack, {"level": "token", "id": tid,
                "label": f"'{voc.get(tid, '?').strip()}'"})
        elif cur["level"] == "token":
            nbr_idx, _ = neighbors(E, cur["id"], 30)
            all_ids = np.concatenate([[cur["id"]], nbr_idx])
            li = int(pt.get("customdata", pt.get("pointIndex", 0)))
            new_tid = int(all_ids[li]) if li < len(all_ids) else cur["id"]
            stack = nav_push(stack, {"level": "token", "id": new_tid,
                "label": f"'{voc.get(new_tid, '?').strip()}'"})

    elif triggered == "search-btn" and search_q:
        results = search(E, voc, search_q, tok)
        if results:
            tid = results[0][0]
            stack = nav_push(stack, {"level": "token", "id": tid,
                "label": f"'{voc.get(tid, '?').strip()}'"})
            search_res = [html.Div(f"{r[2].strip()} ({r[1]:.3f})",
                          style={"padding": "1px 0", "borderBottom": "1px solid #1f1f3a"})
                          for r in results[:8]]
        else:
            search_res = [html.Div("No results", style={"color": "#f88"})]

    elif triggered == "trace-btn" and trace_txt:
        tids = token_ids(tok, trace_txt)
        max_trace = max(len(tids) - 1, 0)
        stack = nav_push(stack, {"level": "trace", "id": trace_txt,
            "label": f"Trace: '{trace_txt[:20]}'"})

    elif triggered == "attn-btn" and trace_txt:
        stack = nav_push(stack, {"level": "attention", "id": trace_txt,
            "label": f"Attn: '{trace_txt[:20]}'"})

    elif triggered == "compare-btn" and compare_txt:
        words = [w.strip() for w in compare_txt.split(",") if w.strip()]
        ids = [tok.encode(w)[0] for w in words if tok.encode(w)]
        if len(ids) >= 2:
            labs = [voc.get(i, "?").strip() for i in ids]
            stack = nav_push(stack, {"level": "compare", "ids": ids,
                "label": f"Compare: {', '.join(labs[:4])}"})

    # Update trace pos max
    cur = nav_current(stack)
    if cur.get("level") == "trace" and trace_txt:
        max_trace = max(len(token_ids(tok, trace_txt)) - 1, 0)

    # ── Render ──
    figs, tex, info = render(stack, viz, red, k, mn, trace_txt or "",
                              attn_layer_val or 0, attn_head_val or 0, trace_pos_val or 0)

    math_str = f"$${tex}$$" if tex else ""
    crumbs = nav_crumbs(stack)
    depth = level_label(stack)
    status = f"Model: {mn} | {E.shape[0]} tokens x {E.shape[1]}d | {cur.get('level', '?')}"
    info_style = {"display": "block"} if info else {"display": "none"}

    return (json.dumps(stack),
            figs["main"], figs.get("detail", EMPTY_FIG), figs.get("bars", EMPTY_FIG),
            crumbs, depth, math_str,
            dcc.Markdown(info) if info else "", info_style,
            search_res, status,
            max_layer, max_head, max_trace)

# ══════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  LLM Vector Space Explorer")
    print("  http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)

