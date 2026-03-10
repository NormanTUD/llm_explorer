def render(stack, viz_type="scatter", reduce="pca", k=12) -> dict:
    """Takes nav state, returns everything the UI needs."""
    return {
        "fig_main":    go.Figure,       # primary plot
        "fig_secondary": go.Figure,     # bars/radar/path (or None)
        "crumbs":      [{"label":..., "depth":...}],
        "tex":         r"\mathbf{X} \in \mathbb{R}^{50257 \times 768}",
        "info":        "Cluster 3: dog, cat, fish, bird, horse...",
        "options":     ["scatter","heatmap","parallel"],  # valid viz types for this level
    }

