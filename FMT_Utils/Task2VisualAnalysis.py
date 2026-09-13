"""Label-free, linked geometry/latent exploration for Task2.

One row is one primitive, never one neighbor line. DBSCAN labels are anonymous;
noise is -1. The analysis subset is drawn before embedding or clustering.
"""
from __future__ import annotations

import colorsys
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def physical_paths(raw, seeds):
    """Undo Build_Task2_Universality_Cache._raw_local_features translation."""
    raw, seeds = np.asarray(raw), np.asarray(seeds)
    if raw.ndim != 2 or raw.shape[1] % 21 or raw.shape[1] < 42:
        raise ValueError("raw must contain flattened 7-line 3D primitives")
    if seeds.shape != (len(raw), 3):
        raise ValueError("seed and primitive row counts differ")
    local = raw.reshape(len(raw), 7, -1, 3)
    if not np.allclose(local[:, 0, 0], 0, atol=1e-5):
        raise ValueError("expected cache coordinates relative to the center seed")
    result = local + seeds[:, None, None, :]
    if not np.isfinite(result).all():
        raise ValueError("non-finite geometry")
    return result.astype(np.float32)


def load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        bundle = {key: data[key].copy() for key in data.files}
    metadata = json.loads(str(bundle.pop("metadata_json")))
    required = {"paths", "sample_ids", "raw_mu", "fmt_mu"}
    if not required.issubset(bundle):
        raise ValueError(f"missing bundle arrays: {sorted(required - bundle.keys())}")
    paths = bundle["paths"]
    n = len(paths)
    if paths.ndim != 4 or paths.shape[1] != 7 or paths.shape[2] < 2 or paths.shape[3] != 3:
        raise ValueError("paths must have shape (N, 7, T>=2, 3)")
    if n < 4 or not np.isfinite(paths).all():
        raise ValueError("need at least four finite primitives")
    ids = bundle["sample_ids"]
    if ids.shape != (n,) or ids.dtype.kind not in "iu" or len(np.unique(ids)) != n:
        raise ValueError("sample_ids must be unique integers, one per primitive")
    for arm in ("raw", "fmt"):
        mu = bundle[f"{arm}_mu"]
        if mu.ndim != 2 or mu.shape[0] != n or mu.shape[1] < 1 or not np.isfinite(mu).all():
            raise ValueError(f"invalid {arm} latent means")
    if bundle["raw_mu"].shape != bundle["fmt_mu"].shape:
        raise ValueError("paired same-VAE arms must have the same latent dimension")
    return bundle, metadata


def sample_rows(n, maximum, seed):
    if maximum < 4:
        raise ValueError("max_samples must be at least four")
    return np.sort(np.random.default_rng(seed).choice(n, min(n, maximum), replace=False))


def latent_values(mu, normalization):
    if normalization == "none":
        return np.asarray(mu, dtype=np.float64)
    if normalization not in {"l1", "l2"}:
        raise ValueError("latent_normalization must be none, l1 or l2")
    return normalize(np.asarray(mu, dtype=np.float64), norm=normalization)


def embed(mu, settings):
    """t-SNE sees no cluster labels, geometry, or reference labels."""
    values = latent_values(mu, settings["latent_normalization"])
    method = settings.get("projection", "tsne")
    if method == "umap":
        from umap import UMAP
        neighbors = settings.get("umap_neighbors", 15)
        minimum = settings.get("umap_min_dist", .1)
        if not np.isfinite(neighbors) or int(neighbors) != neighbors or not 2 <= neighbors < len(values):
            raise ValueError("UMAP n_neighbors must be an integer in [2, sample count)")
        if not np.isfinite(minimum) or not 0 <= minimum <= 1:
            raise ValueError("UMAP min_dist must be in [0, 1]")
        if np.all(np.ptp(values, axis=0) == 0):
            raise ValueError("all latent vectors are identical")
        with threadpool_limits(limits=4):
            xy = UMAP(n_components=2, n_neighbors=int(neighbors), min_dist=float(minimum),
                      metric="euclidean", random_state=int(settings["tsne_seed"]), n_jobs=1).fit_transform(values)
        if not np.isfinite(xy).all():
            raise ValueError("UMAP produced non-finite coordinates")
        return values, xy, None
    if method != "tsne":
        raise ValueError("projection must be tsne or umap")
    perplexity = float(settings["perplexity"])
    iterations = int(settings["tsne_iterations"])
    if not np.isfinite(perplexity) or not 0 < perplexity < len(values):
        raise ValueError("perplexity must be positive and smaller than the analysis sample count")
    if iterations < 250:
        raise ValueError("tsne_iterations must be at least 250")
    if np.all(np.ptp(values, axis=0) == 0):
        raise ValueError("all latent vectors are identical; a t-SNE view would be misleading")
    with threadpool_limits(limits=4):
        model = TSNE(n_components=2, perplexity=perplexity,
                     init="random", learning_rate="auto", metric="euclidean",
                     random_state=int(settings["tsne_seed"]), max_iter=iterations)
        xy = model.fit_transform(values)
    if not np.isfinite(xy).all():
        raise ValueError("t-SNE produced non-finite coordinates")
    return values, xy, float(model.kl_divergence_)


def cluster(values, xy, space, eps, min_samples, method="dbscan", n_clusters=8, seed=7068):
    if space not in {"latent", "tsne", "projection", "umap"}:
        raise ValueError("cluster_space must be latent or tsne")
    x = values if space == "latent" else xy
    if method == "kmeans":
        if n_clusters is None or not np.isfinite(n_clusters) or int(n_clusters) != n_clusters or not 1 <= n_clusters <= len(x):
            raise ValueError("K must be an integer in [1, sample count]")
        if len(np.unique(x, axis=0)) < n_clusters:
            raise ValueError("K exceeds the number of distinct vectors")
        with threadpool_limits(limits=4):
            labels = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=int(seed)).fit_predict(x)
        # Core-point status is a DBSCAN concept, not applicable to KMeans.
        return labels.astype(np.int32), np.zeros(len(x), dtype=bool)
    if method != "dbscan":
        raise ValueError("cluster method must be dbscan or kmeans")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if isinstance(min_samples, bool) or not np.isfinite(min_samples) or int(min_samples) != min_samples or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    x = values if space == "latent" else xy
    with threadpool_limits(limits=4):
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples),
                       metric="euclidean", n_jobs=1).fit(x)
    core = np.zeros(len(x), dtype=bool)
    core[model.core_sample_indices_] = True
    return model.labels_.astype(np.int32), core


def color(label):
    if int(label) == -1:
        return "#9ca3af"
    rgb = colorsys.hsv_to_rgb((int(label) * .61803398875 + .04) % 1, .67, .78)
    return "#" + "".join(f"{round(v * 255):02x}" for v in rgb)


def cluster_name(label):
    return "Noise (-1)" if int(label) == -1 else f"Cluster {label}"


def figures(paths, sample_ids, xy, labels, *, chosen=None, clusters=None,
            hidden_clusters=None, geometry="center", coordinates="physical", revision="analysis", projection_method="tsne"):
    import plotly.graph_objects as go

    visible = np.ones(len(paths), dtype=bool)
    hidden = np.isin(labels, hidden_clusters or [])
    visible &= ~hidden
    if clusters is not None:
        visible &= np.isin(labels, clusters)
    if chosen is not None:
        visible &= np.isin(sample_ids, chosen)
    displayed = paths.copy()
    if coordinates == "centered":
        displayed -= displayed[:, :1, :1, :]
    elif coordinates != "physical":
        raise ValueError("coordinates must be physical or centered")
    if geometry == "center":
        displayed = displayed[:, :1]
    elif geometry != "primitive":
        raise ValueError("geometry must be center or primitive")
    spatial, projection = go.Figure(), go.Figure()
    for label in np.unique(labels):
        rows = np.flatnonzero((labels == label) & visible)
        if not len(rows):
            continue
        p = displayed[rows]
        n, lines, steps, _ = p.shape
        # NaN separators prevent false connections between lines/primitives.
        xyz = np.concatenate((p, np.full((n, lines, 1, 3), np.nan)), axis=2).reshape(-1, 3)
        ids = np.repeat(sample_ids[rows], lines * (steps + 1))
        spatial.add_trace(go.Scatter3d(
            x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), z=xyz[:, 2].tolist(),
            mode="lines", line={"color": color(label), "width": 3},
            name=cluster_name(label), customdata=ids.tolist(),
            hovertemplate="Primitive %{customdata}<extra>%{fullData.name}</extra>"))
    # All points remain available for brushing; excluded ones provide context.
    for label in np.unique(labels):
        rows = np.flatnonzero((labels == label) & ~hidden)
        if not len(rows):
            continue
        projection.add_trace(go.Scattergl(
            x=xy[rows, 0].tolist(), y=xy[rows, 1].tolist(), mode="markers",
            marker={"color": color(label), "size": 6,
                    "opacity": np.where(visible[rows], .9, .12).tolist()},
            customdata=sample_ids[rows].tolist(), name=cluster_name(label),
            hovertemplate="Primitive %{customdata}<extra>%{fullData.name}</extra>"))
    bounds = np.array([displayed.min(axis=(0, 1, 2)), displayed.max(axis=(0, 1, 2))])
    span = np.maximum(bounds[1] - bounds[0], 1e-6)
    axis = {key: {"title": key[0], "range": [float(bounds[0, j] - .03 * span[j]),
                                            float(bounds[1, j] + .03 * span[j])]}
            for j, key in enumerate(("xaxis", "yaxis", "zaxis"))}
    # Normalize all three lengths by one common factor: equal physical units,
    # without Plotly's volume-normalized long boxes extending off the viewport.
    aspect = dict(zip(("x", "y", "z"), (1.7 * span / span.max()).tolist()))
    spatial.update_layout(scene={**axis, "aspectmode": "manual", "aspectratio": aspect,
                          "camera": {"projection": {"type": "orthographic"},
                                     "eye": {"x": 1.35, "y": 1.35, "z": 1.15}}}, showlegend=False,
                          uirevision=revision + coordinates,
                          title="Physical geometry" if coordinates == "physical" else "Shapes at a common origin")
    projection_name = "UMAP" if projection_method == "umap" else "t-SNE"
    projection.update_layout(xaxis_title=f"{projection_name} 1", yaxis_title=f"{projection_name} 2",
                             dragmode="lasso", uirevision=revision,
                             title=f"Latent features projected with {projection_name}", showlegend=False)
    for j, key in enumerate(("xaxis", "yaxis")):
        low, high = float(xy[:, j].min()), float(xy[:, j].max())
        pad = .05 * max(high - low, 1e-6)
        projection.update_layout(**{key: {"range": [low - pad, high + pad]}})
    for fig in (spatial, projection):
        fig.update_layout(template="plotly_white", margin={"l": 20, "r": 20, "t": 45, "b": 35},
                          height=610, font={"family": "Arial", "size": 13})
    return spatial, projection, int(visible.sum())


def offline_html(spatial, projection, metadata):
    """Self-contained linked report; the live app also supports reclustering."""
    import html
    import plotly.io as pio

    left = pio.to_html(spatial, full_html=False, include_plotlyjs=True, div_id="geometry")
    right = pio.to_html(projection, full_html=False, include_plotlyjs=False, div_id="projection")
    detail = html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))
    script = r"""
const g=document.getElementById('geometry'), p=document.getElementById('projection');
const original=g.data.map(t=>({x:[...t.x],y:[...t.y],z:[...t.z],ids:[...t.customdata]}));
const opacity=p.data.map(t=>[...t.marker.opacity]);
function select(ids){
  const selected=ids===null?null:new Set(ids);
  original.forEach((t,i)=>{
    const update={};
    for(const key of ['x','y','z']) update[key]=[t[key].map((v,j)=>selected===null||selected.has(t.ids[j])?v:null)];
    Plotly.restyle(g,update,[i]);
  });
  p.data.forEach((t,i)=>Plotly.restyle(p,{'marker.opacity':[t.customdata.map((id,j)=>selected===null?opacity[i][j]:selected.has(id)?.95:.08)]},[i]));
  document.getElementById('selection').textContent=selected===null?'All exported samples':`${selected.size} selected primitives`;
}
p.on('plotly_selected',e=>{if(e)select(e.points.map(v=>v.customdata));});
p.on('plotly_click',e=>select(e.points.map(v=>v.customdata)));
g.on('plotly_click',e=>select(e.points.map(v=>v.customdata)));
document.getElementById('reset').onclick=()=>select(null);
"""
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><title>Task2 visual analysis</title>"
            "<style>body{font:15px Arial;margin:22px;color:#182333} .views{display:grid;grid-template-columns:1fr 1fr}"
            "@media(max-width:900px){.views{grid-template-columns:1fr}}pre{white-space:pre-wrap}button{padding:8px}</style>"
            "<h2>Task2 visual analysis</h2><p>One point = one 7-line primitive. Matching colors identify the same cluster. "
            "Gray denotes noise. Click or lasso to inspect geometry.</p>"
            f"<p>{html.escape(str(metadata.get('dataset', '')))} | {html.escape(metadata['arm'].upper())} + VAE | {html.escape(metadata.get('cluster_method', 'dbscan').upper())} in {html.escape(metadata['cluster_space'])} space "
            f"| projection = {metadata.get('projection', 'tsne')} | K = {metadata.get('n_clusters')} | eps = {metadata['eps']} | min_samples = {metadata['min_samples']}</p>"
            "<button id='reset'>Reset selection</button> <span id='selection'>All exported samples</span>"
            f"<div class='views'><div>{left}</div><div>{right}</div></div>"
            "<p>Projection-space groups describe the selected embedding. Colors do not match cluster identities across Raw/FMT arms. "
            "Use the live app to change projection and clustering settings.</p>"
            f"<details><summary>Reproducibility record</summary><pre>{detail}</pre></details><script>{script}</script></html>")


def table_csv(ids, xy, labels, core, projection="tsne", method="dbscan"):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["sample_id", "cluster", "is_core", f"{projection}_1", f"{projection}_2", "color"])
    for i, sample_id in enumerate(ids):
        writer.writerow([int(sample_id), int(labels[i]), bool(core[i]) if method == "dbscan" else "",
                         float(xy[i, 0]), float(xy[i, 1]), color(labels[i])])
    return stream.getvalue()
