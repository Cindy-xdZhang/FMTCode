"""Task2 visual analysis: export paired VAE means, render reports, or serve a linked UI.

Run from the repository root, for example:
python experiments/Task2_Visual_Analysis.py export --dataset cylinder3d
python experiments/Task2_Visual_Analysis.py serve --bundle outputs/.../latent_bundle.npz
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "experiments"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
from FMT_Utils.Task2VisualAnalysis import (
    cluster, cluster_name, color, digest, embed, figures, load_bundle,
    offline_html, physical_paths, sample_rows, table_csv,
)

DEFAULT_CONFIG = ROOT / "config/Other_Task2_VisualAnalysis_1.3.json"


def root_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def versions():
    result = {"python": platform.python_version()}
    for name in ("numpy", "scikit-learn", "torch", "plotly", "dash", "umap-learn"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def code_identity():
    result = {"source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__), ROOT / "FMT_Utils/Task2VisualAnalysis.py",
        ROOT / "experiments/Run_Task2_3D_Main.py", ROOT / "experiments/Verify_HighReVAE.py",
        ROOT / "FMT_Utils/VAE_3D.py", ROOT / "FMT_Utils/Task12Data_3D.py",
        ROOT / "FMT_Utils/RawPathline_3D.py", ROOT / "FMT_Utils/DFT_FMT_3D.py")}}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    result["git_commit"] = git.stdout.strip() if git.returncode == 0 else "unavailable"
    return result


def frozen_recipe(config):
    """Check the existing frozen manifest without opening any flow cache."""
    import yaml
    spec_path = root_path(config["frozen_config"])
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    source_path = root_path(spec["selection_config"])
    selection_path = root_path(spec["selection_path"])
    audit_path = root_path(spec["selection_audit_path"])
    manifest_path = root_path(spec["output_root"]) / "frozen_recipe_manifest.json"
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    paths = {"confirmation_config_sha256": spec_path, "selection_config_sha256": source_path,
             "selection_sha256": selection_path, "selection_audit_sha256": audit_path}
    if spec["task"] != "Task2" or source["task"] != "Task2":
        raise ValueError("a frozen Task2 recipe is required")
    for key, path in paths.items():
        if manifest.get(key) != digest(path):
            raise ValueError(f"frozen recipe hash mismatch: {key}")
    if audit.get("status") != "PASS" or selected.get("confirmation_opened", True) or selected.get("outer_ordinals_opened", True):
        raise ValueError("source selection was not a passed development-only selection")
    if manifest["winner"] != selected["winner"] or manifest["splits"] != spec["splits"]:
        raise ValueError("frozen selection/splits changed")
    matches = [a for a in source["architectures"] if a["id"] == selected["winner"]["architecture"]]
    if len(matches) != 1:
        raise ValueError("ambiguous frozen VAE architecture")
    return spec, source, selected["winner"], matches[0], manifest_path


def choose_ordinals(spec, manifest, dataset, policy, display_split):
    """Metadata-only choice, restricted to development data for interactive tuning."""
    if display_split not in {"train", "cluster_calibration"}:
        raise ValueError("interactive exploration must use development train/calibration data")
    by_id = {int(s["ordinal"]): s for s in manifest["slices"]}
    cutoff = None
    if dataset in policy["simulation_time_ranges"]:
        start, end = policy["simulation_time_ranges"][dataset]
        cutoff = max(policy["minimum_physical_start"], start + policy["minimum_original_simulation_fraction"] * (end - start))
    def eligible(i):
        time = float(by_id[i]["source_time"])
        if not np.isfinite(time):
            raise ValueError("non-finite source time in cache manifest")
        return cutoff is None or time >= cutoff
    train = [int(i) for i in spec["splits"]["train"] if eligible(int(i))]
    display = [int(i) for i in spec["splits"][display_split] if eligible(int(i))]
    if not train or not display:
        raise ValueError("no eligible development train/display slices; create a new cache version")
    ordinal = min(display, key=lambda i: (float(by_id[i]["source_time"]), i))
    return train, ordinal, cutoff


def export_bundle(config, dataset, target, device_name="auto"):
    import torch
    from DeepUtils.utils import EasyConfig
    from experiments.Run_Task2_3D_Main import _prepare_inputs
    from experiments.Verify_HighReVAE import _train

    target = Path(target)
    if target.exists():
        raise FileExistsError(f"reuse this bundle for serve/render, or choose a new output: {target}")
    spec, source, winner, architecture, frozen_path = frozen_recipe(config)
    groups = [g for g in source["groups"].values() if dataset in g["datasets"]]
    if len(groups) != 1:
        raise ValueError(f"dataset must occur in exactly one frozen group: {dataset}")
    group = groups[0]
    cache = root_path(group["development_cache"]) / dataset
    manifest_path = cache / "manifest.json"
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(root_path(config["cylinder_time_policy"]).read_text(encoding="utf-8"))
    train_ids, display_id, cutoff = choose_ordinals(spec, cache_manifest, dataset, policy, config["display_split"])
    paths = sorted(cache.glob("slice_*.npz"))
    if len(paths) != int(spec["expected_slices"]):
        raise ValueError(f"expected {spec['expected_slices']} slices in {cache}")
    records = {}
    evidence = []
    for ordinal in sorted(set(train_ids + [display_id])):
        path = paths[ordinal]
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
            entry = next(s for s in cache_manifest["slices"] if int(s["ordinal"]) == ordinal)
            if int(metadata["ordinal"]) != ordinal or metadata["dataset"] != dataset or not np.isclose(metadata["source_time"], entry["source_time"]):
                raise ValueError("cache metadata and manifest identity differ")
            raw = data["raw_features"].astype(np.float32)
            fmt = data["fmt_features"].astype(np.float32)
            if raw.ndim != 2 or fmt.ndim != 2 or len(raw) != len(fmt) or not np.isfinite(raw).all() or not np.isfinite(fmt).all():
                raise ValueError("invalid cached raw/FMT arrays")
            records[ordinal] = {"raw": raw, "fmt": fmt, "features": {}, "metadata": metadata}
            if ordinal == display_id:
                geometry = physical_paths(raw, data["seeds"])
                sample_ids = np.arange(len(raw), dtype=np.int64)
        evidence.append({"ordinal": ordinal, "path": str(path), "sha256": digest(path),
                         "source_time": metadata["source_time"]})
    # Explicitly do not read the cached reference or valid_mask label arrays.
    train = [records[i] for i in train_ids]
    evaluation = [records[display_id]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    torch.set_num_threads(4)
    seed = int(config["training_seed"])
    source_config = EasyConfig(str(root_path(group["source_config"])))
    latent, losses = {}, {}
    for arm in ("raw", "fmt"):
        print(f"{dataset}: training {arm}, {architecture['id']}, seed={seed}, device={device}", flush=True)
        train_x, evaluate_x = _prepare_inputs(train, evaluation, arm, winner["fmt_feature"], device)
        _, latent[f"{arm}_mu"], losses[arm] = _train(train_x, evaluate_x, architecture, source_config, seed, device)
        del train_x, evaluate_x
    metadata = {
        "schema": 1, "experiment": config["experiment"], "dataset": dataset,
        "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
        "frozen_recipe_sha256": digest(frozen_path), "architecture": architecture,
        "fmt_feature": winner["fmt_feature"], "training_seed": seed,
        "source_config_sha256": digest(root_path(group["source_config"])),
        "train_ordinals": train_ids, "historical_train_ordinals": spec["splits"]["train"],
        "display_ordinal": display_id, "display_split": config["display_split"],
        "display_metadata": records[display_id]["metadata"],
        "minimum_physical_time": cutoff, "time_policy": policy,
        "sampling_changed_from_frozen_run": train_ids != spec["splits"]["train"],
        "cache_manifest_sha256": digest(manifest_path), "cache_evidence": evidence,
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "versions": versions(), "code": code_identity(), "losses": losses,
        "sample_id_definition": "row in the displayed source slice, before analysis subsampling",
        "latent_definition": "deterministic encoder posterior mean mu, not sampled z",
        "geometry_definition": "cached local coordinates plus original center seed",
        "reference_labels_used": False, "confirmation_opened": False,
        "checkpoint_policy": "models in memory only; no checkpoints written",
        "scope": "development-only qualitative exploration; no replacement of frozen main-table metrics",
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, paths=geometry, sample_ids=sample_ids, **latent,
                        metadata_json=json.dumps(metadata, sort_keys=True))
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(target, flush=True)
    return target


class Analysis:
    def __init__(self, bundle_path, settings):
        self.path = Path(bundle_path)
        self.bundle, self.source = load_bundle(self.path)
        self.settings = dict(settings)
        self.rows = sample_rows(len(self.bundle["paths"]), int(settings["max_samples"]), int(settings["sampling_seed"]))
        self.paths = self.bundle["paths"][self.rows]
        self.ids = self.bundle["sample_ids"][self.rows]
        self.embeddings = {}
        self.bundle_hash = digest(self.path)

    def get(self, arm, space, eps, min_samples, projection="tsne", method="dbscan", n_clusters=8, neighbors=15, min_dist=.1):
        if arm not in {"raw", "fmt"}:
            raise ValueError("arm must be raw or fmt")
        options = {**self.settings, "projection": projection, "umap_neighbors": neighbors, "umap_min_dist": min_dist}
        key = (arm, projection, neighbors if projection == "umap" else None, min_dist if projection == "umap" else None)
        if key not in self.embeddings:
            self.embeddings[key] = embed(self.bundle[f"{arm}_mu"][self.rows], options)
        values, xy, kl = self.embeddings[key]
        labels, core = cluster(values, xy, space, eps, min_samples, method, n_clusters, self.settings["tsne_seed"])
        counts = {str(int(k)): int((labels == k).sum()) for k in np.unique(labels)}
        meta = {"experiment": self.source.get("experiment"), "dataset": self.source.get("dataset"),
                "source_bundle": str(self.path.resolve()), "source_bundle_sha256": self.bundle_hash,
                "source": self.source, **self.settings, "arm": arm, "cluster_space": "latent" if space == "latent" else "projection",
                "eps": float(eps) if method == "dbscan" else None,
                "min_samples": int(min_samples) if method == "dbscan" else None,
                "projection": projection, "cluster_method": method,
                "n_clusters": int(n_clusters) if method == "kmeans" else None,
                "umap_neighbors": neighbors, "umap_min_dist": min_dist,
                "core_status_applicable": method == "dbscan", "cluster_counts": counts,
                "noise_fraction": float(np.mean(labels == -1)), "analysis_sample_count": len(self.ids),
                "source_sample_count": len(self.bundle["paths"]), "sample_ids": self.ids.tolist(),
                "tsne_kl_divergence": kl, "versions": versions(),
                "analysis_source_sha256": {"driver": digest(Path(__file__)),
                    "core": digest(ROOT / "FMT_Utils/Task2VisualAnalysis.py")},
                "cluster_fit_scope": "exactly the displayed analysis subset, transductive",
                "latent_dimension": int(values.shape[1])}
        return xy, labels, core, meta

    def write_report(self, output, arm, space, eps, min_samples, **kwargs):
        xy, labels, core, meta = self.get(arm, space, eps, min_samples, **kwargs)
        target = Path(output)
        target.mkdir(parents=True, exist_ok=True)
        spatial, projection, _ = figures(self.paths, self.ids, xy, labels, projection_method=meta["projection"])
        (target / "index.html").write_text(offline_html(spatial, projection, meta), encoding="utf-8")
        (target / "samples.csv").write_text(table_csv(self.ids, xy, labels, core, meta["projection"], meta["cluster_method"]), encoding="utf-8")
        (target / "analysis.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        np.savez_compressed(target / "analysis.npz", sample_ids=self.ids, **{meta["projection"]: xy},
                            cluster=labels, is_core=core,
                            latent_mu=self.bundle[f"{arm}_mu"][self.rows],
                            paths=self.paths, metadata_json=json.dumps(meta, sort_keys=True))
        return target / "index.html"


def make_app(analysis, initial_arm="fmt", recompute_config=None):
    from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
    app = Dash(__name__)
    initial_key = str(analysis.path.resolve())
    analyses = {initial_key: analysis}
    jobs = None
    if recompute_config and "recompute" in recompute_config:
        from FMT_Utils.Task2GeometryRecompute import RecomputeJobs
        jobs = RecomputeJobs(recompute_config)
    def current(key=None):
        if key is None:
            key = initial_key
        if key not in analyses:
            raise ValueError("分析数据已失效，请刷新页面")
        return analyses[key]
    def summary(value):
        s = value.source
        integration = s.get("display_metadata", {})
        duration = integration.get("integration_duration", "cached")
        if isinstance(duration, (int, float)):
            duration = f"{duration:.8g}"
        return (f"{s.get('dataset', 'unknown')} | t = {integration.get('source_time', 'unspecified')} | "
                f"latent dimension = {value.bundle['fmt_mu'].shape[1]} | {len(value.ids)} / {len(value.bundle['paths'])} samples | "
                f"实际终点 = {integration.get('integration_end_time', 'cached')} | 实际积分时长 = {duration} | "
                f"{'已在源 tmax 截断 | ' if integration.get('truncated_at_tmax') else ''}interactive exploration")
    settings = analysis.settings
    source = analysis.source
    latent_dim = analysis.bundle["fmt_mu"].shape[1]
    space_names = {"tsne": "当前投影空间（2D）", "projection": "当前投影空间（2D）",
                   "latent": f"VAE 潜在特征空间（{latent_dim}D）"}
    normalization = {"l1": "每个向量除以各分量绝对值之和（L1 归一化）",
                     "l2": "每个向量除以其欧氏长度（L2 归一化）",
                     "none": "不做额外归一化"}[settings["latent_normalization"]]

    input_style = {"width": "110px", "padding": "7px"}
    app.layout = html.Div([
        html.H2("Task2 visual analysis", style={"marginBottom": "8px"}),
        html.Div(summary(analysis), id="dataset-summary"),
        html.Div([
            html.Strong("重新计算轨线几何与特征"),
            html.Div([
                html.Label(["流场", dcc.Dropdown(id="flow", options=[
                    {"label": name + ("（源数据不可用）" if not entry["available"] else ""),
                     "value": name, "disabled": not entry["available"]}
                    for name, entry in (jobs.catalog.items() if jobs else [])],
                    value=source.get("dataset"), clearable=False, style={"width": "250px"})]),
                html.Label(["起始时间 t0", dcc.Input(id="start-time", type="number", style=input_style)]),
                html.Label(["积分 dt（源时间单位）", dcc.Input(id="integration-dt", type="number", min=1e-12, style=input_style)]),
                html.Label(["积分长度 N（步数）", dcc.Input(id="integration-steps", type="number", min=1, step=1, style=input_style)]),
                html.Button("重新计算并切换流场", id="recompute", n_clicks=0, disabled=jobs is None),
            ], style={"display": "flex", "gap": "16px", "alignItems": "end", "flexWrap": "wrap", "marginTop": "8px"}),
            html.Div(id="integration-hint", style={"marginTop": "8px"}),
            html.Small("重算会读取原始流场，按新 dt/N 生成轨线，并用当前全部轨线分别训练两臂 VAE（无数据划分），随后更新投影与聚类；"
                       "超过源数据末尾会截断，最后一步可短于 dt；每线重采样为32点。过程通常需要数分钟，期间旧视图仍可操作。每次结果独立保存。"),
            html.Pre(id="recompute-status", style={"whiteSpace": "pre-wrap"}),
        ], style={"padding": "14px", "border": "1px solid #dce3ed", "borderRadius": "8px", "margin": "14px 0"}),
        html.P("Click a path or point, or lasso points to inspect matching geometry. Cluster colors are shared within each view pair; Raw/FMT cluster IDs are independent."),
        html.Div([
            html.Label(["VAE input", dcc.Dropdown(id="arm", options=[{"label": "FMT + VAE", "value": "fmt"}, {"label": "Raw + same VAE", "value": "raw"}], value=initial_arm, clearable=False, style={"width": "185px"})]),
            html.Label(["降维方法", dcc.Dropdown(id="projection-method", options=[{"label":"t-SNE", "value":"tsne"},{"label":"UMAP", "value":"umap"}], value=settings.get("projection", "tsne"), clearable=False, style={"width":"130px"})]),
            html.Label(["聚类方法", dcc.Dropdown(id="cluster-method", options=[{"label":"DBSCAN", "value":"dbscan"},{"label":"KMeans", "value":"kmeans"}], value=settings.get("cluster_method", "dbscan"), clearable=False, style={"width":"130px"})]),
            html.Label(["K（簇数）", dcc.Input(id="cluster-k", type="number", min=1, step=1, value=settings.get("n_clusters",8), style=input_style)]),
            html.Label(["聚类空间", dcc.Dropdown(id="space", options=[{"label": name, "value": key} for key, name in space_names.items() if key != "tsne"], value="latent" if settings["cluster_space"] == "latent" else "projection", clearable=False, style={"width": "280px"})]),
            html.Label(["eps (radius)", dcc.Input(id="eps", type="number", min=.000001, value=settings["eps"], style=input_style)]),
            html.Label(["min_samples", dcc.Input(id="minimum", type="number", min=1, step=1, value=settings["min_samples"], style=input_style)]),
            html.Button("Apply clustering", id="apply", n_clicks=0),
            html.Button("Reset selection", id="reset", n_clicks=0),
            html.Button("Export current view", id="download-button", n_clicks=0),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "alignItems": "end"}),
        html.Div([
            html.Strong("降维、聚类与特征的关系"),
            html.P(f"特征来自 Task2 VAE（变分自编码器）的 {latent_dim} 维均值 mu；{normalization}。没有使用 FlowNet 的三维卷积网络。"),
            html.P("t-SNE（t分布随机邻域嵌入）或 UMAP（统一流形近似与投影）将同一批潜在向量投影到右图二维平面。"),
            html.P("聚类空间选‘当前投影’时按二维距离分组；选‘VAE 潜在特征’时按高维距离分组，降维只影响显示。投影会改变距离和密度，不同投影的 eps 不能直接比较。"),
            html.P("DBSCAN（基于密度的带噪声聚类）使用 eps 邻域半径和 min_samples 邻域样本数，噪声为 -1。KMeans（K均值聚类）按指定 K 分组，每个样本均有类别，没有噪声/核心点定义。"),
            html.Div([
                html.Label(["UMAP n_neighbors（邻居数）", dcc.Input(id="umap-neighbors",type="number",min=2,step=1,value=15,style=input_style)]),
                html.Label(["UMAP min_dist（投影紧密程度）", dcc.Input(id="umap-min-dist",type="number",min=0,max=1,step=.05,value=.1,style=input_style)]),
            ], style={"display":"flex","gap":"20px"}),
            html.P("修改降维或聚类选项后点击 Apply clustering；下方 Applied 显示已生效设置。隐藏簇仅影响显示，重新分析后清空。"),
        ], style={"background":"#f4f7fb","padding":"14px 18px","marginTop":"16px","borderRadius":"8px"}),
        html.Div([
            dcc.Dropdown(id="clusters", multi=True, placeholder="All clusters (including noise)", style={"flex": "2"}),
            dcc.Dropdown(id="geometry", options=[{"label": "Center pathline", "value": "center"}, {"label": "All 7 pathlines", "value": "primitive"}], value="center", clearable=False, style={"flex": "1"}),
            dcc.Dropdown(id="coordinates", options=[{"label": "Physical coordinates", "value": "physical"}, {"label": "Shapes at common origin", "value": "centered"}], value="physical", clearable=False, style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "marginTop": "14px"}),
        html.Div([
            html.Label("隐藏簇（左右两图均不显示；不重新聚类）"),
            dcc.Dropdown(id="hidden-clusters", multi=True, placeholder="选择要隐藏的 cluster，例如 Noise (-1)"),
        ], style={"marginTop": "12px"}),
        html.Pre(id="status", style={"whiteSpace": "pre-wrap"}),
        dcc.Loading(html.Div([
            dcc.Graph(id="spatial", style={"width": "50%"}),
            dcc.Graph(id="projection", style={"width": "50%"}, config={"displaylogo": False}),
        ], style={"display": "flex"})),
        html.Div(id="selection-status"),
        html.Details([html.Summary("Projection settings"), html.Pre(json.dumps(settings, indent=2))]),
        dcc.Store(id="state"), dcc.Store(id="selected"), dcc.Download(id="download"),
        dcc.Store(id="active-bundle", data=initial_key), dcc.Store(id="recompute-job"),
        dcc.Interval(id="recompute-poll", interval=1500, disabled=True),
    ], style={"fontFamily": "Arial", "padding": "24px", "color": "#182333", "maxWidth": "1800px", "margin": "auto"})

    @app.callback(Output("eps", "disabled"), Output("minimum", "disabled"), Output("cluster-k", "disabled"),
                  Output("umap-neighbors", "disabled"), Output("umap-min-dist", "disabled"),
                  Input("cluster-method", "value"), Input("projection-method", "value"))
    def parameter_controls(method, projection):
        return method != "dbscan", method != "dbscan", method != "kmeans", projection != "umap", projection != "umap"

    @app.callback(Output("state", "data"), Output("clusters", "options"), Output("clusters", "value"), Output("status", "children"),
                  Input("apply", "n_clicks"), State("arm", "value"), State("space", "value"), State("eps", "value"), State("minimum", "value"), Input("active-bundle", "data"),
                  State("projection-method", "value"), State("cluster-method", "value"), State("cluster-k", "value"),
                  State("umap-neighbors", "value"), State("umap-min-dist", "value"))
    def apply_clustering(clicks, arm, space, eps, minimum, bundle_key=None, projection="tsne", method="dbscan", k=8, neighbors=15, min_dist=.1):
        try:
            if method == "dbscan" and (eps is None or minimum is None):
                raise ValueError("enter eps and min_samples")
            value = current(bundle_key)
            xy, labels, core, meta = value.get(arm, space, eps, minimum, projection, method, k, neighbors, min_dist)
            options = [{"label": f"{cluster_name(k)}: {(labels == k).sum()} samples", "value": int(k)} for k in np.unique(labels)]
            count = len(set(labels) - {-1})
            state = {"xy": xy.tolist(), "labels": labels.tolist(), "core": core.tolist(), "meta": meta,
                     "bundle_key": str(value.path.resolve())}
            parameters = f"K={k}" if method == "kmeans" else f"eps={eps}, min_samples={minimum}"
            return state, options, [], f"Applied: {arm.upper()} + VAE | {projection.upper()} | {method.upper()} | {space_names[space]} | {parameters}\n{count} clusters | {int((labels == -1).sum())} noise samples ({meta['noise_fraction']:.1%})"
        except (ValueError, TypeError, ImportError) as exc:
            return no_update, no_update, no_update, f"Settings rejected; previous applied view retained: {exc}"

    @app.callback(Output("selected", "data"), Input("projection", "selectedData"), Input("projection", "clickData"),
                  Input("spatial", "clickData"), Input("reset", "n_clicks"), Input("state", "data"))
    def select(brushed, point, path, reset, state):
        event = ctx.triggered[0]["prop_id"]
        if event in {"reset.n_clicks", "state.data"}:
            return None
        value = {"projection.selectedData": brushed, "projection.clickData": point, "spatial.clickData": path}.get(event)
        if not value:
            return no_update
        return sorted({int(p["customdata"]) for p in value.get("points", []) if p.get("customdata") is not None})

    @app.callback(Output("spatial", "figure"), Output("projection", "figure"), Output("selection-status", "children"),
                  Input("state", "data"), Input("selected", "data"), Input("clusters", "value"), Input("geometry", "value"), Input("coordinates", "value"), Input("hidden-clusters", "value"))
    def update(state, selected, clusters, geometry, coordinates, hidden=None):
        if state is None:
            return no_update, no_update, ""
        meta = state["meta"]
        value = current(state.get("bundle_key"))
        revision = json.dumps([state.get("bundle_key"),meta["arm"],meta["cluster_space"],meta.get("projection"),meta.get("cluster_method"),meta.get("n_clusters"),meta["eps"],meta["min_samples"],meta.get("umap_neighbors"),meta.get("umap_min_dist")])
        spatial, projection, count = figures(value.paths, value.ids, np.asarray(state["xy"]), np.asarray(state["labels"]),
            chosen=selected, clusters=clusters or None, hidden_clusters=hidden, geometry=geometry, coordinates=coordinates, revision=revision, projection_method=meta.get("projection", "tsne"))
        return spatial, projection, f"Showing geometry for {count} / {len(value.ids)} analysis primitives. 隐藏簇: {hidden or '无'}；灰色表示未隐藏的噪声。"

    @app.callback(Output("download", "data"), Input("download-button", "n_clicks"), State("state", "data"),
                  State("spatial", "figure"), State("projection", "figure"), State("selected", "data"),
                  State("clusters", "value"), State("geometry", "value"), State("coordinates", "value"), State("hidden-clusters", "value"), prevent_initial_call=True)
    def download(clicks, state, spatial, projection, selected, clusters, geometry, coordinates, hidden=None):
        if state is None:
            return no_update
        meta = {**state["meta"], "selected_sample_ids": selected, "selected_clusters": clusters,
                "hidden_clusters": hidden or [], "geometry": geometry, "coordinates": coordinates}
        value = current(state.get("bundle_key"))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("index.html", offline_html(spatial, projection, meta))
            archive.writestr("analysis.json", json.dumps(meta, indent=2))
            archive.writestr("samples.csv", table_csv(value.ids, np.asarray(state["xy"]), np.asarray(state["labels"]), np.asarray(state["core"]), meta.get("projection", "tsne"), meta.get("cluster_method", "dbscan")))
        return dcc.send_bytes(buffer.getvalue(), f"task2_{meta['arm']}_{meta['cluster_space']}.zip")

    @app.callback(Output("hidden-clusters", "options"), Output("hidden-clusters", "value"), Input("state", "data"))
    def hide_options(state):
        if not state:
            return [], []
        labels = np.asarray(state["labels"])
        return [{"label": f"{cluster_name(k)}: {int((labels == k).sum())} samples", "value": int(k)}
                for k in np.unique(labels)], []

    @app.callback(Output("integration-dt", "value"), Output("integration-steps", "value"), Output("start-time", "value"), Input("flow", "value"))
    def flow_defaults(dataset):
        entry = jobs.catalog.get(dataset, {}) if jobs else {}
        if dataset == source.get("dataset") and "integration_dt" in source:
            return source["integration_dt"], source["integration_steps"], source["display_metadata"]["source_time"]
        return entry.get("default_dt"), entry.get("default_steps", 48), entry.get("entries", {}).get(entry.get("display_id"), {}).get("source_time")

    @app.callback(Output("integration-hint", "children"), Input("flow", "value"), Input("integration-dt", "value"), Input("integration-steps", "value"), Input("start-time", "value"))
    def duration_hint(dataset, dt, steps, start_time):
        if dt is None or steps is None or start_time is None:
            return "请填写 t0、dt 和 N"
        entry = jobs.catalog.get(dataset, {}) if jobs else {}
        bounds = entry.get("source_time_range")
        endpoint = min(start_time+dt*steps,bounds[1]) if bounds else start_time+dt*steps
        source_hint = f"源时间范围 [{bounds[0]:g}, {bounds[1]:g}]；" if bounds else ""
        return f"{source_hint}请求时长={dt*steps:.8g}，请求终点={start_time+dt*steps:g}；实际终点={endpoint:g}，实际时长={max(0.,endpoint-start_time):.8g}。超过源 tmax 时截断。"

    @app.callback(Output("recompute-job", "data"), Output("recompute-poll", "disabled"),
                  Output("recompute-status", "children"), Output("active-bundle", "data"), Output("recompute", "disabled"),
                  Input("recompute", "n_clicks"), Input("recompute-poll", "n_intervals"),
                  State("recompute-job", "data"), State("flow", "value"), State("integration-dt", "value"),
                  State("integration-steps", "value"), State("start-time", "value"), prevent_initial_call=True)
    def recalculate(clicks, ticks, job_id, dataset, dt, steps, start_time):
        if jobs is None:
            return no_update, True, "未配置重算数据源", no_update, True
        if ctx.triggered_id == "recompute":
            try:
                if start_time is None:
                    raise ValueError("请填写起始时间 t0")
                key = jobs.start(dataset, dt, steps, start_time)
                return key, False, "已开始后台重算；完成后自动切换新几何和特征。", no_update, True
            except (ValueError, OSError) as exc:
                return no_update, True, str(exc), no_update, False
        if not job_id:
            return no_update, True, no_update, no_update, False
        job = jobs.snapshot(job_id)
        if job["status"] == "complete":
            key = str(Path(job["bundle"]).resolve())
            if key not in analyses:
                analyses[key] = Analysis(key, settings)
            return job_id, True, f"重算完成：{job['dataset']}，dt={job['dt']:g}，N={job['steps']}\n结果：{job['run_dir']}", key, False
        if job["status"] == "failed":
            return job_id, True, f"重算失败，保留旧视图：{job['message']}", no_update, False
        return job_id, False, f"{job['dataset']} | dt={job['dt']:g} | N={job['steps']}\n{job['message']}", no_update, True

    @app.callback(Output("dataset-summary", "children"), Input("state", "data"))
    def update_summary(state):
        return summary(current(state.get("bundle_key") if state else None))

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "render", "serve"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset")
    parser.add_argument("--start-time", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--projection", choices=("tsne", "umap"))
    parser.add_argument("--cluster-method", choices=("dbscan", "kmeans"))
    parser.add_argument("--n-clusters", type=int)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--arm", choices=("raw", "fmt"), default="fmt")
    parser.add_argument("--cluster-space", choices=("tsne", "projection", "latent"))
    parser.add_argument("--eps", type=float)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--port", type=int, default=8052)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.command == "export":
        if not args.dataset:
            parser.error("export requires --dataset")
        target = args.output or root_path(config["output_root"]) / args.dataset / "latent_bundle.npz"
        if config.get("display_split") == "interactive":
            from FMT_Utils.Task2GeometryRecompute import catalog, recompute
            entry = catalog(config)[args.dataset]
            if target.exists():
                raise FileExistsError(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            recompute(config, entry, args.dt if args.dt is not None else entry["default_dt"],
                      args.steps if args.steps is not None else entry["default_steps"], target.parent,
                      lambda message: print(message, flush=True), args.start_time)
        else:
            export_bundle(config, args.dataset, target, args.device)
        return
    if args.bundle is None:
        parser.error("render/serve requires --bundle; export it first")
    settings = dict(config["analysis"])
    for key in ("cluster_space", "eps", "min_samples", "projection", "cluster_method", "n_clusters"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    analysis = Analysis(args.bundle, settings)
    if args.command == "render":
        # Each explicit parameter choice gets its own output directory by default.
        suffix = f"{settings.get('projection','tsne')}_{settings.get('cluster_method','dbscan')}_k{settings.get('n_clusters',8)}_{args.arm}_{settings['cluster_space']}_eps{settings['eps']}_min{settings['min_samples']}"
        output = args.output or args.bundle.parent / "reports" / suffix
        print(analysis.write_report(output, args.arm, settings["cluster_space"], settings["eps"], settings["min_samples"], projection=settings.get("projection","tsne"), method=settings.get("cluster_method","dbscan"), n_clusters=settings.get("n_clusters",8)))
    else:
        make_app(analysis, initial_arm=args.arm, recompute_config=config).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
