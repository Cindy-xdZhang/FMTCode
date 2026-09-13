"""Scientific and interaction contracts for the exploratory Task2 viewer."""
import json
import re
import base64
import io
import zipfile

import numpy as np
import pytest

from FMT_Utils.Task2VisualAnalysis import (
    cluster, embed, figures, load_bundle, physical_paths, sample_rows,
)
from experiments.Task2_Visual_Analysis import Analysis, choose_ordinals, make_app


def test_geometry_restores_seed_translation_without_changing_shape():
    rng = np.random.default_rng(3)
    paths = rng.normal(size=(8, 7, 6, 3)).astype(np.float32)
    seeds = paths[:, 0, 0].copy()
    raw = (paths - paths[:, :1, :1]).reshape(8, -1)
    assert np.allclose(physical_paths(raw, seeds), paths, atol=1e-6)
    with pytest.raises(ValueError, match="relative"):
        physical_paths(raw + 2, seeds)


def test_dbscan_space_changes_membership_and_preserves_noise():
    latent = np.array([[0, 0], [.01, 0], [.02, 0], [5, 5], [5.01, 5], [5.02, 5], [30, 30]])
    xy = latent[[0, 3, 2, 1, 4, 5, 6]]
    high, core = cluster(latent, xy, "latent", .1, 3)
    low, _ = cluster(latent, xy, "tsne", .1, 3)
    assert high[0] == high[1] and low[0] != low[1]
    assert high[-1] == low[-1] == -1
    assert core[:-1].all() and not core[-1]
    assert np.all(cluster(latent, xy, "latent", .001, 2)[0] == -1)
    assert np.all(cluster(latent, xy, "latent", 100, 1)[0] == 0)


@pytest.mark.parametrize("eps,minimum", [(0, 2), (float('nan'), 2), (.1, 1.5), (.1, 0)])
def test_invalid_clustering_parameters(eps, minimum):
    with pytest.raises(ValueError):
        cluster(np.ones((4, 2)), np.ones((4, 2)), "latent", eps, minimum)


def test_cylinder_selection_uses_original_time_and_never_confirmation():
    spec = {"splits": {"train": [0, 1, 2], "cluster_calibration": [3, 4], "confirmation": [5]}}
    manifest = {"slices": [{"ordinal": i, "source_time": t} for i, t in enumerate([3., 7.4, 7.6, 9., 10., 11.])]}
    policy = {"simulation_time_ranges": {"cylinder": [0, 15]}, "minimum_physical_start": 7., "minimum_original_simulation_fraction": .5}
    assert choose_ordinals(spec, manifest, "cylinder", policy, "cluster_calibration") == ([2], 3, 7.5)
    with pytest.raises(ValueError, match="development"):
        choose_ordinals(spec, manifest, "cylinder", policy, "confirmation")


@pytest.fixture
def bundle(tmp_path):
    rng = np.random.default_rng(12)
    paths = rng.normal(size=(45, 7, 8, 3)).astype(np.float32)
    mu = rng.normal(size=(45, 5)).astype(np.float32)
    target = tmp_path / "synthetic.npz"
    np.savez_compressed(target, paths=paths, sample_ids=np.arange(45) * 2 + 100,
                        raw_mu=mu, fmt_mu=mu[:, ::-1],
                        metadata_json=json.dumps({"experiment": "test_only", "dataset": "synthetic",
                        "display_split": "synthetic", "confirmation_opened": False}))
    return target


def settings():
    return {"max_samples": 40, "sampling_seed": 11, "latent_normalization": "l1",
            "perplexity": 5., "tsne_seed": 13, "tsne_iterations": 300,
            "cluster_space": "tsne", "eps": 2., "min_samples": 4}


def test_subsampling_and_reports_keep_original_ids_and_means(bundle, tmp_path):
    analysis = Analysis(bundle, settings())
    assert np.array_equal(analysis.rows, sample_rows(45, 40, 11))
    report = analysis.write_report(tmp_path / "report", "fmt", "tsne", 2., 4)
    with np.load(report.parent / "analysis.npz") as data:
        assert np.array_equal(data["sample_ids"], analysis.ids)
        assert np.array_equal(data["latent_mu"], analysis.bundle["fmt_mu"][analysis.rows])
        assert len(data["cluster"]) == len(data["paths"]) == 40
    markup = report.read_text(encoding="utf-8")
    assert "plotly_selected" in markup
    assert not re.search(r"<script[^>]+src=", markup)
    assert "DBSCAN in projection space" in markup
    app = make_app(analysis)
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/_dash-dependencies").status_code == 200


def test_figures_link_ids_and_separate_each_neighbor_line():
    paths = np.arange(3 * 7 * 4 * 3).reshape(3, 7, 4, 3).astype(float)
    ids = np.array([10, 30, 99])
    xy = np.array([[0, 0], [1, 1], [2, 2]])
    labels = np.array([0, 0, -1])
    spatial, projection, count = figures(paths, ids, xy, labels, chosen=[30], geometry="primitive")
    assert count == 1 and len(spatial.data) == 1
    assert len(spatial.data[0].x) == 7 * 5
    assert np.isnan(np.asarray(spatial.data[0].x)).sum() == 7
    assert set(spatial.data[0].customdata) == {30}
    assert set(i for trace in projection.data for i in trace.customdata) == {10, 30, 99}
    assert spatial.data[0].line.color == projection.data[1].marker.color
    empty, _, count = figures(paths, ids, xy, labels, chosen=[])
    assert count == 0 and not empty.data


def test_zero_latents_and_malformed_bundle_are_rejected(bundle):
    with pytest.raises(ValueError, match="identical"):
        embed(np.zeros((45, 5)), settings())
    with pytest.raises(ValueError, match="perplexity"):
        embed(np.zeros((4, 5)), settings())
    data, meta = load_bundle(bundle)
    data["sample_ids"][0] = data["sample_ids"][1]
    np.savez_compressed(bundle, **data, metadata_json=json.dumps(meta))
    with pytest.raises(ValueError, match="unique"):
        load_bundle(bundle)


def test_ui_accepts_unrestricted_interactive_bundle(bundle):
    analysis = Analysis(bundle, settings())
    analysis.source.update(display_split="interactive", confirmation_opened=None)
    assert make_app(analysis).server.test_client().get('/_dash-layout').status_code == 200


def test_ui_export_zip_preserves_clustering_and_current_selection(bundle):
    analysis = Analysis(bundle, settings())
    app = make_app(analysis)
    apply = next(v["callback"].__wrapped__ for k, v in app.callback_map.items() if "state.data" in k)
    state, options, _, message = apply(0, "fmt", "tsne", 2., 4)
    assert state["meta"]["arm"] == "fmt" and "Applied" in message
    selected = [int(analysis.ids[0])]
    left, right, _ = figures(analysis.paths, analysis.ids, np.array(state["xy"]), np.array(state["labels"]), chosen=selected)
    download = app.callback_map["download.data"]["callback"].__wrapped__
    result = download(1, state, left.to_dict(), right.to_dict(), selected, [], "center", "physical")
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(result["content"]))) as archive:
        metadata = json.loads(archive.read("analysis.json"))
        assert metadata["selected_sample_ids"] == selected
        assert metadata["sample_ids"] == analysis.ids.tolist()
        assert len(archive.read("samples.csv").decode().splitlines()) == 41
        assert "plotly_selected" in archive.read("index.html").decode()


def test_hiding_clusters_removes_both_views_without_relabeling():
    paths = np.arange(3 * 7 * 4 * 3).reshape(3, 7, 4, 3).astype(float)
    ids = np.array([10, 30, 99])
    labels = np.array([0, 1, -1])
    xy = np.array([[0, 0], [1, 1], [2, 2]])
    spatial, projection, count = figures(paths, ids, xy, labels, hidden_clusters=[1, -1])
    assert count == 1
    assert {i for t in spatial.data for i in t.customdata} == {10}
    assert {i for t in projection.data for i in t.customdata} == {10}
    assert np.array_equal(labels, [0, 1, -1])
    assert spatial.data[0].line.color == projection.data[0].marker.color
    empty, empty_projection, count = figures(paths, ids, xy, labels, hidden_clusters=[0, 1, -1])
    assert count == 0 and not empty.data and not empty_projection.data
    assert projection.layout.xaxis.range == empty_projection.layout.xaxis.range


def test_umap_kmeans_and_export_use_actual_method_names(bundle, tmp_path):
    analysis = Analysis(bundle, settings())
    xy, labels, core, meta = analysis.get("fmt","latent",None,None,"umap","kmeans",3,8,.2)
    assert xy.shape == (40,2) and np.isfinite(xy).all()
    assert len(np.unique(labels)) == 3 and not np.any(labels == -1) and not core.any()
    xy2, labels2, _, _ = analysis.get("fmt","latent",None,None,"tsne","kmeans",3)
    assert np.array_equal(labels,labels2)
    assert not np.allclose(xy,xy2)
    path = analysis.write_report(tmp_path/'umap',"fmt","latent",None,None,projection="umap",method="kmeans",n_clusters=3,neighbors=8,min_dist=.2)
    with np.load(path.parent/'analysis.npz') as data:
        assert 'umap' in data.files and 'tsne' not in data.files
    assert 'umap_1' in (path.parent/'samples.csv').read_text()
    assert 'KMEANS in latent space' in path.read_text()
    _, right, _ = figures(analysis.paths,analysis.ids,xy,labels,projection_method="umap")
    assert right.layout.xaxis.title.text == "UMAP 1"
    with pytest.raises(ValueError,match="K"):
        cluster(xy,xy,"latent",None,None,"kmeans",41)
