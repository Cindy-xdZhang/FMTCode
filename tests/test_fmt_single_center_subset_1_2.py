"""Original-center subset contracts and frozen training-loop checks."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import FMT_SingleCenter_Task354C_1_2 as runner


def fixture_source(tmp_path):
    rng = np.random.default_rng(1740)
    source = {'source_output': str(tmp_path/'source'), 'source_files': {},
              'expected_counts': {'train': 6},
              'subset': {'root': str(tmp_path/'subset'), 'files': {}}}
    for flow in ('channel', 'tbl'):
        folder = tmp_path/'source'/'physical'/flow/'train'
        folder.mkdir(parents=True)
        g = rng.normal(size=(4, 7, 32, 3)).cumsum(2)
        seeds = np.zeros((4, 7, 3)); seeds[:, :, 0] = np.arange(1, 8)
        rows, centers = np.array([0, 2, 3]), np.array([1, 3, 0])
        seeds[rows, centers] = 0.
        np.save(folder/'geometry.npy', g)
        np.save(folder/'seeds.npy', seeds)
        np.savez(folder/'metadata.npz', center=np.zeros((4, 3)), centroid=np.zeros((4, 3)),
                 radius=np.ones(4), neighbor_distance=np.ones(4), counts=np.full(4, 7),
                 labels=np.array([0, 1, 0, 1]), instance=np.arange(4))
        subset = tmp_path/'subset'/'subsets'/f'{flow}_train.npz'
        subset.parent.mkdir(parents=True, exist_ok=True)
        np.savez(subset, row_ids=rows, center_ids=centers)
        source['source_files'][flow] = {'train': {'samples': 4, 'files': {
            name: runner.sha(folder/name) for name in ('geometry.npy', 'seeds.npy', 'metadata.npz')}}}
        source['subset']['files'][flow] = {'train': {'sha256': runner.sha(subset), 'retained': 3}}
    return source


def test_subset_keeps_source_rows_and_exact_centers(tmp_path):
    source = fixture_source(tmp_path)
    for flow in ('channel', 'tbl'):
        check = runner.verify_subset_centers(source, flow, 'train')
        assert check['retained'] == 3 and check['excluded'] == 1
        assert check['nearest_substitution'] is False
    values, ids, evidence = runner.read4(source, 'train', 'cpu', 2)
    assert values.shape == (6, 141)
    np.testing.assert_array_equal(ids['row_in_split'], [0, 2, 3, 0, 2, 3])
    np.testing.assert_array_equal(ids['center_id'], [1, 3, 0, 1, 3, 0])
    np.testing.assert_array_equal(ids['labels'], [0, 0, 1, 0, 0, 1])
    assert all(row['nearest_substitution'] is False for row in evidence)


def test_missing_center_cannot_enter_subset(tmp_path):
    source = fixture_source(tmp_path)
    path = Path(source['subset']['root'])/'subsets/channel_train.npz'
    np.savez(path, row_ids=np.array([0, 1, 3]), center_ids=np.array([1, 0, 0]))
    source['subset']['files']['channel']['train']['sha256'] = runner.sha(path)
    with pytest.raises(AssertionError):
        runner.verify_subset_centers(source, 'channel', 'train')


def test_training_loop_and_budget_are_frozen():
    root = Path(__file__).resolve().parents[1]
    def function(version):
        tree = ast.parse((root/f'experiments/FMT_SingleCenter_Task354C_{version}.py').read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train'))
    assert function('1_1') == function('1_2')
    old = json.loads((root/'config/Verify_FMT_SingleCenter_Task354C_1.1.json').read_text())
    new = json.loads((root/'config/Verify_FMT_SingleCenter_Task354C_1.2.json').read_text())
    for field in ('training', 'seed', 'execution', 'candidate', 'architecture', 'parameters', 'neighbors'):
        assert old[field] == new[field]
