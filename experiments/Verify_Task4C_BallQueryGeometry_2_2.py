import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from FMT_Utils.Task4C_BallQuery_2_2 import describe

spec = json.loads(Path('config/Ablation_Task4C_BallQuery_2.2.json').read_text())
root = Path(spec['output'])
stats = json.loads(Path('outputs/Verify_Task4C_BallQueryCounts_2.2/local_min/radius_statistics.json').read_text())
record = {}
for flow in ('channel', 'tbl'):
    seeds = np.load(Path('outputs/mainExp_Task4C_FixedDataset_2.1/physical')/flow/'seeds.npy')
    distances, ids = cKDTree(seeds).query(seeds, k=18, workers=4)
    assert np.all(distances[:, 1] > stats['grids'][flow]['h'])
    with np.load(root/'neighbors'/f'neighbors_{flow}.npz') as z:
        report = {}
        for k in (6, 16):
            np.testing.assert_array_equal(np.sort(z[f'order{k}'], axis=1), np.sort(ids[:, 1:k+1], axis=1))
            assert np.all(distances[:, k] < distances[:, k+1])
            report[str(k)] = dict(all_neighbor_sets_equal_nearest_k=True, no_kth_distance_ties=True,
                                  expanded=int(z[f'expanded{k}'].sum()), effective_radius_physical=describe(z[f'effective_radius{k}']))
        with np.load(Path('outputs/Ablation_Task4C_BallQuery_2.2/neighbors')/f'neighbors_{flow}.npz') as local:
            for k in (6, 16): np.testing.assert_array_equal(z[f'order{k}'], local[f'order{k}'])
        record[flow] = dict(global_1h_all_empty=True, active_initial_maximum_count=int(z['initial_count'].max()),
                            global_and_local_h_give_identical_selected_neighbor_sets=True, arms=report)
(root/'h_definition_equivalence.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
