"""Read-only local first-window integration check; never a performance result."""
import json
from pathlib import Path
import numpy as np
from FMT_Utils.NetCDF_window_3D import load_netcdf_window_3d
from FMT_Utils.FlowMapData_3D import write_json, sha256
from FMT_Utils.HanFlowMapData_3D import dense_window, partition_roles
from experiments.Build_Task678_FlowMap_1_1 import inspect_source, separated_seed_grid, check_present_frames
from experiments.Build_Task678_HanSampling_1_1 import DEFAULT_CONFIG


def main():
    config = DEFAULT_CONFIG
    spec = json.loads(Path(config).read_text()); s = spec['sampling']
    paths = {
        'halfcylinderRe6400': 'outputs/Verify_AIVDLongtime_1.1/source/halfcylinderRe6400_late96.nc',
        'deltaWing_LBM': 'outputs/Verify_LargeNeighbor_1.1/source/deltaWing_LBM_windows96.nc',
        'f22raptor': 'outputs/Verify_LargeNeighbor_1.1/source/f22raptor_windows96.nc',
        'boeing747': 'outputs/Verify_LargeNeighbor_1.1/source/boeing747_windows96.nc'}
    root = Path('outputs/Verify_Task678_HanDataPreflight_1.1')
    for dataset, path in paths.items():
        if not Path(path).exists():
            continue
        spec['source_fields'][dataset] = path
        source = inspect_source(spec, dataset); row = source['windows'][0]
        check_present_frames(path, row['start_index'], s['source_frames'])
        field, meta = load_netcdf_window_3d(path, row['start_index'], s['source_frames'], s['max_spatial_dim'])
        radius = float(np.min(field.gridInterval)) * s['radius_grid_fraction']
        lo, hi = field.domainMinBoundary + 5.5*radius, field.domainMaxBoundary - 5.5*radius
        grid, shape, gap = separated_seed_grid(lo, hi, radius, s['seed_grid_side'])
        rng = np.random.default_rng(s['data_seed'] + row['ordinal'])
        centers = grid[rng.permutation(len(grid))[:s['max_bundles']]]
        data, audit = dense_window(field, centers, radius, field.tmax/2, s, s['data_seed'])
        roles = partition_roles(data, 'train', s)
        counts = {k:len(v['origin0']) for k,v in roles.items()}
        assert len(data['origin0']) >= s['minimum_bundles']
        assert min(counts.values()) >= s['minimum_role_bundles']
        result = dict(experiment='Verify_Task678_HanDataPreflight_1.1', status='PASS', dataset=dataset,
            scope='local existing source, first predetermined window only; no training or performance selection',
            sampling_config_sha256=sha256(config), source=source, **row, **audit, counts=counts,
            grid_min_center_separation=gap, grid_shape=shape, **meta)
        write_json(root / f'{dataset}.json', result)
        print('LOCAL PHYSICAL DATA PASS', dataset, counts, audit, flush=True)


if __name__ == '__main__':
    main()
