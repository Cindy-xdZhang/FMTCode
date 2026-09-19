"""Ball-count revision: h is the minimum of the three axis mean grid spacings."""
import argparse
import json
from pathlib import Path
from types import FunctionType
import numpy as np
from experiments import Analyze_Task4C_BallQuery_2_3 as previous

CONFIG = 'config/Verify_Task4C_BallQueryCounts_2.4.json'


def fixed_grids(config, out, spec):
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    evidence = json.loads(Path(config['grid_evidence']).read_text())
    assert evidence['dataset_audit_sha256'] == spec['source_audit_sha256']
    grids = {}
    for name, choice in config['fixed_h'].items():
        old = evidence['grids'][name]; source = Path(old['source'])
        assert previous.sha(source) == old['source_sha256']
        reader = vtk.vtkDataSetReader(); reader.SetFileName(str(source)); reader.Update()
        grid = reader.GetOutput(); assert reader.GetErrorCode() == 0 and grid.IsA('vtkStructuredGrid')
        dims = [0, 0, 0]; grid.GetDimensions(dims); assert dims == choice['dimensions']
        nx, ny, nz = dims
        xyz = vtk_to_numpy(grid.GetPoints().GetData()).reshape(nz, ny, nx, 3)
        axes = [xyz[0, 0, :, 0].astype(np.float64), xyz[0, :, 0, 1].astype(np.float64), xyz[:, 0, 0, 2].astype(np.float64)]
        bounds = [[float(a[0]), float(a[-1])] for a in axes]
        assert np.array_equal(np.asarray(bounds).ravel(), grid.GetBounds())
        mean = [(b-a)/(n-1) for (a, b), n in zip(bounds, dims)]
        selected = int(np.argmin(mean)); h = float(choice['value'])
        assert selected == choice['axis_index'] and abs(h-min(mean)) < 0.5e-9
        spacing = []
        for a in axes:
            d = np.diff(a); assert np.all(d > 0)
            spacing.append(dict(minimum=float(d.min()), maximum=float(d.max()), first_four_coordinates=a[:4].tolist()))
        seeds = np.load(Path(spec['source_output'])/'physical'/name/'seeds.npy', mmap_mode='r')
        values = np.full(len(seeds), h, dtype=np.float64)
        file = out/f'h_{name}.npy'; np.save(file, values)
        grids[name] = dict(h=h, h_mode=config['h_mode'], h_values=previous.ball.describe(values), h_sha256=previous.sha(file),
            definition='minimum of axis means: (axis_max-axis_min)/(grid_points-1); user-confirmed rounded constant',
            dimensions=dims, axis_order=['x', 'y', 'z'], bounds=bounds, mean_spacing=mean,
            selected_axis=choice['axis'], unrounded_h=min(mean), axis_spacing=spacing,
            source=str(source), source_sha256=old['source_sha256'],
            authority='User-confirmed minimum axis mean spacing, 2026-09-19')
        del reader, grid, xyz
    return grids


def run(config_path=CONFIG):
    # Reuse the frozen full-population counting and independent distance verification.
    report = FunctionType(previous.run.__code__, dict(previous.__dict__, fixed_grids=fixed_grids, __file__=__file__),
                          argdefs=previous.run.__defaults__)(config_path)
    report['sources']['experiments/Analyze_Task4C_BallQuery_2_3.py'] = previous.sha(previous.__file__)
    out = Path(json.loads(Path(config_path).read_text())['output'])
    (out/'radius_statistics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=CONFIG)
    run(parser.parse_args().config)
