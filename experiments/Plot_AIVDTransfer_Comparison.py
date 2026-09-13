"""Show paired F1 changes under unchanged Task3 feature transfer.

Question: does the scalar transfer improve direct and learned clustering,
and does retaining the original kin4 addon change that effect?
Evidence: Task1 direct clustering and Task2 learned representation, with the
predeclared core-replacement control in each panel. Quantitative grid; all ten
datasets and five paired seeds. No significance test or outcome filtering.
Python/matplotlib, 183 x 115 mm, editable PDF/SVG, 600-dpi PNG.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--skill-root', default=os.environ.get('NATURE_FIGURE_SKILL_ROOT'))
    args = parser.parse_args()
    if not args.skill_root:
        raise ValueError('Set figure QA skill root')
    sys.path.insert(0, str(Path(args.skill_root)/'scripts'))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    root = Path(args.root)
    audit = json.loads((root/'local_aggregate_audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['underlying_metric_rows'] == 350
    for name in ('dataset_metrics.csv', 'paired_comparisons.csv', 'task_macro.csv'):
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == audit['source_sha256'][name]
    with (root/'dataset_metrics.csv').open() as f:
        dataset_rows = list(csv.DictReader(f))
    with (root/'paired_comparisons.csv').open() as f:
        paired_rows = list(csv.DictReader(f))
    with (root/'task_macro.csv').open() as f:
        macro_rows = list(csv.DictReader(f))
    datasets = ['channel', 'cylinder3d', 'halfcylinderRe640', 'halfcylinderRe6400', 'tangaroa',
                'deltaWing_resampled', 'deltaWing_LBM', 'f22raptor', 'boeing747', 'smokeBuoyancy']
    names = ['Channel', 'Half-cylinder Re160', 'Half-cylinder Re640', 'Half-cylinder Re6400',
             'Tangaroa', 'Delta wing (resampled)', 'Delta wing (LBM)', 'F-22', 'Boeing 747', 'Buoyant smoke']
    tasks = ['Task1', 'Task2']
    arms = ['aivd', 'aivd_kin4']
    colors = {'aivd': '#087E8B', 'aivd_kin4': '#A96C45'}
    markers = {'aivd': 'o', 'aivd_kin4': 'D'}
    source, effects = [], {}
    for task in tasks:
        for dataset in datasets+['macro']:
            for arm in arms:
                if dataset == 'macro':
                    selected = [r for r in macro_rows if r['task'] == task]
                    paired = next(r for r in selected if r['arm'] == arm)
                    std_field = 'delta_f1_macro_seed_std'
                else:
                    selected = [r for r in dataset_rows if r['task'] == task and r['dataset'] == dataset]
                    paired = next(r for r in paired_rows if r['task'] == task and r['dataset'] == dataset and r['new_arm'] == arm)
                    std_field = 'delta_f1_std'
                new = next(float(r['f1_mean']) for r in selected if r['arm'] == arm)
                old = next(float(r['f1_mean']) for r in selected if r['arm'] == 'old_fmt')
                mean, std = 100*float(paired['delta_f1_mean']), 100*float(paired[std_field])
                assert abs(mean-100*(new-old)) < 1e-10
                effects[task, dataset, arm] = (mean, std)
                source.append({'task': task, 'dataset': dataset, 'arm': arm, 'seeds': 5,
                               'old_f1_mean': old, 'new_f1_mean': new,
                               'delta_f1_percentage_points_mean': mean,
                               'delta_f1_percentage_points_sample_std': std})
    limit = max(abs(mean)+std for mean, std in effects.values())
    limit = max(10, 10*np.ceil((limit+3)/10))
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .65, 'legend.frameon': False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.20472440945, 4.52755905512), sharey=True,
                             gridspec_kw={'left': .258, 'right': .985, 'bottom': .18, 'top': .80, 'wspace': .18})
    for panel, (ax, task) in enumerate(zip(axes, tasks)):
        for index, dataset in enumerate(datasets+['macro']):
            y = index if dataset != 'macro' else 11
            for arm, offset in [('aivd', -.17), ('aivd_kin4', .17)]:
                mean, std = effects[task, dataset, arm]
                ax.errorbar(mean, y+offset, xerr=std,
                            fmt=markers[arm], color=colors[arm], markersize=3.3,
                            markeredgewidth=.5, capsize=1.5, elinewidth=.8, zorder=3)
        ax.axvline(0, color='#78818A', lw=.8, zorder=1)
        ax.axhline(10.25, color='#D7DCE0', lw=.6)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(11.65, -.65)
        ax.set_xticks(np.linspace(-limit, limit, 5))
        ax.set_xlabel('Change in F1 (percentage points)')
        ax.set_title('Task1: direct clustering' if task == 'Task1' else 'Task2: learned representation', pad=10)
        ax.set_yticks(list(range(10))+[11])
        ax.set_yticklabels(names+['Mean of 10 entries'])
        ax.tick_params(axis='y', length=0, pad=5)
        ax.tick_params(axis='x', length=2.5)
        ax.spines['left'].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E8EBED', lw=.5)
        ax.text(-.015, 1.095, 'ab'[panel], transform=ax.transAxes, fontsize=8, fontweight='bold', va='bottom')
    fig.text(.258, .960, 'Transferring the Task3 feature to Task1 and Task2', fontsize=9, va='top')
    handles = [Line2D([], [], color=colors[a], marker=markers[a], linestyle='none', markersize=4,
                     label='aivd1w3_dft only' if a == 'aivd' else 'aivd1w3_dft + kin4') for a in arms]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.258, .925), ncol=2,
               handletextpad=.5, columnspacing=1.8)
    fig.text(.258, .085, 'Relative to retrained fmt_all + kin4. Mean ± 1 s.d. of 5 paired seed differences.', fontsize=6.5)
    fig.text(.258, .047, 'Cylinder: t ≥ 7.5 in every split. All 10 entries; no outcome filtering.', fontsize=6.5)
    out = root/'figures'
    out.mkdir(exist_ok=True)
    base = out/'aivd_transfer_comparison'
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(base)+'.alignment.json',
        overlay_svg=str(base)+'.alignment.svg', tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(f'{base}.pdf', facecolor='white')
    fig.savefig(f'{base}.svg', facecolor='white')
    fig.savefig(f'{base}.png', dpi=600, facecolor='white')
    plt.close(fig)
    with (out/'source_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(source[0]))
        writer.writeheader()
        writer.writerows(source)
    (out/'caption.md').write_text(
        'Task3 feature transfer to direct clustering (Task1) and variational-autoencoder representation '
        'learning followed by clustering (Task2). Each point is the mean change in F1 relative to the '
        'retrained original fmt_all+kin4 input; bars show one sample standard deviation of five paired '
        'random-seed differences. Positive changes favour the '
        'transferred representation. The standalone arm uses the one-dimensional aivd1w3_dft; the '
        '29-dimensional control retains the original kin4 addon. All ten 3D flow entries and every '
        'registered FMT run are included. The final row averages the ten entries within each seed '
        'before computing the same mean and standard deviation. Cylinder source slices satisfy t>=7.5 '
        'in all roles, with original train/validation/test roles retained. Ground truth uses frozen '
        'whole-field instantaneous vorticity deviation (IVD) p95 references. The input is a geometric '
        'IVD estimate, not the ground-truth label. Both new arms retain the original index-time '
        'derivatives. Task1 scalar input skips dimensionality reduction; other arms retain PCA8. '
        'Task2 uses the same hidden architecture and 7000 updates in all arms; input/output widths '
        'and parameter counts differ. This is an already used benchmark, not a fresh confirmation. '
        'No significance test is performed. The 50 Task2 Raw-control rows are reported in the full '
        'tables, not drawn in this FMT-replacement contrast. Individual runs are represented by '
        'their audited paired mean and standard deviation; individual predictions remain on Ibex. '
        'Source data: source_data.csv, ../paired_comparisons.csv, ../dataset_metrics.csv and '
        '../task_macro.csv.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
