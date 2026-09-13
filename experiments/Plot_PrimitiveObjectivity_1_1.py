"""Analytic seven-line schematic; illustrative coordinates, no measured data."""
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    out = Path('outputs/Verify_FMTObjectivityMechanism_1.1/figure')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial','DejaVu Sans'], 'font.size': 10,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(9.0, 5.0))
    ax = fig.add_axes([.05, .27, .9, .53])
    ax.set(xlim=(-1.3, 9.2), ylim=(-1.75, 2.0), aspect='equal')
    ax.axis('off')
    t = np.linspace(0, 1, 161)
    center = np.column_stack([7.5*t, .35*np.sin(np.pi*t)])
    offsets = np.array([[0,0], [.75,.20], [-.75,-.20], [.1,.92],
                        [-.1,-.92], [-.48,.52], [.48,-.52]])
    paths = []
    for j in range(7):
        theta = .35*t
        a, b = offsets[j]
        off = np.column_stack([a*np.cos(theta)-b*np.sin(theta),
                               a*np.sin(theta)+b*np.cos(theta)])
        path = center + off*(1+.3*t[:,None])
        paths.append(path)
        ax.plot(*path.T, color='#A4A9AE', lw=1.0, zorder=1)
    paths = np.array(paths)
    for k, label in zip([0,80,160], ['t0', 't1', 't2']):
        for j in range(1,7):
            ax.plot(*paths[[0,j],k].T, color='#167DAD', lw=1.5, zorder=2)
        ax.scatter(*paths[1:,k].T, s=26, facecolor='white', edgecolor='#167DAD', zorder=4)
        ax.scatter(*paths[0,k], s=36, color='#20252A', zorder=5)
        ax.text(center[k,0], -1.6, label, ha='center', fontsize=11)
    ax.annotate('', xy=paths[0,80], xytext=paths[0,0],
                arrowprops={'arrowstyle':'->', 'color':'#D05C25',
                            'lw':2.2, 'linestyle':(0,(5,3))}, zorder=6)
    fig.text(.05,.94, 'One centre + six material neighbours', fontsize=16, weight='bold')
    fig.text(.05,.875, 'Seven pathlines; dots on each cross share the same time.', color='#51575D')
    handles = [Line2D([0],[0],color='#167DAD',lw=2,label='Solid blue: same-time separation'),
               Line2D([0],[0],color='#D05C25',lw=2,ls='--',label='Dashed orange: centre displacement across time')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.04,.19), frameon=False, fontsize=10)
    fig.text(.05,.15, 'Objective: distance and same-time angles are unchanged.',color='#167DAD')
    fig.text(.05,.10, 'Not objective: cross-time displacement depends on the moving observer.',color='#B44A1E')
    fig.text(.05,.035, 'Grey curves: illustrative trajectories. The orange rule applies to any of the seven particles.',
             fontsize=8, color='#51575D')
    fig.canvas.draw()
    if len(sys.argv)>1:
        sys.path.insert(0,sys.argv[1])
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=str(out/'primitive.alignment.json'), strict=True)
    fig.savefig(out/'primitive.png',dpi=300)
    fig.savefig(out/'primitive.svg')
    fig.savefig(out/'primitive.pdf')
    np.savez_compressed(out/'illustrative_coordinates.npz',t=t,paths=paths)


if __name__ == '__main__':
    main()
