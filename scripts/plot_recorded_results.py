"""Plot archived aggregate thesis metrics; does not train or validate models."""
from pathlib import Path
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/h060m_b015m/micro_metrics.csv'
MODELS = [
    ('Seasonal daylag', 'Previous-day seasonal'),
    ('ETS local', 'Local ETS'),
    ('Residual Linear ETS', 'ETS + linear correction'),
    ('Residual XGB ETS', 'ETS + XGBoost correction'),
    ('Residual XGB ETS LW', 'ETS + weighted XGBoost'),
]


def main():
    with FIXTURE.open(newline='') as stream:
        rows = {row['']: row for row in csv.DictReader(stream)}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'short-history-recorded-results'})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    fig.patch.set_facecolor('#f6f8fc')
    colors = ['#b1bfd2', '#617793', '#8ea8c9', '#6698ce', '#1765ba']
    for ax, metric, title in zip(axes, ['MAE', 'LW-MAE'], ['Mean absolute error', 'Load-weighted MAE']):
        values = [float(rows[key][metric]) for key, label in MODELS]
        ax.set_facecolor('#f6f8fc')
        ax.barh(range(len(values)), values, color=colors, height=.6, zorder=3)
        ax.set_yticks(range(len(values)), [label for key, label in MODELS])
        ax.set_xlim(0, max(values) * 1.19)
        ax.set_title(title, loc='left', fontweight='bold', color='#152a43', pad=12)
        ax.grid(axis='x', alpha=.18, zorder=0)
        ax.tick_params(axis='both', length=0, labelcolor='#42566c')
        for spine in ax.spines.values(): spine.set_visible(False)
        for i, value in enumerate(values):
            ax.text(value+1, i, f'{value:.2f}', va='center', color='#152a43', fontsize=10)
        ax.set_xlabel('Passenger entries; lower is better', fontsize=9, color='#42566c')
    axes[0].invert_yaxis()
    fig.suptitle('Local forecasts with a shared residual correction', x=.03, y=.97,
                 ha='left', color='#152a43', fontsize=15, fontweight='bold')
    fig.text(.03,.865,'Archived thesis results | 60-minute horizon, 15-minute grid', color='#42566c',fontsize=10)
    fig.text(.03,.035,'Recorded values only. Rerun after target-dependent fallback correction is pending.',
             fontsize=9,color='#7a4a17')
    fig.subplots_adjust(left=.245, right=.975, top=.72, bottom=.19, wspace=.21)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'recorded_results.png',dpi=180)
    fig.savefig(out/'recorded_results.svg', metadata={'Date': None})
    svg = out/'recorded_results.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    print(f'Plotted archived values from {FIXTURE}')

if __name__ == '__main__':
    main()
