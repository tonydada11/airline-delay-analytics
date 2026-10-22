"""Export a compact, reproducible figure from completed monthly diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    code = Path(__file__).resolve().parent
    study = code.parents[2]
    months = ('2024-05', '2024-06', '2024-07', '2024-08')
    data = [json.loads((study / 'diagnostics' / f'validation-{m}.json').read_text()) for m in months]
    x = list(range(4))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    pr_auc = [d['pr_auc'] for d in data]
    axes[0].plot(x, pr_auc, 'o-', color='#2166ac', label='Monthly PR-AUC')
    axes[0].axhline(sum(pr_auc) / 4, linestyle='--', color='#444444',
                   label=f'Equal-month mean {sum(pr_auc)/4:.6f}')
    axes[0].plot(x, [d['prevalence'] for d in data], 's:', color='#999999', label='Monthly prevalence')
    axes[0].set_title('Ranking and base rate')
    axes[0].set_ylim(0, 0.6)
    axes[0].legend(fontsize=8)
    for percent, color in ((10, '#2166ac'), (20, '#b2182b')):
        for metric, style in (('precision', '-'), ('recall', '--')):
            axes[1].plot(x, [d['budgets'][str(percent)][metric] for d in data],
                         marker='o', linestyle=style, color=color,
                         label=f'{metric.capitalize()} at top {percent}%')
    axes[1].set_title('Monthly alert budgets')
    axes[1].set_ylim(0, 0.65)
    axes[1].legend(fontsize=8)
    for label, color in ((0, '#2166ac'), (1, '#b2182b')):
        stats = [next(s for s in d['score_distribution']['by_label'] if s['is_arrival_delayed_15m'] == label) for d in data]
        axes[2].plot(x, [s['quantiles'][3] for s in stats], 'o-', color=color,
                     label=f'Label {label}: median and IQR')
        axes[2].fill_between(x, [s['quantiles'][2] for s in stats], [s['quantiles'][4] for s in stats],
                             alpha=0.15, color=color)
    axes[2].axhline(0.39, linestyle=':', color='#333333', label='Accepted threshold 0.39')
    axes[2].set_title('Weighted LR score distributions')
    axes[2].set_ylim(0, 0.8)
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xticks(x, ['May', 'June', 'July', 'August'])
        axis.grid(alpha=0.2)
        axis.set_xlabel('Validation month, 2024')
    figure.suptitle('Accepted LR baseline: retrospective validation diagnostics', fontsize=13)
    figure.savefig(code / 'diagnostics.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    main()
