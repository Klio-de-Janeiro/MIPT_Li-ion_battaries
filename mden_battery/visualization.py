from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .runtime import move_batch_to_device


def plot_history(history, output_dir=None):
    epochs = [h['epoch'] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for axis, key, title in zip(axes.flat, ['loss', 'rmse_soc', 'rmse_soh', 'balanced_rmse'],
                                ['Joint loss', 'SOC RMSE, п.п.', 'SOH RMSE, п.п.', 'Balanced RMSE, п.п.']):
        scale = 1 if key == 'loss' else 100
        for split in ('train', 'val'):
            axis.plot(epochs, [h[split][key]*scale for h in history], marker='.', label=split)
        axis.set(title=title, xlabel='Эпоха'); axis.grid(alpha=.2); axis.legend()
    if output_dir is not None:
        fig.savefig(Path(output_dir)/'training_history.png', dpi=150)
    plt.show()
    return fig


@torch.inference_mode()
def collect_examples(model, dataset, precision, count=2, seed=42):
    ids = dataset.example_ids(count, seed)
    batch, metadata = dataset.fetch(ids, metadata=True)
    prior_mode = model.training
    try:
        model.eval()
        x = move_batch_to_device({'x': batch['x']}, precision.device)['x']
        with precision.context():
            outputs = model(x)
        examples = []
        for i, window_id in enumerate(ids):
            row = {'cell_id': dataset.cell_ids[metadata['source'][i]],
                   'split': dataset.split, 'window_id': window_id, 'start': int(metadata['start'][i])}
            for task in ('soc', 'soh'):
                row['history_'+task] = metadata['history_'+task][i].float().cpu().numpy()
                row['true_'+task] = batch['y_'+task][i].float().cpu().numpy()
                row['pred_'+task] = outputs[task][i].float().cpu().numpy()
            examples.append(row)
        return examples
    finally:
        model.train(prior_mode)


def plot_examples(examples, interval_s=30, output_dir=None):
    n = len(examples)
    if not n:
        raise ValueError('No examples')
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.6*n), squeeze=False, layout='constrained')
    for i, row in enumerate(examples):
        for j, task in enumerate(('soc', 'soh')):
            ax = axes[i, j]
            history, truth, pred = [100*row[k+'_'+task] for k in ('history', 'true', 'pred')]
            past_t = np.arange(1-len(history), 1) * interval_s/60
            future_t = np.arange(1, len(truth)+1) * interval_s/60
            ax.plot(past_t, history, '.-', c='#4266c9', label='Метка в истории (не вход)')
            ax.plot(future_t, truth, 'o-', c='#242424', label='Целевая метка')
            ax.plot(future_t, pred, 's--', c='#d44b4b', label='Прогноз')
            ax.axvline(0, c='gray', ls=':'); ax.axvspan(0, future_t[-1], color='#9675ff', alpha=.1)
            rmse = np.sqrt(np.mean((pred-truth)**2))
            ax.set(title=f'{row["cell_id"]} | {task.upper()} RMSE={rmse:.3f} п.п.',
                   xlabel='Конец блока, мин относительно границы входа', ylabel=f'{task.upper()}, %')
            ax.grid(alpha=.2); ax.legend(fontsize=8)
    split = examples[0]['split']
    fig.suptitle(f'{split.upper()}: 32×5 входных признаков → следующие 8 SOC и SOH\n'
                 'KIT: SOC — интервальная оценка; SOH — разметка по ёмкости. Прогноз без clipping.')
    if output_dir is not None:
        path = Path(output_dir)/f'predictions_{split}.png'
        fig.savefig(path, dpi=150)
        # Raw values allow independent checking of the plot.
        records = []
        for row in examples:
            for h in range(len(row['true_soc'])):
                records.append([row['cell_id'], row['window_id'], h+1,
                                row['true_soc'][h], row['pred_soc'][h], row['true_soh'][h], row['pred_soh'][h]])
        import csv
        with (Path(output_dir)/f'predictions_{split}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream); writer.writerow(['cell_id','window_id','horizon','true_soc','pred_soc','true_soh','pred_soh']); writer.writerows(records)
    plt.show()
    return fig
