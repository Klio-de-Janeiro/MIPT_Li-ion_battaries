"""Benchmark real training windows, select settings and run a sustained GPU probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import torch
import yaml

from mden_battery.data.article_preprocessing import WindowConfig
from mden_battery.data.window_batches import WindowBatchDataset, choose_cache_device
from mden_battery.loss import MDENJointLoss
from mden_battery.physics import PhysicsConfig
from mden_battery.model import MDEN, MDENConfig
from mden_battery.performance import configure_execution, tune_training_setup
from mden_battery.runtime import configure_precision, build_optimizer, train_epoch_amp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-physics', action='store_true')
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/article_mden.yaml'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batches', type=int, nargs='+', default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument('--amp', choices=['auto', 'float16', 'bfloat16', 'float32'], default='auto')
    parser.add_argument('--compile-scan', action='store_true')
    parser.add_argument('--allow-tf32', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--stress-steps', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path('gpu_benchmark.json'))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is unavailable; GPU speed cannot be measured on this machine')
    if min(args.warmup, args.steps, args.stress_steps, *args.batches) < 1:
        parser.error('Step counts and batch sizes must be positive')
    raw = yaml.safe_load(args.config.read_text())
    config = MDENConfig(**raw['model'])
    loss_config = {'physics': None if args.no_physics else raw.get('physics', {})}
    if not args.no_physics:
        physics = PhysicsConfig(**loss_config['physics'])
        prepared = json.loads((args.generation / 'preprocessing.json').read_text())['config']
        if physics.interval_s != prepared['interval_s'] or physics.rated_capacity_ah != prepared['rated_capacity_ah']:
            parser.error('Physics interval/capacity must match preprocessing')
    window = WindowConfig(**{k: raw['data'][k] for k in ('input_length', 'horizon', 'stride')})
    precision = configure_precision(args.device, amp_dtype=args.amp, allow_tf32=args.allow_tf32)
    cache = choose_cache_device(args.generation, precision.device)
    dataset = WindowBatchDataset(args.generation, split='train', config=window,
                                 shuffle=True, cache_device=cache, physics=not args.no_physics)
    batch, report = tune_training_setup(
        config, dataset, precision, candidates=args.batches,
        compile_scan=args.compile_scan, warmup_steps=args.warmup, timed_steps=args.steps,
        learning_rate=raw['training']['learning_rate'], weight_decay=raw['training']['weight_decay'], loss_config=loss_config)
    dataset.set_batch_size(batch)
    model = MDEN(config).to(precision.device)
    criterion = MDENJointLoss(config.input_dim, **loss_config).to(precision.device)
    configure_execution(model, report['fem_execution'], report['compile_scan'])
    optimizer = build_optimizer(model, criterion,
                                learning_rate=raw['training']['learning_rate'],
                                weight_decay=raw['training']['weight_decay'])

    def batches():
        completed, epoch = 0, 0
        while completed < args.stress_steps:
            dataset.set_epoch(epoch)
            for values in dataset:
                yield values
                completed += 1
                if completed >= args.stress_steps:
                    return
            epoch += 1

    report['stress'] = train_epoch_amp(model, criterion, batches(), optimizer,
                                       precision.scaler(), precision)
    if report['stress']['skipped_steps']:
        raise FloatingPointError('Stress run skipped AMP updates; inspect precision/scale')
    report.update(gpu=torch.cuda.get_device_name(precision.device),
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  amp=str(precision.dtype) if precision.enabled else 'float32',
                  allow_tf32=args.allow_tf32, cache_device=str(cache),
                  training_windows=dataset.total_windows)
    manifest = pl.read_csv(args.generation / 'manifest.csv')
    report['physical_cells_all_splits'] = manifest.height
    report['raw_csv_gb_all_splits'] = manifest['size_bytes'].sum() / 1e9
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    dataset.close()


if __name__ == '__main__':
    main()
