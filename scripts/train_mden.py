"""Train the audited pipeline without editing notebook cells."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from mden_battery.model import MDEN, MDENConfig
from mden_battery.loss import MDENJointLoss
from mden_battery.cache import cleanup_prepared_cache, save_data_recipe
from mden_battery.physics import PhysicsConfig, diagnose_physics_targets
from mden_battery.training import seed_everything
from mden_battery.data.log_age import PrepConfig, prepare_dataset
from mden_battery.data.article_preprocessing import WindowConfig
from mden_battery.data.window_batches import WindowBatchDataset, choose_cache_device
from mden_battery.runtime import configure_precision, build_optimizer, train_epoch_amp, eval_epoch_amp
from mden_battery.performance import configure_execution, tune_training_setup
from mden_battery.checkpoints import TRACKED, new_training_state, complete_epoch, save_epoch, load_inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--no-physics', action='store_true')
    parser.add_argument('--keep-prepared', action='store_true')
    parser.add_argument('--cells', type=int, default=10)
    parser.add_argument('--target-raw-gb', type=float, default=None)
    parser.add_argument('--fem-execution', choices=['auto', 'dynamic', 'static', 'matrix'], default='auto')
    parser.add_argument('--compile-scan', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--monitor', choices=list(TRACKED.values()), default=None)
    parser.add_argument('--config', type=Path, default=Path('configs/article_mden.yaml'))
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    train = raw['training']
    loss_config = {'physics': None if args.no_physics else raw.get('physics', {})}
    args.epochs = args.epochs if args.epochs is not None else int(train['epochs'])
    args.monitor = args.monitor or train.get('monitor', 'loss')
    if (args.batch_size is not None and args.batch_size <= 0) or args.epochs <= 0 or args.monitor not in TRACKED.values():
        parser.error('Positive batch/epochs and a supported monitor are required')
    model_config = MDENConfig(**raw['model'])
    window = WindowConfig(**{key: raw['data'][key] for key in ('input_length', 'horizon', 'stride')})
    if model_config.input_dim != 5 or model_config.horizon != window.horizon:
        parser.error('Model input/horizon must match the LOG_AGE window contract')
    seed = int(train['seed']); seed_everything(seed)
    label = f'{args.cells}_cells' if args.target_raw_gb is None else f'{args.target_raw_gb:g}GB'
    generation, _ = prepare_dataset(PrepConfig(
        args.raw_dir, Path('data')/f'prepared_log_age_30s_{label}_v8',
        cell_limit=args.cells, selection_seed=seed, split_seed=seed,
        target_raw_gb=args.target_raw_gb,
        input_length=window.input_length, horizon=window.horizon, stride=window.stride))
    if not args.no_physics:
        physics = PhysicsConfig(**loss_config['physics'])
        prepared = json.loads((generation / 'preprocessing.json').read_text())['config']
        if physics.interval_s != prepared['interval_s'] or physics.rated_capacity_ah != prepared['rated_capacity_ah']:
            parser.error('Physics interval/capacity must match preprocessing')
    print('Old cache:', cleanup_prepared_cache(generation.parent, dry_run=False))
    precision = configure_precision(args.device)
    cache = choose_cache_device(generation, precision.device)
    datasets = {key: WindowBatchDataset(generation, split=key, config=window, batch_size=args.batch_size or 512,
                                       shuffle=key=='train', seed=seed, cache_device=cache, physics=not args.no_physics) for key in ('train','val','test')}
    args.batch_size, probe = tune_training_setup(
        model_config, datasets['train'], precision, fixed=args.batch_size,
        fem_execution=args.fem_execution, compile_scan=args.compile_scan,
        learning_rate=train['learning_rate'], weight_decay=train['weight_decay'], loss_config=loss_config)
    for dataset in datasets.values():
        dataset.set_batch_size(args.batch_size)
    model = MDEN(model_config).to(args.device); criterion = MDENJointLoss(5, **loss_config).to(args.device)
    execution = configure_execution(model, probe['fem_execution'], probe['compile_scan'])
    optimizer = build_optimizer(model, criterion, learning_rate=train['learning_rate'], weight_decay=train['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=.5)
    scaler = precision.scaler(); state = new_training_state()
    config = dict(monitor=args.monitor, seed=seed, batch_size=args.batch_size, accumulation_steps=1,
                  learning_rate=train['learning_rate'], weight_decay=train['weight_decay'], epochs=args.epochs,
                  patience=train['patience'], gradient_clip_norm=train['gradient_clip_norm'],
                  amp=str(precision.dtype) if precision.enabled else 'off',
                  execution=execution, probe=probe)
    output = args.output or Path('runs')/f'mden_cli_{time.time_ns()}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty run directory')
    output.mkdir(parents=True, exist_ok=True)
    save_data_recipe(generation, output)
    if not args.no_physics:
        report = diagnose_physics_targets(datasets['train'].sample_batch(min(1024, datasets['train'].total_windows)), PhysicsConfig(**loss_config['physics']))
        (output / 'physics_target_check.json').write_text(json.dumps(report, indent=2))
    for epoch in range(1, args.epochs+1):
        started=time.perf_counter(); datasets['train'].set_epoch(epoch)
        tr = train_epoch_amp(model, criterion, datasets['train'], optimizer, scaler, precision,
                             gradient_clip_norm=float(train['gradient_clip_norm']))
        val = eval_epoch_amp(model, criterion, datasets['val'], precision)
        scheduler.step(val[args.monitor])
        improved = complete_epoch(state, tr, val, epoch=epoch, learning_rate=optimizer.param_groups[0]['lr'],
                                  seconds=time.perf_counter()-started, monitor=args.monitor)
        save_epoch(output, model, criterion, optimizer, scheduler, scaler, state, config=config, generation=generation, improved=improved)
        print(epoch, {key:val[key] for key in TRACKED.values()}, flush=True)
        if state['bad_epochs'] >= train['patience']:
            break
    name = next(k for k,v in TRACKED.items() if v==args.monitor)
    model, criterion, _ = load_inference(output/f'best_{name}.pt', args.device)
    metrics = eval_epoch_amp(model, criterion, datasets['test'], precision)
    (output/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
    if not args.keep_prepared:
        for dataset in datasets.values():
            dataset.close()
        print('Cleanup:', cleanup_prepared_cache(generation.parent, include_current=True, dry_run=False))
    print('Results:', output)


if __name__ == '__main__':
    main()
