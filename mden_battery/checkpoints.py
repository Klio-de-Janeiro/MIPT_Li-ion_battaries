from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .model import MDEN, MDENConfig
from .loss import MDENJointLoss

CHECKPOINT_VERSION = 3
TRACKED = {'joint': 'loss', 'soc': 'rmse_soc', 'soh': 'rmse_soh', 'balanced': 'balanced_rmse'}


def capture_rng():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    s = state['numpy']
    np.random.set_state((s[0], np.array(s[1], dtype=np.uint32), *s[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def atomic_save(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def new_training_state():
    return {'epoch': 0, 'best': {k: float('inf') for k in TRACKED},
            'best_epoch': {k: None for k in TRACKED}, 'bad_epochs': 0, 'history': []}


def complete_epoch(state, train, val, *, epoch, learning_rate, seconds, monitor):
    if monitor not in TRACKED.values():
        raise ValueError(f'Unknown monitor {monitor}')
    if not all(np.isfinite(val[key]) for key in TRACKED.values()):
        raise FloatingPointError('Non-finite validation metrics')
    improved = []
    for name, metric in TRACKED.items():
        if val[metric] < state['best'][name]:
            state['best'][name] = val[metric]
            state['best_epoch'][name] = epoch
            improved.append(name)
    chosen = next(k for k, v in TRACKED.items() if v == monitor)
    state['bad_epochs'] = 0 if chosen in improved else state['bad_epochs'] + 1
    state['epoch'] = epoch
    state['history'].append(dict(epoch=epoch, train=train, val=val, learning_rate=learning_rate, seconds=seconds))
    return improved


def save_epoch(run_dir, model, criterion, optimizer, scheduler, scaler, state,
               *, config, generation, improved):
    run_dir, generation = Path(run_dir), Path(generation)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {'version': CHECKPOINT_VERSION, 'model': model.state_dict(),
               'criterion': criterion.state_dict(), 'loss_config': criterion.loss_config(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'grad_scaler': scaler.state_dict(),
               'model_config': asdict(model.config), 'resolved_mamba_backend': model.soh_mamba.backend,
               'config': config, 'training_state': state, 'rng': capture_rng(),
               'feature_scaler': json.loads((generation/'scaler.json').read_text()),
               'preprocessing': json.loads((generation/'preprocessing.json').read_text()),
               'torch_version': str(torch.__version__)}
    atomic_save(run_dir/'last.pt', payload)
    for name in improved:
        atomic_save(run_dir/f'best_{name}.pt', payload)
    history = run_dir/'history.json'
    tmp = history.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(state['history'], ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(history)


def load_inference(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Old checkpoint architecture. Retrain using the audited project.')
    config = dict(payload['model_config'])
    # Both kernels use identical parameter names. CPU always uses native scan.
    config['mamba_backend'] = 'native' if torch.device(device).type == 'cpu' else payload['resolved_mamba_backend']
    model = MDEN(MDENConfig(**config)).to(device)
    model.load_state_dict(payload['model'], strict=True)
    if torch.device(device).type == 'cuda':
        from .performance import configure_execution
        execution = payload.get('config', {}).get('execution', {})
        configure_execution(model, execution.get('fem_execution', 'dynamic'), False)
    criterion = MDENJointLoss(config['input_dim'], **payload.get('loss_config', {})).to(device)
    criterion.load_state_dict(payload['criterion'], strict=True)
    model.eval(); criterion.eval()
    return model, criterion, payload


def resume_training(path, model, criterion, optimizer, scheduler, scaler, config, generation):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('version') != CHECKPOINT_VERSION:
        raise ValueError('Checkpoint version mismatch')
    saved_loss_config = payload.get('loss_config', {'sigma_floor': 1e-4, 'physics': None})
    if saved_loss_config != criterion.loss_config():
        raise ValueError('Resume mismatch: loss/physics configuration; start a new experiment')
    if payload['model_config'] != asdict(model.config):
        raise ValueError('Cannot resume with another model configuration')
    if payload['resolved_mamba_backend'] != model.soh_mamba.backend:
        raise ValueError('Cannot resume with another resolved Mamba backend')
    if payload['preprocessing'] != json.loads((Path(generation)/'preprocessing.json').read_text()):
        raise ValueError('Cannot resume with another dataset/scaler')
    if payload['feature_scaler'] != json.loads((Path(generation)/'scaler.json').read_text()):
        raise ValueError('Cannot resume with another feature scaler')
    for key in ('monitor', 'batch_size', 'accumulation_steps', 'seed',
                'amp', 'learning_rate', 'weight_decay', 'gradient_clip_norm', 'patience'):
        if payload['config'].get(key) != config.get(key):
            raise ValueError(f'Resume mismatch: {key}')
    if payload['config'].get('allow_tf32', False) != config.get('allow_tf32', False):
        raise ValueError('Resume mismatch: allow_tf32')
    model.load_state_dict(payload['model']); criterion.load_state_dict(payload['criterion'])
    optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
    scaler.load_state_dict(payload['grad_scaler'])
    restore_rng(payload['rng'])
    return payload['training_state']
