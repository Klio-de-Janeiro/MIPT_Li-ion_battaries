import json
from dataclasses import asdict, replace
import subprocess
import sys

import numpy as np
import pytest
import torch

from mden_battery.physics import PhysicsConfig, charge_increment, physics_terms, diagnose_physics_targets
from mden_battery.loss import MDENJointLoss
from mden_battery.cache import cleanup_prepared_cache, save_data_recipe
from mden_battery.data.window_batches import WindowBatchDataset
from mden_battery.data.article_preprocessing import WindowConfig
from mden_battery.data.log_age import prepare_dataset, prepared_values, resample_query
from test_audit_regressions import prep_fixture, raw_frame


@pytest.mark.parametrize('sign', [1, -1])
def test_balance_for_interval_means_including_current_step(sign):
    # Analytic exact solution: z mean in each constant-current bin equals
    # the midpoint of its start/end z. A current step needs HALF each bin.
    cfg = PhysicsConfig(charge_sign=sign, residual_tolerance=0)
    current = torch.tensor([[1., 1., -2., 0., 3., -1., 0., .5]])
    soh = torch.full_like(current, .8)
    dz = sign * current * 30 / (3600 * 2.4)
    z = .5 + dz.cumsum(1) - dz / 2
    torch.testing.assert_close(z.diff(dim=1), charge_increment(current, soh, cfg), atol=5e-8, rtol=1e-5)
    terms = physics_terms({'soc': z, 'soh': soh}, soh, current, cfg)
    assert terms['charge_loss'] < 1e-9
    # Endpoint convention is not interchangeable with interval mean targets.
    endpoints = .5 + dz.cumsum(1)
    assert physics_terms({'soc': endpoints, 'soh': soh}, soh, current, cfg)['charge_loss'] > .01


def test_charge_penalty_has_soc_gradient_and_cannot_inflate_predicted_soh():
    cfg = PhysicsConfig(bounds_weight=0, residual_tolerance=0)
    soc = torch.full((2, 8), .5, requires_grad=True)
    pred_soh = torch.full((2, 8), .9, requires_grad=True)
    reference = torch.full((2, 8), .8, requires_grad=True)
    current = torch.ones(2, 8, requires_grad=True)
    result = physics_terms({'soc': soc, 'soh': pred_soh}, reference, current, cfg)
    result['physics_loss'].backward()
    assert soc.grad.abs().sum() > 0
    assert pred_soh.grad is None or torch.count_nonzero(pred_soh.grad) == 0
    assert reference.grad is None and current.grad is None
    inflated = physics_terms({'soc': soc, 'soh': pred_soh * 100}, reference, current, cfg)
    torch.testing.assert_close(result['charge_loss'], inflated['charge_loss'])


def test_missing_current_and_bad_physics_config_fail_instead_of_silent_skip():
    loss = MDENJointLoss(5, physics={})
    features = torch.zeros(2, 32, 5)
    pred = dict(soc=torch.ones(2, 8)*.5, soh=torch.ones(2, 8), soc_fused=features, soh_fused=features)
    with pytest.raises(ValueError, match='no physics_current_a'):
        loss(pred, pred['soc'], pred['soh'])
    for kwargs in ({'charge_weight': -1}, {'interval_s': 0}, {'residual_scale': float('nan')}):
        with pytest.raises(ValueError):
            PhysicsConfig(**kwargs)
    with pytest.raises(ValueError, match='positive reference SOH'):
        charge_increment(torch.ones(1, 8), torch.zeros(1, 8), PhysicsConfig())


def test_future_current_is_denormalized_but_not_part_of_x(tmp_path):
    cfg = prep_fixture(tmp_path)
    generation, _ = prepare_dataset(cfg)
    base = WindowBatchDataset(generation, split='train', config=WindowConfig())
    physical = WindowBatchDataset(generation, split='train', config=WindowConfig(), physics=True)
    a = base.fetch([0, 1]); b = physical.fetch([0, 1])
    assert set(b) - set(a) == {'physics_current_a'}
    torch.testing.assert_close(a['x'], b['x'])
    torch.testing.assert_close(a['y_soc'], b['y_soc'])
    # Same raw fixture for every cell: future indices 32..39 have cos(t/500).
    expected = np.cos(np.arange(32, 40)*30/500)
    np.testing.assert_allclose(b['physics_current_a'][0], expected, atol=2e-7)
    report = diagnose_physics_targets(b, PhysicsConfig())
    assert report['transitions'] == 14 and report['target_charge_rmse_pp'] > 0
    base.close(); physical.close()


def test_cache_cleanup_respects_active_handles_and_preserves_raw_and_recipe(tmp_path):
    cfg = prep_fixture(tmp_path)
    raw_contents = {p.name: p.read_bytes() for p in cfg.raw_dir.iterdir()}
    old, _ = prepare_dataset(cfg)
    current, _ = prepare_dataset(cfg, rebuild=True)
    dataset = WindowBatchDataset(current, split='train', config=WindowConfig())
    expected = dataset.fetch([0])
    recipe = save_data_recipe(current, tmp_path / 'run')
    preview = cleanup_prepared_cache(cfg.prepared_dir)
    assert preview[0]['status'] == 'would_delete' and old.exists()
    report = cleanup_prepared_cache(cfg.prepared_dir, include_current=True, dry_run=False)
    assert not old.exists() and current.exists()
    assert any(r['status'] == 'skipped_in_use_or_unrecognized' for r in report)
    dataset.close()
    with pytest.raises(RuntimeError, match='closed'):
        dataset.fetch([0])
    cleanup_prepared_cache(cfg.prepared_dir, include_current=True, dry_run=False)
    assert not current.exists() and not (cfg.prepared_dir / 'CURRENT.json').exists()
    assert (recipe / 'scaler.json').exists()
    assert raw_contents == {p.name: p.read_bytes() for p in cfg.raw_dir.iterdir()}
    rebuilt, reused = prepare_dataset(cfg)
    assert not reused
    again = WindowBatchDataset(rebuilt, split='train', config=WindowConfig())
    for key, value in again.fetch([0]).items():
        torch.testing.assert_close(expected[key], value, rtol=0, atol=0)
    again.close()


def test_cleanup_refuses_unknown_files_and_symlinks(tmp_path):
    cfg = prep_fixture(tmp_path)
    gen, _ = prepare_dataset(cfg)
    (gen / 'my_notes.txt').write_text('must survive')
    cleanup_prepared_cache(cfg.prepared_dir, include_current=True, dry_run=False)
    assert (gen / 'my_notes.txt').read_text() == 'must survive'
    (gen / 'my_notes.txt').unlink()
    target = tmp_path / 'important'; target.mkdir(); (target / 'keep').write_text('keep')
    (gen / 'arrays' / 'P999_9.npy').symlink_to(target / 'keep')
    cleanup_prepared_cache(cfg.prepared_dir, include_current=True, dry_run=False)
    assert (target / 'keep').read_text() == 'keep' and gen.exists()


def test_os_lease_prevents_cleanup_by_another_process(tmp_path):
    cfg = prep_fixture(tmp_path)
    gen, _ = prepare_dataset(cfg)
    dataset = WindowBatchDataset(gen, split='train', config=WindowConfig())
    script = ('from mden_battery.cache import cleanup_prepared_cache; '
              'import sys; print(cleanup_prepared_cache(sys.argv[1], include_current=True, dry_run=False))')
    result = subprocess.run([sys.executable, '-c', script, str(cfg.prepared_dir)], text=True, capture_output=True, check=True)
    assert 'skipped_in_use' in result.stdout and gen.exists()
    dataset.close()


def test_optional_anchor_gap_limit_excludes_only_long_interpolation(tmp_path):
    cfg = replace(prep_fixture(tmp_path), max_anchor_gap_s=1200)
    frame = resample_query(raw_frame().lazy(), 30, cfg).collect()
    values, offsets, _ = prepared_values(frame, (np.array([30., 1230., 3000., 3600.]),
                                                np.array([3., 2.9, 2.8, 2.7])), cfg)
    assert values[:, 1].max() == 1230  # last short segment has <40 rows
    assert np.all((values[:, 1] >= 30) & (values[:, 1] <= 1230))


def test_physics_loss_training_checkpoint_roundtrip_and_resume_guard(tmp_path):
    from mden_battery.checkpoints import save_epoch, load_inference, resume_training, new_training_state
    from mden_battery.model import MDEN, MDENConfig
    from mden_battery.runtime import build_optimizer, train_epoch_amp, configure_precision
    cfg = prep_fixture(tmp_path); gen, _ = prepare_dataset(cfg)
    ds = WindowBatchDataset(gen, split='train', config=WindowConfig(), physics=True)
    model = MDEN(MDENConfig(soc_depth=1, shared_depth=1, soh_depth=1, conv_rounds=1,
                           top_k=1, fem_hidden_channels=8, fem_groups=2))
    loss = MDENJointLoss(5, physics=PhysicsConfig())
    precision = configure_precision('cpu'); scaler = precision.scaler()
    optimizer = build_optimizer(model, loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    batch = ds.fetch([0, 1, 2])
    metrics = train_epoch_amp(model, loss, [batch], optimizer, scaler, precision)
    assert metrics['physics_loss'] > 0 and metrics['samples'] == 3
    run = tmp_path / 'run'
    save_epoch(run, model, loss, optimizer, scheduler, scaler, new_training_state(),
               config={}, generation=gen, improved=[])
    restored, restored_loss, payload = load_inference(run / 'last.pt')
    assert restored_loss.loss_config() == loss.loss_config()
    model.eval()
    with torch.no_grad():
        a = loss.from_batch(model(batch['x']), batch)
        b = restored_loss.from_batch(restored(batch['x']), batch)
    for key in a:
        torch.testing.assert_close(a[key], b[key])
    with pytest.raises(ValueError, match='loss/physics'):
        resume_training(run / 'last.pt', model, MDENJointLoss(5), optimizer, scheduler, scaler, {}, gen)
    ds.close()
