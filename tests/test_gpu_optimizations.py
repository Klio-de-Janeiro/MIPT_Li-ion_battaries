import copy
import json

import numpy as np
import pytest
import torch

from mden_battery.feature_extraction import FeatureExtractionModule
from mden_battery.model import MDEN, MDENConfig
from mden_battery.loss import MDENJointLoss
from mden_battery.performance import configure_execution
from mden_battery.runtime import MetricAccumulator, prefetch_batches
from mden_battery.data.log_age import (
    PrepConfig, discover_sources, prepare_dataset, validate_prepared_dataset,
)
from test_audit_regressions import raw_frame


@pytest.mark.parametrize('length', [7, 32, 33])
@pytest.mark.parametrize('mode', ['static', 'matrix'])
def test_fem_execution_outputs_and_all_gradients(length, mode):
    torch.manual_seed(71)
    ref = FeatureExtractionModule(3, top_k=length // 2,
                                  hidden_channels=8, groups=2).double()
    fast = copy.deepcopy(ref)
    fast.execution = mode
    x = torch.randn(2, length, 3, dtype=torch.double, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    a, b = ref(x), fast(z)
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, z.grad, atol=1e-11, rtol=1e-9)
    for p, q in zip(ref.parameters(), fast.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize('mode', ['static', 'matrix'])
def test_full_model_and_loss_parameter_gradients(mode):
    torch.manual_seed(4)
    ref = MDEN(MDENConfig(top_k=3, fem_hidden_channels=8, fem_groups=2))
    fast = copy.deepcopy(ref)
    configure_execution(fast, mode)
    criterion = MDENJointLoss(5)
    other = copy.deepcopy(criterion)
    x = torch.randn(3, 32, 5)
    targets = (torch.rand(3, 8), torch.rand(3, 8))
    a, b = ref(x), fast(x)
    for task in ('soc', 'soh'):
        torch.testing.assert_close(a[task], b[task], atol=2e-5, rtol=2e-4)
    criterion(a, *targets)['loss'].backward()
    other(b, *targets)['loss'].backward()
    for p, q in zip(list(ref.parameters()) + list(criterion.parameters()),
                    list(fast.parameters()) + list(other.parameters())):
        torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-3)
    assert list(ref.state_dict()) == list(fast.state_dict())
    fast.eval()
    with torch.no_grad():
        torch.testing.assert_close(fast(x)['soc'][:1], fast(x[:1])['soc'],
                                   atol=2e-5, rtol=2e-4)


def test_matrix_rejects_nonlinear_fem():
    model = MDEN(MDENConfig(fem_activation='gelu'))
    with pytest.raises(ValueError, match='identity'):
        configure_execution(model, 'matrix')


def test_matrix_rejects_amp_rearrangement():
    layer = FeatureExtractionModule(5)
    layer.execution = 'matrix'
    with torch.autocast('cpu', dtype=torch.bfloat16):
        with pytest.raises(ValueError, match='autocast disabled'):
            layer(torch.randn(2, 32, 5))


def test_batched_kernel_fusion_matches_individual_rounds():
    layer = FeatureExtractionModule(5, hidden_channels=8, groups=2).double()
    fused = layer._kernels()
    references = [r.fused_parameters() for r in layer.rounds]
    for pair, reference in zip(fused, references):
        for a, b in zip(pair, reference):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    x = torch.randn(3, 5, 3, 11, dtype=torch.double)
    y = x
    for r, kernel in zip(layer.rounds, fused):
        y = r.apply_fused(y, kernel)
    expected = layer.rounds(x)
    torch.testing.assert_close(y, expected, atol=1e-12, rtol=1e-10)
    a = torch.autograd.grad(y.square().sum(), list(layer.parameters()), retain_graph=True)
    b = torch.autograd.grad(expected.square().sum(), list(layer.parameters()))
    for p, q in zip(a, b):
        torch.testing.assert_close(p, q, atol=1e-10, rtol=1e-8)


def test_shared_first_fft_preserves_outputs_and_gradients():
    from mden_battery.feature_extraction import MDENFeatureExtractor
    model = MDENFeatureExtractor(hidden_channels=8, groups=2,
                                 soc_depth=1, shared_depth=1, soh_depth=1).double()
    x = torch.randn(2, 32, 5, dtype=torch.double, requires_grad=True)
    actual = model(x)
    expected = {'soc_specific': model.soc_net(x), 'shared': model.shared_net(x),
                'soh_specific': model.soh_net(x)}
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
    a = torch.autograd.grad(sum(v.square().sum() for v in actual.values()), x)[0]
    b = torch.autograd.grad(sum(v.square().sum() for v in expected.values()), x)[0]
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-8)


def test_streamed_metrics_match_direct_reference():
    torch.manual_seed(11)
    targets = 0.85 + torch.randn(17, 8) * 1e-3
    predictions = targets + torch.randn(17, 8) * 2e-3
    accumulator = MetricAccumulator(8, 'cpu')
    for start, end in ((0, 13), (13, 17)):
        y, p = targets[start:end], predictions[start:end]
        err = (p - y).square().mean()
        batch = {'x': torch.zeros(len(y), 32, 5), 'y_soc': y, 'y_soh': y}
        accumulator.add({'soc': p, 'soh': p},
                        {'loss': err, 'mse_soc': err, 'mse_soh': err}, batch)
    result = accumulator.result()
    expected = 1 - (predictions.double() - targets.double()).square().sum() / (
        targets.double() - targets.double().mean()).square().sum()
    assert result['r2_soc'] == pytest.approx(float(expected), abs=5e-5)
    assert result['samples'] == 17


def test_budget_uses_physical_cells_and_preferred_30s_source(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for i in range(6):
        raw_frame(120 + i * 10).write_csv(
            raw / f'cell_log_age_30s_P{i:03d}_1_S01_C01.csv', separator=';')
    # This duplicate must not count toward the budget or be parsed.
    (raw / 'cell_log_age_2s_P000_1_S01_C01.csv').write_text('duplicate')
    sources = discover_sources(raw)
    assert sources.height == 6
    budget = sources['size_bytes'].sum() * 0.65 / 1e9
    cfg = PrepConfig(raw, tmp_path / 'prepared', target_raw_gb=budget,
                     show_progress=False)
    generation, reused = prepare_dataset(cfg)
    assert not reused
    index = validate_prepared_dataset(generation, cfg)
    manifest = json.loads((generation / 'preprocessing.json').read_text())['sources']
    assert sum(r['size_bytes'] for r in manifest) >= budget * 1e9
    assert index.height >= 3
    assert all(r['sampling_interval_s'] == 30 for r in manifest)
    assert prepare_dataset(cfg)[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_prefetch_keeps_order_values_and_final_partial_batch():
    batches = [{'x': torch.full((n, 32, 5), float(i))}
               for i, n in enumerate([17, 17, 3])]
    received = list(prefetch_batches(iter(batches), torch.device('cuda:0')))
    assert len(received) == len(batches)
    for expected, actual in zip(batches, received):
        torch.testing.assert_close(expected['x'], actual['x'].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_compiled_scan_outputs_and_gradients():
    from mden_battery.mamba import selective_scan_native
    from mden_battery.performance import _compiled_scan
    inputs = [torch.randn(2, 32, 3, device='cuda'),
              torch.rand(2, 32, 3, device='cuda') * .1,
              -torch.rand(3, 4, device='cuda'),
              torch.randn(2, 32, 4, device='cuda'),
              torch.randn(2, 32, 4, device='cuda'), torch.randn(3, device='cuda')]
    a = [x.requires_grad_() for x in inputs]
    b = [x.detach().clone().requires_grad_() for x in inputs]
    y = selective_scan_native(*a)
    z = _compiled_scan()(*b)
    torch.testing.assert_close(y, z, atol=2e-5, rtol=2e-4)
    y.square().mean().backward()
    z.square().mean().backward()
    for x, v in zip(a, b):
        torch.testing.assert_close(x.grad, v.grad, atol=2e-5, rtol=2e-4)
