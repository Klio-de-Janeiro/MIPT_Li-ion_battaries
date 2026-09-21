import math

import torch

from mden_battery.feature_extraction import (
    FeatureExtractionModule,
    MDENFeatureExtractor,
    fft_select_periods,
)


def test_fft_selects_known_dominant_bin():
    t = torch.arange(32, dtype=torch.float32)
    signal = torch.sin(2 * math.pi * 4 * t / 32)
    x = signal[None, :, None].repeat(3, 1, 5)
    selected = fft_select_periods(x, top_k=1)
    assert selected.frequency_indices.tolist() == [[4], [4], [4]]
    assert selected.periods.tolist() == [[8], [8], [8]]
    assert selected.scores.shape == (3, 1)


def test_one_fem_contains_four_conv_rounds_and_preserves_shape():
    fem = FeatureExtractionModule(
        channels=5,
        top_k=2,
        hidden_channels=8,
        groups=2,
        conv_rounds=4,
        dropout=0.0,
    )
    assert len(fem.rounds) == 4
    x = torch.randn(2, 32, 5)
    y = fem(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_article_expert_depths_are_2_4_6():
    extractor = MDENFeatureExtractor(
        input_dim=5,
        top_k=1,
        hidden_channels=8,
        groups=2,
    )
    assert len(extractor.soc_net.layers) == 2
    assert len(extractor.shared_net.layers) == 4
    assert len(extractor.soh_net.layers) == 6
