import torch

from mden_battery.loss import MDENJointLoss
from mden_battery.mamba import MambaSequence
from mden_battery.model import MDEN, MDENConfig
from mden_battery.training import overfit_one_batch, seed_everything


def small_config() -> MDENConfig:
    return MDENConfig(
        input_dim=5,
        horizon=8,
        top_k=1,
        fem_hidden_channels=8,
        fem_groups=2,
        fem_dropout=0.0,
        sequence_dim=8,
        fusion_hidden=8,
        mamba_d_state=4,
        mamba_d_conv=3,
        mamba_expand=1,
        mamba_backend="native",
        dropout=0.0,
    )


def test_mden_outputs_synchronous_8_step_soc_soh():
    model = MDEN(small_config())
    x = torch.randn(2, 32, 5)
    out = model(x)
    assert out["soc"].shape == (2, 8)
    assert out["soh"].shape == (2, 8)
    assert out["soc_fused"].shape == (2, 32, 5)
    assert out["soh_fused"].shape == (2, 32, 5)
    assert torch.allclose(out["soc_weights"].sum(-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(out["soh_weights"].sum(-1), torch.ones(2), atol=1e-6)
    assert isinstance(model.soh_mamba, MambaSequence)
    assert isinstance(model.soc_lstm, torch.nn.LSTM)


def test_joint_loss_matches_article_structure_and_backpropagates():
    model = MDEN(small_config())
    criterion = MDENJointLoss(feature_dim=5)
    x = torch.randn(2, 32, 5)
    out = model(x)
    y_soc = torch.rand(2, 8)
    y_soh = 0.8 + 0.2 * torch.rand(2, 8)
    losses = criterion(out, y_soc, y_soh)
    assert losses["loss"].ndim == 0
    assert losses["sigma_soc"] > 0
    assert losses["sigma_soh"] > 0
    losses["loss"].backward()
    assert criterion.soc_uncertainty.linear.weight.grad is not None
    assert criterion.soh_uncertainty.linear.weight.grad is not None


def test_model_overfits_one_fixed_batch_without_regularization():
    seed_everything(7)
    config = MDENConfig(
        input_dim=5,
        horizon=8,
        top_k=1,
        fem_hidden_channels=8,
        fem_groups=2,
        fem_dropout=0.0,
        soc_depth=1,
        shared_depth=1,
        soh_depth=1,
        conv_rounds=1,
        fusion_hidden=8,
        sequence_dim=16,
        mamba_d_state=4,
        mamba_d_conv=3,
        mamba_expand=1,
        mamba_backend="native",
        prediction_hidden=32,
        dropout=0.2,
    )
    x = torch.randn(4, 32, 5)
    batch = {
        "x": x,
        "y_soc": torch.sigmoid(x[:, -8:, 0]),
        "y_soh": torch.sigmoid(x[:, -8:, 1]),
    }

    result = overfit_one_batch(
        MDEN(config),
        batch,
        torch.device("cpu"),
        steps=250,
        learning_rate=3e-3,
        target_rmse=0.02,
    )

    assert result.converged
    assert result.soc_rmse[-1] <= 0.02
    assert result.soh_rmse[-1] <= 0.02
