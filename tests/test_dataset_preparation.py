"""Independent tests for EnergyStatus dataset discovery and preparation."""

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from mden_battery.data import (
    PreparedBatchIterableDataset,
    PreparedWindowIterableDataset,
    build_result_manifest,
    extract_archive,
    prepare_log_age_dataset,
    prepare_result_dataset,
    read_prepared_frame,
)


def _write_semicolon(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep=";", index=False)


def test_result_dataset_manifest_archive_and_article_soh(tmp_path: Path):
    raw = tmp_path / "raw"
    eoc_name = "cell_eocv2_P001_1_S01_C01.csv"
    eoc = pd.DataFrame(
        {
            "timestamp_s": [0, 100],
            "num_cycles_op": [0, 10],
            "cap_aged_est_Ah": [3.0, 2.7],
            "soh_cap": [100.0, 80.0],
        }
    )
    _write_semicolon(eoc, raw / eoc_name)
    _write_semicolon(
        pd.DataFrame({"dummy": [1]}), raw / "cell_cfg_P001_1_S01_C01.csv"
    )

    archive = tmp_path / "result.tar"
    with tarfile.open(archive, "w") as tf:
        tf.add(raw, arcname="result")
    extracted = extract_archive(archive, tmp_path / "extracted")

    manifest = build_result_manifest(extracted)
    assert set(manifest["kind"]) == {"cfg", "eocv2"}

    outputs = prepare_result_dataset(
        extracted, tmp_path / "prepared", rated_capacity_ah=3.0
    )
    soh = read_prepared_frame(outputs["eoc_soh"])
    assert np.allclose(soh["soh_article"], [1.0, 0.9])
    # Dataset SOH is preserved but not substituted for article Eq. (3).
    assert np.allclose(soh["soh_dataset"], [100.0, 80.0])


def _make_log_age(path: Path, offset: float) -> None:
    n = 90
    time = np.arange(n, dtype=float) * 30.0
    capacity = np.full(n, np.nan)
    capacity[0] = 3.0 - offset
    capacity[-1] = 2.8 - offset
    frame = pd.DataFrame(
        {
            "timestamp_s": time,
            "v_raw_V": 3.2 + 0.005 * np.arange(n),
            "i_raw_A": np.sin(np.arange(n) / 8),
            "t_cell_degC": 25 + 0.01 * np.arange(n),
            "soc_est": np.linspace(10.0, 90.0, n),
            "EFC": np.arange(n) / 50,
            "cap_aged_est_Ah": capacity,
        }
    )
    _write_semicolon(frame, path)


def test_log_age_preparation_is_chunked_split_by_cell_and_windowable(
    tmp_path: Path,
):
    raw = tmp_path / "logs"
    for idx in range(1, 4):
        _make_log_age(
            raw / f"cell_log_age_30s_P00{idx}_1_S0{idx}_C01.csv",
            offset=0.05 * (idx - 1),
        )

    outputs = prepare_log_age_dataset(
        raw,
        tmp_path / "prepared_logs",
        rated_capacity_ah=3.0,
        chunksize=25,
        split_seed=7,
        test_fraction=0.2,
        val_fraction=0.2,
    )
    index = pd.read_csv(outputs["index"])
    splits = pd.read_csv(outputs["splits"])
    assert set(splits["split"]) == {"train", "val", "test"}
    # Each cell belongs to one split, preventing cross-split leakage.
    assert splits.groupby("cell_id")["split"].nunique().max() == 1
    assert index["rows"].sum() > 0
    assert Path(outputs["scaler"]).exists()

    # Normalization statistics must be fitted on TRAIN cells only.
    train_parts = index[index["split"] == "train"]
    train_frame = pd.concat(
        [read_prepared_frame(path) for path in train_parts["path"]],
        ignore_index=True,
    )
    scaler = pd.read_csv(outputs["scaler"]).set_index("feature")
    for feature in [
        "cycle",
        "time_s",
        "voltage_V",
        "current_A",
        "temperature_C",
    ]:
        assert np.isclose(
            scaler.loc[feature, "mean"],
            train_frame[feature].mean(),
            rtol=1e-6,
            atol=1e-6,
        )

    train_ds = PreparedWindowIterableDataset(
        outputs["index"],
        split="train",
        scaler_csv=outputs["scaler"],
    )
    sample = next(iter(train_ds))
    assert sample["x"].shape == (32, 5)
    assert sample["y_soc"].shape == (8,)
    assert sample["y_soh"].shape == (8,)
    assert np.isfinite(sample["x"].numpy()).all()

    batch_ds = PreparedBatchIterableDataset(
        outputs["index"],
        batch_size=16,
        split="train",
        scaler_csv=outputs["scaler"],
    )
    batch = next(iter(batch_ds))
    assert batch["x"].ndim == 3
    assert batch["x"].shape[1:] == (32, 5)
    assert batch["y_soc"].shape[1:] == (8,)
    assert batch["y_soh"].shape[1:] == (8,)
    assert np.isfinite(batch["x"].numpy()).all()
