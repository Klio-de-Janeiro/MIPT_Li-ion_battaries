from __future__ import annotations

import logging
import re
import tarfile
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Official pattern from bat-age-data-scripts/config_main.py.
CELL_RE = re.compile(
    r"cell_(?P<kind>cfg|eocv2|eoc|eisv2|eis|plsv2|pls|log_age_\d+s|logext)_"
    r"P(?P<param>\d+)_(?P<rep>\d+)_S(?P<slave>\d+)_C(?P<channel>\d+)\.csv$",
    re.IGNORECASE,
)

CSV_SEP = ";"


@dataclass(frozen=True)
class EnergyStatusPaths:
    root: Path
    working: Path


def _safe_target(base: Path, name: str) -> Path:
    target = (base / name).resolve()
    base_resolved = base.resolve()
    if target != base_resolved and base_resolved not in target.parents:
        raise ValueError(f"Unsafe archive path: {name}")
    return target


def _archive_stem(path: Path) -> str:
    name = path.name
    for suffix in (
        ".tar.gz",
        ".tar.xz",
        ".tar.bz2",
        ".tgz",
        ".zip",
        ".7z",
        ".tar",
    ):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def extract_archive(path: Path, output_dir: Path) -> Path:
    """Safely extract an archive and return its output directory."""
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / _archive_stem(path)
    dest.mkdir(parents=True, exist_ok=True)

    if tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tf:
            for member in tf.getmembers():
                _safe_target(dest, member.name)
            try:
                tf.extractall(dest, filter="data")
            except TypeError:  # Python versions without extraction filters
                tf.extractall(dest)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                _safe_target(dest, name)
            zf.extractall(dest)
    elif path.suffix.lower() == ".7z":
        try:
            import py7zr  # type: ignore
        except ImportError as exc:
            raise ImportError("Install py7zr to extract .7z archives") from exc
        with py7zr.SevenZipFile(path, mode="r") as zf:
            # py7zr does not expose tar-like members before extraction in a
            # completely uniform way; archive is assumed trusted Kaggle input.
            zf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive: {path}")
    return dest


def recursively_extract(
    root: Path, work_dir: Path, max_depth: int = 3
) -> Path:
    """Extract nested archives without a fixed Kaggle layout."""
    root = Path(root)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    search_roots = [root]
    seen: set[Path] = set()
    for _ in range(max_depth):
        discovered: list[Path] = []
        for search_root in search_roots:
            for path in search_root.rglob("*"):
                if not path.is_file() or path in seen:
                    continue
                is_archive = path.suffix.lower() in {
                    ".zip",
                    ".7z",
                    ".tar",
                    ".tgz",
                }
                if not is_archive:
                    try:
                        is_archive = tarfile.is_tarfile(path)
                    except OSError:
                        is_archive = False
                if not is_archive:
                    continue
                try:
                    dest = extract_archive(path, work_dir)
                except (
                    ValueError,
                    tarfile.TarError,
                    zipfile.BadZipFile,
                    OSError,
                ):
                    logger.debug(
                        "Skipping non-extractable file %s", path, exc_info=True
                    )
                    continue
                seen.add(path)
                discovered.append(dest)
        if not discovered:
            break
        search_roots = discovered
    return work_dir


def parse_cell_filename(path: Path) -> dict[str, object] | None:
    match = CELL_RE.match(path.name)
    if match is None:
        return None
    d = match.groupdict()
    parameter = int(d["param"])
    replicate = int(d["rep"])
    return {
        "kind": d["kind"].lower(),
        "parameter_id": parameter,
        "replicate": replicate,
        "slave": int(d["slave"]),
        "channel": int(d["channel"]),
        "cell_id": f"P{parameter:03d}_{replicate}",
        "path": str(path),
    }


def build_result_manifest(root: Path) -> pd.DataFrame:
    """Discover official EnergyStatus cell CSV files recursively."""
    rows: list[dict[str, object]] = []
    for path in Path(root).rglob("*.csv"):
        parsed = parse_cell_filename(path)
        if parsed is not None:
            rows.append(parsed)
    columns = [
        "kind",
        "parameter_id",
        "replicate",
        "slave",
        "channel",
        "cell_id",
        "path",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["cell_id", "kind"])
        .reset_index(drop=True)
    )


def _read_csv(
    path: str | Path, usecols: Sequence[str] | None = None
) -> pd.DataFrame:
    """Read an official semicolon-separated file."""
    try:
        return pd.read_csv(
            path, sep=CSV_SEP, usecols=usecols, engine="pyarrow"
        )
    except (ImportError, ValueError):
        return pd.read_csv(path, sep=CSV_SEP, usecols=usecols)


def _iter_csv_chunks(
    path: str | Path,
    *,
    usecols: Sequence[str] | None = None,
    chunksize: int = 500_000,
) -> Iterator[pd.DataFrame]:
    # pandas' pyarrow CSV engine does not support chunksize. The C parser is
    # deliberately used for the large LOG_AGE files to keep RAM bounded.
    yield from pd.read_csv(
        path,
        sep=CSV_SEP,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
    )


def _pick_column(
    columns: Iterable[str], candidates: Sequence[str]
) -> str | None:
    columns_set = set(columns)
    return next((name for name in candidates if name in columns_set), None)


def _write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> Path:
    """Write Parquet or fall back to compressed CSV."""
    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError):
        csv_path = path_without_suffix.with_suffix(".csv.gz")
        frame.to_csv(csv_path, index=False, compression="gzip")
        return csv_path


def read_prepared_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported prepared frame: {path}")


def prepare_result_dataset(
    root: Path,
    output_dir: Path,
    *,
    rated_capacity_ah: float = 3.0,
) -> dict[str, Path]:
    """Prepare the ~333.5 MB result-only EnergyStatus dataset.

    The result dataset contains capacity, impedance and pulse-check-up results,
    not the continuous V/I/T log windows required by MDEN. It is therefore
    prepared for schema verification, SOH analysis and baselines only.

    ``soh_article`` is recomputed as capacity/rated_capacity (paper Eq. 3).
    EnergyStatus's published ``soh_cap`` has a different scaling convention and
    is preserved separately as ``soh_dataset`` when present.
    """
    if rated_capacity_ah <= 0:
        raise ValueError("rated_capacity_ah must be positive")

    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_result_manifest(root)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    eoc_rows: list[pd.DataFrame] = []
    eoc_manifest = (
        manifest[manifest["kind"].isin(["eoc", "eocv2"])]
        if len(manifest)
        else manifest
    )
    for _, row in eoc_manifest.iterrows():
        df = _read_csv(row["path"])
        cap_col = _pick_column(
            df.columns, ["cap_aged_est_Ah", "capacity_Ah", "capacity"]
        )
        time_col = _pick_column(
            df.columns, ["timestamp_s", "timestamp", "time_s"]
        )
        cycle_col = _pick_column(
            df.columns, ["num_cycles_op", "cycle", "EFC", "efc"]
        )
        if cap_col is None:
            logger.warning(
                "Skipping EOC file without capacity column: %s", row["path"]
            )
            continue

        out = pd.DataFrame(
            {
                "cell_id": row["cell_id"],
                "capacity_ah": pd.to_numeric(df[cap_col], errors="coerce"),
            }
        )
        out["timestamp_s"] = (
            pd.to_numeric(df[time_col], errors="coerce")
            if time_col
            else np.nan
        )
        out["cycle"] = (
            pd.to_numeric(df[cycle_col], errors="coerce")
            if cycle_col
            else np.nan
        )
        out["soh_article"] = out["capacity_ah"] / float(rated_capacity_ah)
        if "soh_cap" in df.columns:
            out["soh_dataset"] = pd.to_numeric(df["soh_cap"], errors="coerce")
        eoc_rows.append(out)

    if eoc_rows:
        eoc = pd.concat(eoc_rows, ignore_index=True)
    else:
        eoc = pd.DataFrame(
            columns=[
                "cell_id",
                "capacity_ah",
                "timestamp_s",
                "cycle",
                "soh_article",
            ]
        )
    eoc_path = _write_frame(eoc, output_dir / "eoc_soh")

    if len(manifest):
        summary = (
            manifest.groupby("kind").size().rename("file_count").reset_index()
        )
    else:
        summary = pd.DataFrame(columns=["kind", "file_count"])
    summary_path = output_dir / "file_summary.csv"
    summary.to_csv(summary_path, index=False)

    return {
        "manifest": manifest_path,
        "eoc_soh": eoc_path,
        "summary": summary_path,
    }


def _capacity_anchors(
    path: Path, chunksize: int
) -> tuple[np.ndarray, np.ndarray]:
    """Collect sparse capacity anchors with bounded memory."""
    times: list[np.ndarray] = []
    caps: list[np.ndarray] = []
    usecols = ["timestamp_s", "cap_aged_est_Ah"]
    try:
        iterator = _iter_csv_chunks(path, usecols=usecols, chunksize=chunksize)
        for chunk in iterator:
            t = pd.to_numeric(chunk["timestamp_s"], errors="coerce").to_numpy(
                float
            )
            c = pd.to_numeric(
                chunk["cap_aged_est_Ah"], errors="coerce"
            ).to_numpy(float)
            mask = np.isfinite(t) & np.isfinite(c)
            if mask.any():
                times.append(t[mask])
                caps.append(c[mask])
    except ValueError:
        # Some generated LOG_AGE variants intentionally omit capacity.
        return np.array([], dtype=float), np.array([], dtype=float)

    if not times:
        return np.array([], dtype=float), np.array([], dtype=float)
    t = np.concatenate(times)
    c = np.concatenate(caps)
    order = np.argsort(t)
    t, c = t[order], c[order]
    # np.interp expects increasing x; collapse duplicate timestamps.
    unique_t, unique_idx = np.unique(t, return_index=True)
    return unique_t, c[unique_idx]


def _interp_capacity(
    timestamp_s: np.ndarray,
    anchor_t: np.ndarray,
    anchor_capacity: np.ndarray,
) -> np.ndarray:
    if anchor_t.size == 0:
        return np.full(timestamp_s.shape, np.nan, dtype=float)
    return np.interp(
        timestamp_s,
        anchor_t,
        anchor_capacity,
        left=anchor_capacity[0],
        right=anchor_capacity[-1],
    )


def _split_cells(
    cells: Sequence[str],
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, str]:
    if (
        val_fraction < 0
        or test_fraction < 0
        or val_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "val_fraction and test_fraction must be >=0 and sum to <1"
        )
    cells_arr = np.asarray(sorted(set(cells)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(cells_arr)
    if len(cells_arr) == 0:
        return {}

    n_test = round(len(cells_arr) * test_fraction)
    n_val = round(len(cells_arr) * val_fraction)
    if test_fraction > 0 and len(cells_arr) >= 3:
        n_test = max(1, n_test)
    if val_fraction > 0 and len(cells_arr) - n_test >= 2:
        n_val = max(1, n_val)
    if n_test + n_val >= len(cells_arr):
        overflow = n_test + n_val - (len(cells_arr) - 1)
        n_val = max(0, n_val - overflow)

    result = {str(cell): "train" for cell in cells_arr}
    for cell in cells_arr[:n_test]:
        result[str(cell)] = "test"
    for cell in cells_arr[n_test : n_test + n_val]:
        result[str(cell)] = "val"
    return result


def prepare_log_age_dataset(
    root: Path,
    output_dir: Path,
    *,
    rated_capacity_ah: float = 3.0,
    include_profile_2s: bool = True,
    split_seed: int = 42,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    chunksize: int = 500_000,
    soc_scale: float = 100.0,
) -> dict[str, Path]:
    """Prepare the large EnergyStatus LOG_AGE data in bounded memory.

    This is an *adapter* to the article, not the MIT preprocessing used by the
    authors. The five model inputs are mapped to [EFC/cycle, time, voltage,
    current, temperature]. ``soc_est`` is used as SOC supervision because
    EnergyStatus already publishes a state estimate; SOH is recomputed as
    ``cap_aged_est_Ah / rated_capacity_ah`` to match Eq. (3).

    Processing is two-pass per file: sparse capacity anchors are collected,
    then the log is streamed in chunks. This is suitable for the ~89 GB v2 log
    dataset without reading it all into RAM.
    """
    if rated_capacity_ah <= 0:
        raise ValueError("rated_capacity_ah must be positive")
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")

    root = Path(root)
    output_dir = Path(output_dir)
    parts_dir = output_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_result_manifest(root)
    logs = (
        manifest[manifest["kind"].str.startswith("log_age_")].copy()
        if len(manifest)
        else manifest
    )
    if not include_profile_2s and len(logs):
        logs = logs[logs["kind"] == "log_age_30s"]
    if logs.empty:
        raise FileNotFoundError(
            "No cell_log_age_* CSV files found under the supplied root."
        )

    split_map = _split_cells(
        logs["cell_id"].astype(str).tolist(),
        seed=split_seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    index_rows: list[dict[str, object]] = []
    feature_columns = [
        "cycle",
        "time_s",
        "voltage_V",
        "current_A",
        "temperature_C",
    ]
    running_sum = np.zeros(len(feature_columns), dtype=np.float64)
    running_sumsq = np.zeros(len(feature_columns), dtype=np.float64)
    running_count = 0
    required = ["timestamp_s", "v_raw_V", "i_raw_A", "t_cell_degC", "soc_est"]

    for _, row in logs.iterrows():
        path = Path(row["path"])
        header = pd.read_csv(path, sep=CSV_SEP, nrows=0)
        cycle_col = _pick_column(
            header.columns, ["num_cycles_op", "cycle", "EFC", "efc"]
        )
        if cycle_col is None:
            logger.warning("Skipping %s: no EFC/cycle column", path.name)
            continue
        missing = [col for col in required if col not in header.columns]
        if missing:
            logger.warning(
                "Skipping %s: missing columns %s", path.name, missing
            )
            continue

        anchor_t, anchor_capacity = _capacity_anchors(
            path, chunksize=chunksize
        )
        usecols = [*required, cycle_col]
        # Avoid duplicate usecol when candidate happens to overlap.
        usecols = list(dict.fromkeys(usecols))

        split = split_map[str(row["cell_id"])]
        for chunk_idx, df in enumerate(
            _iter_csv_chunks(path, usecols=usecols, chunksize=chunksize)
        ):
            timestamp = pd.to_numeric(
                df["timestamp_s"], errors="coerce"
            ).to_numpy(float)
            capacity = _interp_capacity(timestamp, anchor_t, anchor_capacity)
            out = pd.DataFrame(
                {
                    "cell_id": str(row["cell_id"]),
                    "cycle": pd.to_numeric(df[cycle_col], errors="coerce"),
                    "time_s": timestamp,
                    "voltage_V": pd.to_numeric(df["v_raw_V"], errors="coerce"),
                    "current_A": pd.to_numeric(df["i_raw_A"], errors="coerce"),
                    "temperature_C": pd.to_numeric(
                        df["t_cell_degC"], errors="coerce"
                    ),
                    "soc": pd.to_numeric(df["soc_est"], errors="coerce"),
                    "soh": capacity / float(rated_capacity_ah),
                }
            )
            # Explicit units for the entire run; never infer them per chunk.
            if soc_scale <= 0:
                raise ValueError("soc_scale must be positive")
            out["soc"] = out["soc"] / soc_scale
            out = (
                out.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .reset_index(drop=True)
            )
            if out.empty:
                continue

            # Fit normalization statistics on TRAIN cells only to avoid
            # validation/test leakage. The split is assigned at cell level.
            if split == "train":
                values = out[feature_columns].to_numpy(dtype=np.float64)
                running_sum += values.sum(axis=0)
                running_sumsq += np.square(values).sum(axis=0)
                running_count += len(values)

            stem = f"{split}_{row['cell_id']}_{path.stem}_part{chunk_idx:04d}"
            part_path = _write_frame(out, parts_dir / stem)
            index_rows.append(
                {
                    "cell_id": str(row["cell_id"]),
                    "split": split,
                    "source": str(path),
                    "path": str(part_path),
                    "rows": len(out),
                    "chunk": chunk_idx,
                }
            )

    index = pd.DataFrame(index_rows)
    index_path = output_dir / "prepared_index.csv"
    index.to_csv(index_path, index=False)
    split_path = output_dir / "cell_splits.csv"
    pd.DataFrame(
        [
            {"cell_id": cell, "split": split}
            for cell, split in sorted(split_map.items())
        ]
    ).to_csv(split_path, index=False)

    scaler_path = output_dir / "scaler.csv"
    if running_count > 0:
        mean = running_sum / running_count
        var = np.maximum(running_sumsq / running_count - np.square(mean), 0.0)
        std = np.sqrt(var)
        std[std == 0] = 1.0
        pd.DataFrame(
            {"feature": feature_columns, "mean": mean, "std": std}
        ).to_csv(scaler_path, index=False)
    else:
        pd.DataFrame(columns=["feature", "mean", "std"]).to_csv(
            scaler_path, index=False
        )

    return {
        "index": index_path,
        "parts": parts_dir,
        "splits": split_path,
        "scaler": scaler_path,
    }
