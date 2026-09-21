"""Execution-only optimizations and on-device, full-training-step selection."""
from __future__ import annotations

from functools import lru_cache

import torch

from .feature_extraction import FeatureExtractionModule
from .mamba import selective_scan_native


@lru_cache(maxsize=1)
def _compiled_scan():
    # Compile only the tensor recurrence. LSTM, FFT and checkpoint names stay
    # outside the compiled wrapper; default mode does not request CUDA graphs.
    return torch.compile(selective_scan_native, fullgraph=True, dynamic=False)


def configure_execution(model, fem_execution="dynamic", compile_scan=False):
    """Keep state_dict unchanged; choose how to compute existing parameters."""
    if fem_execution not in {"dynamic", "static", "matrix"}:
        raise ValueError("Unknown FEM execution")
    for module in model.modules():
        if isinstance(module, FeatureExtractionModule):
            if fem_execution == "matrix" and not all(r.fuse_linear for r in module.rounds):
                raise ValueError("matrix requires identity FEM without dropout")
            module.execution = fem_execution
    block = model.soh_mamba.impl
    enabled = bool(compile_scan and block.backend == "native"
                   and next(model.parameters()).device.type == "cuda")
    block.scan_function = _compiled_scan() if enabled else selective_scan_native
    if next(model.parameters()).device.type == "cuda":
        # Dynamic routing has many varying sub-batch sizes: autotuning each
        # would be expensive. Static/matrix geometry repeats between steps.
        torch.backends.cudnn.benchmark = fem_execution != "dynamic"
    return {"fem_execution": fem_execution, "compile_scan": enabled}


@torch.no_grad()
def verify_execution(model, batch, precision, fem_execution, compile_scan=False):
    """Compare a few real windows to the existing path before timing it."""
    from .runtime import move_batch_to_device

    x = move_batch_to_device({"x": batch["x"][:4]}, precision.device)["x"]
    prior = model.training
    model.eval()
    try:
        configure_execution(model, "dynamic", False)
        with precision.context():
            reference = model(x)
        configure_execution(model, fem_execution, compile_scan)
        with precision.context():
            candidate = model(x)
        # AMP rearrangements are not bitwise equal. FP64 gradient checks are
        # in the tests; here reject material differences on actual input data.
        tolerance = 2e-3 if precision.enabled else 5e-5
        for key in ("soc", "soh", "soc_fused", "soh_fused"):
            torch.testing.assert_close(candidate[key], reference[key],
                                       atol=tolerance, rtol=0 if key in {"soc", "soh"}
                                       else (2e-2 if precision.enabled else 5e-4))
        return {task + "_max_diff_pp": float(
            (candidate[task] - reference[task]).abs().max().cpu()) * 100
                for task in ("soc", "soh")}
    finally:
        model.train(prior)
        configure_execution(model, fem_execution, compile_scan)


def tune_training_setup(model_config, dataset, precision, *, fixed=None,
                        candidates=(256, 512, 1024, 2048, 4096, 8192),
                        fem_execution="auto", compile_scan=True,
                        learning_rate=1e-3, weight_decay=1e-4,
                        max_vram_fraction=0.88, warmup_steps=3, timed_steps=10, loss_config=None):
    """Choose execution, micro-batch, then optionally compiled native scan.

    Only training windows enter this decision. Timing includes transfers,
    loss, backward, AdamW and metrics. Probe parameters never become the
    actual training parameters. All records are returned for inspection.
    """
    from .runtime import choose_micro_batch_size

    common = dict(learning_rate=learning_rate, weight_decay=weight_decay,
                  max_vram_fraction=max_vram_fraction,
                  warmup_steps=warmup_steps, timed_steps=timed_steps, loss_config=loss_config)
    if precision.device.type != "cuda":
        size, report = choose_micro_batch_size(model_config, dataset, precision,
                                               fixed=fixed)
        mode = 'dynamic' if fem_execution == 'auto' else fem_execution
        return size, dict(report, fem_execution=mode, compile_scan=False)
    if fem_execution == "matrix" and precision.enabled:
        raise ValueError('matrix requires AMP off (AMP_DTYPE="float32")')
    if fem_execution == "auto":
        modes = ["dynamic", "static"]
        if (not precision.enabled and model_config.fem_activation == "identity"
                and model_config.fem_dropout == 0
                and dataset.config.input_length * model_config.input_dim <= 512):
            modes.append("matrix")
    else:
        modes = [fem_execution]
    records = []
    for mode in modes:
        try:
            _, result = choose_micro_batch_size(
                model_config, dataset, precision, fixed=fixed,
                candidates=candidates, fem_execution=mode, **common)
            records.append(dict(result, fem_execution=mode, compile_scan=False))
        except (AssertionError, FloatingPointError) as error:
            records.append(dict(fem_execution=mode, rejected=str(error)))
        except RuntimeError as error:
            if "No probe fits" not in str(error):
                raise
            records.append(dict(fem_execution=mode, rejected=str(error)))
    valid = [r for r in records if "rejected" not in r]
    if not valid:
        raise RuntimeError(f"No valid GPU execution: {records}")
    chosen = max(valid, key=lambda r: r["samples_per_second"])
    mode = chosen["fem_execution"]
    size, batches = chosen['batch_size'], chosen
    report = dict(batches, fem_execution=mode, compile_scan=False,
                  execution_probes=records)
    if compile_scan and model_config.mamba_backend != "official":
        try:
            _, compiled = choose_micro_batch_size(
                model_config, dataset, precision, fixed=size,
                fem_execution=mode, compile_scan=True, **common)
            report["compiled_probe"] = compiled
            if compiled['compile_scan'] and compiled["samples_per_second"] > 1.05 * batches["samples_per_second"]:
                report["compile_scan"] = True
                report["samples_per_second"] = compiled["samples_per_second"]
        except Exception as error:
            # Do not conceal compile failure or retry a poisoned CUDA context.
            if any(s in str(error).lower() for s in ("device-side assert", "illegal memory")):
                raise
            report["compiled_probe"] = {"rejected": str(error)}
            print("Compiled scan unavailable; using measured eager path:", error)
    return size, report
