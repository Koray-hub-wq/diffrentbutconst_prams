"""Resample windows from constant-parameter trajectory data.

This script is the constant-parameter counterpart to
time_varying_bifurcation_data/resample_time_varying_bifurcation_windows.py. It
does not use phi-crossing logic, because phi is constant over each trajectory.
Training windows are sampled randomly per trajectory, and one test trajectory is
taken from the end of every full trajectory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

Array = np.ndarray


def path_for_io(path: Path) -> str:
    resolved = str(path.resolve())
    if not resolved.startswith("\\\\?\\") and len(resolved) >= 240:
        if resolved.startswith("\\\\"):
            return "\\\\?\\UNC\\" + resolved[2:]
        return "\\\\?\\" + resolved
    return resolved


def save_npy(path: Path, value: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_for_io(path), value)


def open_for_write_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path_for_io(path), "w", encoding="utf-8")


def optional_load(path: Path, default: Array) -> Array:
    if path.exists():
        return np.load(path, allow_pickle=True)
    return default


def load_constant_values(path: Path, fallback: Array) -> Array:
    if path.exists():
        values = np.load(path)
        return np.asarray(values).reshape(-1)
    return np.asarray(fallback[:, 0, 0]).reshape(-1)


def values_match_any(values: Array, requested: list[float], atol: float) -> Array:
    requested_arr = np.asarray(requested, dtype=np.float64)
    values_arr = np.asarray(values, dtype=np.float64)
    return np.any(np.isclose(values_arr[:, None], requested_arr[None, :], atol=atol, rtol=0.0), axis=1)


def filter_series_by_parameter_values(
    input_dir: Path,
    trajectories: Array,
    phi: Array,
    phi_raw: Array,
    regime_names: Array,
    include_phi_values: list[float] | None,
    include_raw_parameter_values: list[float] | None,
    parameter_atol: float,
) -> tuple[Array, Array, Array, Array, Array, dict[str, list[float] | int | None]]:
    if include_phi_values and include_raw_parameter_values:
        raise ValueError("Use only one of --include-phi-values or --include-raw-parameter-values")

    n_series = trajectories.shape[0]
    keep = np.ones(n_series, dtype=bool)
    phi_constants = load_constant_values(input_dir / "trajectory_phi_constants.npy", phi)
    raw_constants = load_constant_values(input_dir / "trajectory_phi_raw_constants.npy", phi_raw)

    if include_phi_values:
        keep = values_match_any(phi_constants, include_phi_values, parameter_atol)
    elif include_raw_parameter_values:
        keep = values_match_any(raw_constants, include_raw_parameter_values, parameter_atol)

    if not np.any(keep):
        available_phi = ", ".join(f"{value:.8g}" for value in sorted(np.unique(phi_constants)))
        available_raw = ", ".join(f"{value:.8g}" for value in sorted(np.unique(raw_constants)))
        raise ValueError(
            "Parameter filter removed all trajectories. "
            f"Available phi values: {available_phi}. "
            f"Available raw parameter values: {available_raw}."
        )

    selected_indices = np.flatnonzero(keep)
    info: dict[str, list[float] | int | None] = {
        "include_phi_values": include_phi_values,
        "include_raw_parameter_values": include_raw_parameter_values,
        "parameter_atol": parameter_atol,
        "n_series_before_filter": int(n_series),
        "n_series_after_filter": int(selected_indices.size),
        "selected_series_indices": [int(i) for i in selected_indices],
        "selected_phi_values": [float(v) for v in sorted(np.unique(phi_constants[keep]))],
        "selected_raw_parameter_values": [float(v) for v in sorted(np.unique(raw_constants[keep]))],
    }

    return (
        trajectories[keep],
        phi[keep],
        phi_raw[keep],
        regime_names[keep],
        selected_indices,
        info,
    )


def save_or_copy_filtered_array(
    input_dir: Path,
    output_dir: Path,
    name: str,
    selected_indices: Array,
) -> None:
    src = input_dir / name
    if not src.exists():
        return
    value = np.load(src, allow_pickle=True)
    if np.array_equal(selected_indices, np.arange(selected_indices.shape[0])):
        shutil.copy2(src, output_dir / name)
        return
    if value.shape[0] <= int(selected_indices.max(initial=-1)):
        raise ValueError(
            f"Cannot filter {name}: first dimension {value.shape[0]} does not match "
            f"source trajectory count implied by selected indices."
        )
    save_npy(output_dir / name, value[selected_indices])


def build_training_windows(
    trajectories: Array,
    phi: Array,
    phi_raw: Array,
    regime_names: Array,
    windows_per_trajectory: int,
    context_len: int,
    seq_len: int,
    overlap_len: int,
    test_len: int,
    rng: np.random.Generator,
    allow_train_overlap_test_tail: bool,
) -> dict[str, Array]:
    if seq_len <= context_len:
        raise ValueError("seq_len must be greater than context_len")
    if overlap_len < 0:
        raise ValueError("overlap_len must be non-negative")
    if overlap_len > context_len:
        raise ValueError("overlap_len must be <= context_len")
    if windows_per_trajectory <= 0:
        raise ValueError("windows_per_trajectory must be positive")

    prediction_window_len = seq_len - context_len
    data_len = overlap_len + prediction_window_len
    if data_len <= 0:
        raise ValueError("data_len must be positive")

    n_series, n_steps, n_dims = trajectories.shape
    if n_steps < seq_len:
        raise ValueError(f"Need n_steps >= seq_len, got {n_steps} < {seq_len}")
    if n_steps < test_len:
        raise ValueError(f"Need n_steps >= test_len, got {n_steps} < {test_len}")

    max_start = n_steps - seq_len
    if not allow_train_overlap_test_tail:
        max_start = n_steps - test_len - seq_len
        if max_start < 0:
            raise ValueError(
                "No room for non-overlapping train windows before the test tail. "
                "Use shorter lengths or pass --allow-train-overlap-test-tail."
            )

    total_windows = n_series * windows_per_trajectory
    data = np.empty((data_len, total_windows, n_dims), dtype=np.float32)
    context = np.empty((context_len, total_windows, n_dims), dtype=np.float32)
    phi_data = np.empty((data_len, total_windows, 1), dtype=np.float32)
    phi_context = np.empty((context_len, total_windows, 1), dtype=np.float32)
    phi_raw_data = np.empty((data_len, total_windows, 1), dtype=np.float32)
    phi_raw_context = np.empty((context_len, total_windows, 1), dtype=np.float32)
    start_indices = np.empty(total_windows, dtype=np.int64)
    window_regime: list[str] = []

    out_idx = 0
    for series_idx in range(n_series):
        for _ in range(windows_per_trajectory):
            start = int(rng.integers(0, max_start + 1))
            end = start + seq_len
            data_start = start + context_len - overlap_len

            context[:, out_idx, :] = trajectories[series_idx, start : start + context_len, :]
            data[:, out_idx, :] = trajectories[series_idx, data_start:end, :]
            phi_context[:, out_idx, :] = phi[series_idx, start : start + context_len, :]
            phi_data[:, out_idx, :] = phi[series_idx, data_start:end, :]
            phi_raw_context[:, out_idx, :] = phi_raw[series_idx, start : start + context_len, :]
            phi_raw_data[:, out_idx, :] = phi_raw[series_idx, data_start:end, :]
            start_indices[out_idx] = start
            window_regime.append(str(regime_names[series_idx]))
            out_idx += 1

    return {
        "data": data,
        "context": context,
        "phi": phi_data,
        "context_phi": phi_context,
        "phi_raw": phi_raw_data,
        "context_phi_raw": phi_raw_context,
        "start_indices": start_indices,
        "window_regime_names": np.array(window_regime),
        "window_sampling_modes": np.array(["random"] * total_windows),
    }


def build_test_windows(
    trajectories: Array,
    phi: Array,
    phi_raw: Array,
    test_len: int,
) -> dict[str, Array]:
    if trajectories.shape[1] < test_len:
        raise ValueError(f"Need n_steps >= test_len, got {trajectories.shape[1]} < {test_len}")
    return {
        "test": trajectories[:, -test_len:, :].transpose(1, 0, 2).astype(np.float32),
        "test_phi": phi[:, -test_len:, :].transpose(1, 0, 2).astype(np.float32),
        "test_phi_raw": phi_raw[:, -test_len:, :].transpose(1, 0, 2).astype(np.float32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--windows-per-trajectory", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=550)
    parser.add_argument(
        "--overlap-len",
        type=int,
        default=None,
        help=(
            "Number of final context steps also included at the beginning of data. "
            "Defaults to seq_len - context_len to preserve the previous behavior."
        ),
    )
    parser.add_argument(
        "--test-len",
        type=int,
        default=10000,
        help="Length of the test trajectory taken from the end of every full trajectory.",
    )
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--allow-train-overlap-test-tail",
        action="store_true",
        help="Allow random training windows to overlap the final test segment.",
    )
    parser.add_argument(
        "--copy-full-arrays",
        action="store_true",
        help="Also copy full trajectory arrays into the output directory.",
    )
    parser.add_argument(
        "--include-phi-values",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Only include trajectories whose constant normalized phi value matches "
            "one of these values, e.g. --include-phi-values -1 -0.5 0 0.5 1."
        ),
    )
    parser.add_argument(
        "--include-raw-parameter-values",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Only include trajectories whose raw parameter value matches one of "
            "these values, e.g. Halvorsen a values."
        ),
    )
    parser.add_argument(
        "--parameter-atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for matching included phi/raw parameter values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlap_len = (
        args.seq_len - args.context_len
        if args.overlap_len is None
        else args.overlap_len
    )
    if args.test_len <= 0:
        raise ValueError("test_len must be positive")

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = np.load(args.input_dir / "trajectories.npy")
    phi = np.load(args.input_dir / "trajectory_phi.npy")
    phi_raw = np.load(args.input_dir / "trajectory_phi_raw.npy")
    regime_names = optional_load(
        args.input_dir / "trajectory_regime_names.npy",
        np.array([f"series_{i}" for i in range(trajectories.shape[0])]),
    )
    (
        trajectories,
        phi,
        phi_raw,
        regime_names,
        selected_indices,
        parameter_filter,
    ) = filter_series_by_parameter_values(
        args.input_dir,
        trajectories,
        phi,
        phi_raw,
        regime_names,
        args.include_phi_values,
        args.include_raw_parameter_values,
        args.parameter_atol,
    )

    windows = build_training_windows(
        trajectories,
        phi,
        phi_raw,
        regime_names,
        windows_per_trajectory=args.windows_per_trajectory,
        context_len=args.context_len,
        seq_len=args.seq_len,
        overlap_len=overlap_len,
        test_len=args.test_len,
        rng=rng,
        allow_train_overlap_test_tail=args.allow_train_overlap_test_tail,
    )
    test = build_test_windows(trajectories, phi, phi_raw, test_len=args.test_len)

    for name, value in windows.items():
        save_npy(args.output_dir / f"{name}.npy", value)
    for name, value in test.items():
        save_npy(args.output_dir / f"{name}.npy", value)

    passthrough_arrays = [
        "trajectory_regime_index.npy",
        "trajectory_regime_names.npy",
        "trajectory_phi_value_index.npy",
        "trajectory_phi_constants.npy",
        "trajectory_phi_raw_constants.npy",
    ]
    for name in passthrough_arrays:
        save_or_copy_filtered_array(args.input_dir, args.output_dir, name, selected_indices)

    if args.copy_full_arrays:
        save_npy(args.output_dir / "trajectories.npy", trajectories.astype(np.float32))
        save_npy(args.output_dir / "trajectory_phi.npy", phi.astype(np.float32))
        save_npy(args.output_dir / "trajectory_phi_raw.npy", phi_raw.astype(np.float32))

    source_metadata_path = args.input_dir / "metadata.json"
    source_metadata = {}
    if source_metadata_path.exists():
        with open(source_metadata_path, "r", encoding="utf-8") as f:
            source_metadata = json.load(f)

    metadata = {
        "description": "Random training windows resampled from constant-parameter trajectories.",
        "source_dir": str(args.input_dir),
        "source_args": source_metadata.get("args", {}),
        "resample_args": vars(args)
        | {
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "resolved_overlap_len": overlap_len,
        },
        "shapes": {
            "data": list(windows["data"].shape),
            "context": list(windows["context"].shape),
            "phi": list(windows["phi"].shape),
            "test": list(test["test"].shape),
            "test_phi": list(test["test_phi"].shape),
        },
        "sample_counts": {"random": int(windows["data"].shape[1])},
        "parameter_filter": parameter_filter,
        "window_layout_convention": (
            "context covers [start, start + context_len). data covers the last "
            "overlap_len context steps plus all steps after context until seq_len, "
            "so data_len = overlap_len + (seq_len - context_len)."
        ),
        "test_layout_convention": (
            "test, test_phi, and test_phi_raw are the final test_len steps of every "
            "trajectory, transposed to (test_len, n_trajectories, channels)."
        ),
    }
    for key in ["regimes", "phi_convention", "requested_phi_values"]:
        if key in source_metadata:
            metadata[key] = source_metadata[key]

    with open_for_write_text(args.output_dir / "metadata.json") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved resampled windows to {args.output_dir.resolve()}")
    for key, shape in metadata["shapes"].items():
        print(f"{key}: {shape}")
    print(f"sample_counts: {metadata['sample_counts']}")
    if parameter_filter["include_phi_values"] or parameter_filter["include_raw_parameter_values"]:
        print(
            "parameter_filter: "
            f"{parameter_filter['n_series_before_filter']} -> "
            f"{parameter_filter['n_series_after_filter']} trajectories"
        )


if __name__ == "__main__":
    main()
