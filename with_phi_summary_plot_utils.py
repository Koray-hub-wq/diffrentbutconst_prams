from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


TRUTH_COLOR = "#8f9aa5"
PRED_COLOR = "#ff3333"


def load_metadata(data_dir: Path) -> dict[str, Any]:
    path = data_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def context_steps_from_data_dir(data_dir: Path, override: int | None) -> int:
    if override is not None:
        return int(override)
    context = np.load(data_dir / "context.npy")
    return int(context.shape[0])


def raw_from_phi(phi: np.ndarray | float, metadata: dict[str, Any]) -> np.ndarray:
    regimes = metadata.get("regimes") or metadata.get("data_metadata", {}).get("regimes")
    if not regimes:
        return np.asarray(phi, dtype=float)
    regime = regimes[0]
    boundary = float(regime["boundary"])
    start = float(regime["start"])
    end = float(regime["end"])
    scale = max(abs(start - boundary), abs(end - boundary), 1e-8)
    return boundary + np.asarray(phi, dtype=float) * scale


def safe_phi_label(phi: float) -> str:
    return f"{phi:+.6g}".replace("+", "p").replace("-", "m").replace(".", "p")


def group_series_by_phi(test_phi: np.ndarray, decimals: int) -> dict[float, np.ndarray]:
    phi_values = np.round(test_phi[0, :, 0].astype(float), decimals)
    groups: dict[float, np.ndarray] = {}
    for phi in np.unique(phi_values):
        groups[float(phi)] = np.flatnonzero(np.isclose(phi_values, phi))
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def resolve_requested_phi_values(
    available_phi: np.ndarray,
    available_raw: np.ndarray,
    requested: list[float] | None,
    parameter_space: str,
    atol: float,
) -> list[int]:
    if requested is None:
        return list(range(len(available_phi)))
    source = available_phi if parameter_space == "phi" else available_raw
    indices: list[int] = []
    for value in requested:
        distances = np.abs(source - float(value))
        idx = int(np.argmin(distances))
        if distances[idx] > atol:
            available = ", ".join(f"{item:.6g}" for item in source)
            raise ValueError(
                f"Requested {parameter_space}={value:g} was not found. "
                f"Closest is {source[idx]:.6g}; available values: {available}"
            )
        if idx not in indices:
            indices.append(idx)
    return indices


def median_index(values: np.ndarray) -> int:
    order = np.argsort(np.asarray(values, dtype=float))
    return int(order[len(order) // 2])


def select_group_median_prediction(
    group_indices: np.ndarray,
    scores: np.ndarray,
    preds: list[np.ndarray],
) -> dict[str, Any]:
    per_series = []
    for series_idx in group_indices:
        rollout_idx = median_index(scores[:, series_idx])
        per_series.append(
            {
                "series_idx": int(series_idx),
                "rollout_idx": int(rollout_idx),
                "score": float(scores[rollout_idx, series_idx]),
            }
        )
    ordered = sorted(per_series, key=lambda item: item["score"])
    selected = dict(ordered[len(ordered) // 2])
    selected["prediction"] = preds[selected["rollout_idx"]][:, selected["series_idx"], :]
    selected["rank"] = int(len(ordered) // 2 + 1)
    selected["rank_total"] = int(len(ordered))
    selected["per_series"] = per_series
    return selected


def make_plot_indices(n_steps: int, max_points: int | None) -> np.ndarray:
    if max_points is None or max_points <= 0 or max_points >= n_steps:
        return np.arange(n_steps)
    return np.unique(np.linspace(0, n_steps - 1, max_points).astype(int))


def row_lengths(n_plots: int, max_cols: int = 3) -> list[int]:
    if n_plots <= 0:
        return []
    if n_plots == 4:
        return [2, 2]
    rows = int(math.ceil(n_plots / max_cols))
    base = n_plots // rows
    rem = n_plots % rows
    return [base + (1 if row < rem else 0) for row in range(rows)]


def add_centered_3d_axes(
    fig: plt.Figure,
    row_lengths_: list[int],
    *,
    left: float = 0.04,
    right: float = 0.98,
    bottom: float = 0.08,
    top: float = 0.88,
    hgap: float = 0.04,
    vgap: float = 0.07,
):
    max_cols = max(row_lengths_)
    n_rows = len(row_lengths_)
    cell_w = (right - left - hgap * (max_cols - 1)) / max_cols
    cell_h = (top - bottom - vgap * (n_rows - 1)) / n_rows
    axes = []
    for row, n_cols in enumerate(row_lengths_):
        row_width = n_cols * cell_w + (n_cols - 1) * hgap
        x0 = left + ((right - left) - row_width) / 2.0
        y0 = top - (row + 1) * cell_h - row * vgap
        for col in range(n_cols):
            axes.append(fig.add_axes([x0 + col * (cell_w + hgap), y0, cell_w, cell_h], projection="3d"))
    return axes


def set_axes_equal_3d(ax, xyz: np.ndarray) -> None:
    finite = xyz[np.all(np.isfinite(xyz), axis=1)]
    if finite.size == 0:
        return
    mins = finite.min(axis=0)
    maxs = finite.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius <= 0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def format_parameter_label(phi: float, raw: float, label_space: str, raw_name: str) -> str:
    if label_space == "none":
        return ""
    if label_space == "raw":
        return f"{raw_name} = {raw:.4g}"
    if label_space == "both":
        return rf"$\varphi$ = {phi:.4g}, {raw_name} = {raw:.4g}"
    return rf"$\varphi$ = {phi:.4g}"


def plot_prediction_grid(
    output_path: Path,
    truth_future: np.ndarray,
    pred_future: np.ndarray,
    phi_values: np.ndarray,
    raw_values: np.ndarray,
    *,
    selected_indices: list[int] | None = None,
    title: str | None = None,
    label_space: str = "phi",
    raw_name: str = "a",
    max_points: int | None = 2500,
    dpi: int = 220,
    figsize: tuple[float, float] | None = None,
    view_elev: float = 22.0,
    view_azim: float = -58.0,
) -> None:
    selected_indices = selected_indices if selected_indices is not None else list(range(len(phi_values)))
    n_plots = len(selected_indices)
    if n_plots == 0:
        raise ValueError("No parameter values selected for plotting.")

    rows = row_lengths(n_plots)
    if figsize is None:
        figsize = (13.333, 7.5) if len(rows) <= 2 else (13.333, 2.9 * len(rows) + 1.1)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    axes = add_centered_3d_axes(fig, rows)

    for ax, item_idx in zip(axes, selected_indices):
        truth = truth_future[item_idx]
        pred = pred_future[item_idx]
        plot_idx = make_plot_indices(truth.shape[0], max_points)
        truth_plot = truth[plot_idx]
        pred_plot = pred[plot_idx]
        ax.plot(
            truth_plot[:, 0],
            truth_plot[:, 1],
            truth_plot[:, 2],
            color=TRUTH_COLOR,
            lw=1.1,
            label="Ground Truth",
        )
        ax.plot(
            pred_plot[:, 0],
            pred_plot[:, 1],
            pred_plot[:, 2],
            color=PRED_COLOR,
            lw=1.1,
            label="DynaMix",
        )
        label = format_parameter_label(
            float(phi_values[item_idx]), float(raw_values[item_idx]), label_space, raw_name
        )
        if label:
            ax.set_title(label, fontsize=10, pad=1)
        ax.set_xlabel("Dimension 1", fontsize=8, labelpad=-5)
        ax.set_ylabel("Dimension 2", fontsize=8, labelpad=-5)
        ax.set_zlabel("Dimension 3", fontsize=8, labelpad=-5)
        ax.tick_params(labelsize=6, pad=-2)
        ax.view_init(elev=view_elev, azim=view_azim)
        set_axes_equal_3d(ax, np.vstack([truth_plot, pred_plot]))

    handles = [
        mlines.Line2D([], [], color=TRUTH_COLOR, lw=1.4, label="Ground Truth"),
        mlines.Line2D([], [], color=PRED_COLOR, lw=1.4, label="DynaMix"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.945))
    if title:
        fig.suptitle(title, fontsize=17, fontweight="semibold", y=0.985)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
