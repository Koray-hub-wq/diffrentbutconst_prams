from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from constant_parameter_stochastic_median_slides_server import (
    compute_metric_scores,
    configure_runtime,
    forecast_once,
    load_model,
    selection_metric_for_phi,
    set_repo_root,
)
from with_phi_summary_plot_utils import (
    context_steps_from_data_dir,
    group_series_by_phi,
    load_metadata,
    plot_prediction_grid,
    raw_from_phi,
    resolve_requested_phi_values,
    select_group_median_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run stochastic DynaMix rollouts, select one median prediction "
            "per requested parameter value, save a cache, and write a 3D summary plot."
        )
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--model-run",
        "--with-phi-run",
        dest="model_run",
        type=Path,
        required=True,
        help=(
            "Model run directory. --with-phi-run is kept as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--model-label",
        default="DynaMix",
        help="Legend/cache label for the prediction curve, e.g. 'with phi' or 'no phi'.",
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--cache-name", default=None)
    parser.add_argument("--summary-name", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--context-steps", type=int, default=None)
    parser.add_argument(
        "--metric-steps",
        "--rmse-steps",
        dest="metric_steps",
        type=int,
        default=1000,
        help=(
            "Forecast steps used for RMSE computation and RMSE-based median "
            "selection. Non-RMSE metrics use the full horizon. Use 0 for full-horizon RMSE."
        ),
    )
    parser.add_argument(
        "--nonpositive-phi-metric",
        choices=("rmse", "mase", "dh", "dstsp"),
        default="dstsp",
        help="Median-selection metric for phi <= 0, matching the slide server default.",
    )
    parser.add_argument(
        "--positive-phi-metric",
        choices=("rmse", "mase", "dh", "dstsp"),
        default="rmse",
        help="Median-selection metric for phi > 0, matching the slide server default.",
    )
    parser.add_argument("--phi-round-decimals", type=int, default=8)
    parser.add_argument(
        "--params",
        nargs="+",
        type=float,
        default=None,
        help="Parameter values to plot. Interpreted as phi unless --parameter-space raw is set.",
    )
    parser.add_argument("--parameter-space", choices=("phi", "raw"), default="phi")
    parser.add_argument("--parameter-atol", type=float, default=1e-5)
    parser.add_argument("--label-space", choices=("phi", "raw", "both", "none"), default="phi")
    parser.add_argument("--raw-name", default="a")
    parser.add_argument(
        "--single-row",
        action="store_true",
        help="Place all selected parameter plots in one row instead of wrapping after three.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--view-elev", type=float, default=22.0)
    parser.add_argument("--view-azim", type=float, default=-58.0)
    return parser.parse_args()


def safe_label_stem(label: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in label.strip()]
    stem = "".join(chars).strip("_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem or "model"


def save_cache(
    path: Path,
    *,
    truth_future: np.ndarray,
    pred_future: np.ndarray,
    phi_values: np.ndarray,
    raw_values: np.ndarray,
    series_indices: np.ndarray,
    rollout_indices: np.ndarray,
    selection_scores: np.ndarray,
    selection_metrics: np.ndarray,
    model_label: str,
    context_steps: int,
    metric_steps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        truth_future=truth_future.astype(np.float32),
        pred_future=pred_future.astype(np.float32),
        phi_values=phi_values.astype(np.float64),
        raw_values=raw_values.astype(np.float64),
        series_indices=series_indices.astype(np.int64),
        rollout_indices=rollout_indices.astype(np.int64),
        selection_scores=selection_scores.astype(np.float64),
        selection_metrics=selection_metrics.astype(str),
        model_label=np.array(model_label),
        context_steps=np.array(context_steps, dtype=np.int64),
        metric_steps=np.array(metric_steps, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    label_stem = safe_label_stem(args.model_label)
    output_name = args.output_name or f"{label_stem}_median_prediction_grid.png"
    cache_name = args.cache_name or f"{label_stem}_median_prediction_cache.npz"
    summary_name = args.summary_name or f"{label_stem}_median_prediction_cache_summary.json"

    device = configure_runtime(args)
    set_repo_root(args.repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.data_dir)
    test = np.load(args.data_dir / "test.npy").astype(np.float32)
    test_phi = np.load(args.data_dir / "test_phi.npy").astype(np.float32)
    context_steps = context_steps_from_data_dir(args.data_dir, args.context_steps)
    if context_steps <= 0 or context_steps >= test.shape[0]:
        raise ValueError(f"context_steps must be in (0, {test.shape[0]}), got {context_steps}")

    truth_all = test[context_steps:, :, :]
    forecast_horizon = truth_all.shape[0]
    metric_steps = forecast_horizon if args.metric_steps == 0 else min(args.metric_steps, forecast_horizon)
    phi_groups = group_series_by_phi(test_phi, args.phi_round_decimals)
    available_phi = np.array(list(phi_groups.keys()), dtype=float)
    available_raw = raw_from_phi(available_phi, metadata).astype(float)
    selected_param_indices = resolve_requested_phi_values(
        available_phi, available_raw, args.params, args.parameter_space, args.parameter_atol
    )

    print(f"Data: {args.data_dir}")
    print(f"test={test.shape}, context_steps={context_steps}, horizon={forecast_horizon}")
    print(
        f"Selection metrics: phi<=0 -> {args.nonpositive_phi_metric}, "
        f"phi>0 -> {args.positive_phi_metric}"
    )
    print(
        f"RMSE horizon: {metric_steps}/{forecast_horizon} forecast steps; "
        "MASE, DH and DSTSP use the full forecast horizon"
    )
    print(
        "Selected parameters: "
        + ", ".join(
            f"phi={available_phi[i]:.6g}, {args.raw_name}={available_raw[i]:.6g}"
            for i in selected_param_indices
        )
    )

    print(f"Model run: {args.model_run}")
    print(f"Model label: {args.model_label}")
    model, forecaster, *_ = load_model(args.model_run, device)
    preds: list[np.ndarray] = []
    for rollout_idx in range(args.rollouts):
        print(f"Rollout {rollout_idx + 1}/{args.rollouts}", flush=True)
        pred = forecast_once(model, forecaster, test, test_phi, context_steps, device)
        preds.append(pred)

    print("Computing model metrics", flush=True)
    scores = compute_metric_scores(truth_all, preds, metric_steps)

    selected_truth = []
    selected_pred = []
    selected_phi = []
    selected_raw = []
    selected_series = []
    selected_rollouts = []
    selected_scores = []
    selected_metrics = []
    summary: dict[str, Any] = {
        "rollouts": args.rollouts,
        "repo_root": str(args.repo_root),
        "model_run": str(args.model_run),
        "model_label": args.model_label,
        "data_dir": str(args.data_dir),
        "context_steps": context_steps,
        "forecast_horizon": int(forecast_horizon),
        "rmse_steps": int(metric_steps),
        "full_horizon_metrics": ["mase", "dh", "dstsp"],
        "selection": {
            "nonpositive_phi_metric": args.nonpositive_phi_metric,
            "positive_phi_metric": args.positive_phi_metric,
        },
        "selected": [],
    }

    for idx in selected_param_indices:
        phi = float(available_phi[idx])
        raw = float(available_raw[idx])
        group_indices = phi_groups[phi]
        metric = selection_metric_for_phi(
            phi, args.nonpositive_phi_metric, args.positive_phi_metric
        )
        selected = select_group_median_prediction(group_indices, scores[metric], preds)
        series_idx = int(selected["series_idx"])
        selected_truth.append(truth_all[:, series_idx, :])
        selected_pred.append(selected["prediction"])
        selected_phi.append(phi)
        selected_raw.append(raw)
        selected_series.append(series_idx)
        selected_rollouts.append(int(selected["rollout_idx"]))
        selected_scores.append(float(selected["score"]))
        selected_metrics.append(metric)
        summary["selected"].append(
            {
                "phi": phi,
                args.raw_name: raw,
                "series_indices": [int(i) for i in group_indices],
                "selection_metric": metric,
                "selected_series_idx": series_idx,
                "selected_rollout_idx": int(selected["rollout_idx"]),
                "selection_score": float(selected["score"]),
                "rank": int(selected["rank"]),
                "rank_total": int(selected["rank_total"]),
            }
        )
        print(
            f"phi={phi:g}: selected series {series_idx}, rollout {selected['rollout_idx']}, "
            f"{metric.upper()}={selected['score']:.6g}"
        )

    truth_arr = np.stack(selected_truth, axis=0)
    pred_arr = np.stack(selected_pred, axis=0)
    phi_arr = np.asarray(selected_phi, dtype=float)
    raw_arr = np.asarray(selected_raw, dtype=float)
    cache_path = args.output_dir / cache_name
    save_cache(
        cache_path,
        truth_future=truth_arr,
        pred_future=pred_arr,
        phi_values=phi_arr,
        raw_values=raw_arr,
        series_indices=np.asarray(selected_series),
        rollout_indices=np.asarray(selected_rollouts),
        selection_scores=np.asarray(selected_scores),
        selection_metrics=np.asarray(selected_metrics),
        model_label=args.model_label,
        context_steps=context_steps,
        metric_steps=metric_steps,
    )

    output_path = args.output_dir / output_name
    plot_prediction_grid(
        output_path,
        truth_arr,
        pred_arr,
        phi_arr,
        raw_arr,
        title=args.title,
        label_space=args.label_space,
        raw_name=args.raw_name,
        prediction_label=args.model_label,
        single_row=args.single_row,
        dpi=args.dpi,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )

    summary["cache_path"] = str(cache_path)
    summary["plot_path"] = str(output_path)
    summary_path = args.output_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Written cache: {cache_path}")
    print(f"Written plot: {output_path}")
    print(f"Written summary: {summary_path}")


if __name__ == "__main__":
    main()
