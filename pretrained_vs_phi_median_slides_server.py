"""Compare original pretrained DynaMix against one locally trained phi model.

This script keeps constant_parameter_stochastic_median_slides_server.py unchanged
and reuses its plotting, metrics, and median-selection helpers.

The comparison slots are:
  no_phi   -> original pretrained Hugging Face DynaMix baseline
  with_phi -> local trained phi-conditioned run
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import constant_parameter_stochastic_median_slides_server as slides


DEFAULT_HF_MODEL = "dynamix-3d-alrnn-v1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Path to the modified DynaMix repo used by your phi run and metrics.",
    )
    parser.add_argument(
        "--hf-repo-root",
        type=Path,
        required=True,
        help="Path to the original DurstewitzLab/DynaMix-python repository.",
    )
    parser.add_argument("--with-phi-run", type=Path, required=True)
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=slides.DEFAULT_DEVICE)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument(
        "--context-steps",
        type=int,
        default=None,
        help="Defaults to context.npy length from data-dir.",
    )
    parser.add_argument(
        "--forecast-preprocessing-method",
        choices=("pos_embedding", "zero_embedding", "delay_embedding", "delay_embedding_random"),
        default=None,
        help="Optional preprocessing_method for the pretrained DynaMix forecaster.",
    )
    parser.add_argument(
        "--no-hf-standardize",
        action="store_true",
        help="Pass standardize=False to the pretrained DynaMix forecaster.",
    )
    parser.add_argument(
        "--nonpositive-phi-metric",
        choices=slides.METRIC_NAMES,
        default="dstsp",
    )
    parser.add_argument(
        "--positive-phi-metric",
        choices=slides.METRIC_NAMES,
        default="rmse",
    )
    parser.add_argument("--plot-points", type=int, default=None)
    parser.add_argument(
        "--timeseries-prediction-steps",
        type=int,
        default=slides.DEFAULT_TIMESERIES_PREDICTION_STEPS,
    )
    parser.add_argument(
        "--rmse-steps",
        "--metric-steps",
        dest="rmse_steps",
        type=int,
        default=slides.DEFAULT_RMSE_STEPS,
        help="Use 0 for full-horizon RMSE.",
    )
    parser.add_argument("--phi-round-decimals", type=int, default=8)
    parser.add_argument(
        "--slide-modes",
        nargs="+",
        choices=("all", "own", "with_phi_anchor", "no_phi_anchor"),
        default=["all"],
    )
    return parser.parse_args()


def put_repo_src_first(repo_root: Path) -> None:
    src = repo_root.resolve() / "src"
    if not src.exists():
        raise FileNotFoundError(f"Expected DynaMix src directory at {src}")
    src_str = str(src)
    if src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)


def load_hf_pretrained_model(hf_repo_root: Path, model_name: str, device: str):
    slides.purge_dynamix_modules()
    put_repo_src_first(hf_repo_root)
    from dynamix.model.forecaster import DynaMixForecaster
    from dynamix.utilities.utilities import load_hf_model

    model = load_hf_model(model_name).to(device)
    model.eval()
    return model, DynaMixForecaster(model)


def load_phi_model(repo_root: Path, run_dir: Path, device: str):
    put_repo_src_first(repo_root)
    slides.set_repo_root(repo_root)
    slides._REPO_METRICS = None
    return slides.load_model(run_dir, device)[:2]


@torch.no_grad()
def forecast_hf_once(
    model,
    forecaster,
    test: np.ndarray,
    context_steps: int,
    device: str,
    preprocessing_method: str | None,
    standardize: bool | None,
) -> np.ndarray:
    horizon = test.shape[0] - context_steps
    context_t = torch.tensor(test[:context_steps, :, :], device=device)
    kwargs: dict[str, Any] = {}
    forecast_params = inspect.signature(forecaster.forecast).parameters
    if preprocessing_method is not None and "preprocessing_method" in forecast_params:
        kwargs["preprocessing_method"] = preprocessing_method
    if standardize is not None and "standardize" in forecast_params:
        kwargs["standardize"] = standardize
    pred = forecaster.forecast(context_t, horizon, **kwargs)
    return pred.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    if args.rollouts <= 0:
        raise ValueError("rollouts must be positive")

    device = slides.configure_runtime(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slide_modes = slides.selected_slide_modes(args.slide_modes)
    for mode in slide_modes:
        slides.mode_output_dir(args.output_dir, mode).mkdir(parents=True, exist_ok=True)

    test, test_phi, context_steps, data_metadata = slides.load_test_data(
        args.data_dir, args.context_steps
    )
    truth_future = test[context_steps:, :, :]
    forecast_horizon = truth_future.shape[0]
    if args.rmse_steps < 0:
        raise ValueError("rmse_steps must be non-negative")
    rmse_steps = forecast_horizon if args.rmse_steps == 0 else args.rmse_steps
    rmse_steps = min(rmse_steps, forecast_horizon)
    phi_groups = slides.group_series_by_phi(test_phi, args.phi_round_decimals)

    print(f"Modified repo root: {args.repo_root.resolve()}")
    print(f"HF repo root: {args.hf_repo_root.resolve()}")
    print(f"HF baseline: {args.hf_model}")
    print(f"With-phi run: {args.with_phi_run}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Rollouts: {args.rollouts}")
    print(f"test: {test.shape}, context_steps={context_steps}")
    print(f"Forecast horizon: {forecast_horizon}; RMSE horizon: {rmse_steps}")
    print("Slide modes: " + ", ".join(slide_modes))
    print(
        "Phi groups: "
        + ", ".join(f"{phi:g}: {len(indices)}" for phi, indices in phi_groups.items())
    )

    hf_model, hf_forecaster = load_hf_pretrained_model(
        args.hf_repo_root, args.hf_model, device
    )
    phi_model, phi_forecaster = load_phi_model(args.repo_root, args.with_phi_run, device)
    hf_standardize = False if args.no_hf_standardize else None

    hf_preds: list[np.ndarray] = []
    phi_preds: list[np.ndarray] = []
    for rollout_idx in range(args.rollouts):
        print(f"Rollout {rollout_idx + 1}/{args.rollouts}", flush=True)
        hf_preds.append(
            forecast_hf_once(
                hf_model,
                hf_forecaster,
                test,
                context_steps,
                device,
                args.forecast_preprocessing_method,
                hf_standardize,
            )
        )
        phi_preds.append(
            slides.forecast_once(
                phi_model,
                phi_forecaster,
                test,
                test_phi,
                context_steps,
                device,
            )
        )

    print("Computing pretrained baseline metrics", flush=True)
    hf_scores = slides.compute_metric_scores(truth_future, hf_preds, rmse_steps)
    print("Computing with-phi metrics", flush=True)
    phi_scores = slides.compute_metric_scores(truth_future, phi_preds, rmse_steps)

    summary: dict[str, Any] = {
        "rollouts": args.rollouts,
        "repo_root": str(args.repo_root),
        "hf_repo_root": str(args.hf_repo_root),
        "hf_model": args.hf_model,
        "with_phi_run": str(args.with_phi_run),
        "data_dir": str(args.data_dir),
        "data_metadata": data_metadata,
        "context_steps": context_steps,
        "test_shape": list(test.shape),
        "forecast_horizon": int(forecast_horizon),
        "rmse_steps": int(rmse_steps),
        "forecast_preprocessing_method": args.forecast_preprocessing_method,
        "hf_standardize_override": hf_standardize,
        "comparison_slots": {
            "no_phi": "original_pretrained_hf_dynamix",
            "with_phi": "local_phi_conditioned_run",
        },
        "selection": {
            "nonpositive_phi_metric": args.nonpositive_phi_metric,
            "positive_phi_metric": args.positive_phi_metric,
            "slide_modes": slide_modes,
        },
        "phi_values": {},
    }

    for phi_value, group_indices in phi_groups.items():
        metric = slides.selection_metric_for_phi(
            phi_value, args.nonpositive_phi_metric, args.positive_phi_metric
        )
        print(f"Selecting phi={phi_value:g} with {metric.upper()}", flush=True)
        hf_selected = slides.select_group_median(group_indices, hf_scores, hf_preds, metric)
        phi_selected = slides.select_group_median(group_indices, phi_scores, phi_preds, metric)
        hf_selected = slides.add_group_rank(hf_selected, group_indices, hf_scores, metric)
        phi_selected = slides.add_group_rank(phi_selected, group_indices, phi_scores, metric)

        safe_phi = (
            f"{phi_value:+.6g}".replace("+", "p").replace("-", "m").replace(".", "p")
        )
        mode_selections: dict[str, dict[str, Any]] = {
            "own": {
                "no_phi": hf_selected,
                "with_phi": phi_selected,
            },
            "with_phi_anchor": {
                "no_phi": slides.select_fixed_series_median(
                    int(phi_selected["series_idx"]),
                    group_indices,
                    hf_scores,
                    hf_preds,
                    metric,
                ),
                "with_phi": phi_selected,
            },
            "no_phi_anchor": {
                "no_phi": hf_selected,
                "with_phi": slides.select_fixed_series_median(
                    int(hf_selected["series_idx"]),
                    group_indices,
                    phi_scores,
                    phi_preds,
                    metric,
                ),
            },
        }

        summary["phi_values"][str(phi_value)] = {
            "series_indices": [int(i) for i in group_indices],
            "selection_metric": metric,
            "slide_sets": {},
        }
        for mode in slide_modes:
            mode_dir = slides.mode_output_dir(args.output_dir, mode)
            slide_path = mode_dir / f"median_rollout_phi_{safe_phi}.png"
            mode_hf_selected = mode_selections[mode]["no_phi"]
            mode_phi_selected = mode_selections[mode]["with_phi"]
            slides.plot_phi_slide(
                slide_path,
                test,
                phi_value,
                metric,
                context_steps,
                mode_hf_selected,
                mode_phi_selected,
                args.plot_points,
                args.timeseries_prediction_steps,
                slides.mode_title(mode),
            )
            summary["phi_values"][str(phi_value)]["slide_sets"][mode] = {
                "slide_path": str(slide_path),
                "pretrained_hf": slides.strip_predictions(mode_hf_selected),
                "with_phi": slides.strip_predictions(mode_phi_selected),
                "metric_improvements_pct": {
                    name: slides.percent_improvement(
                        mode_hf_selected["metrics"][name],
                        mode_phi_selected["metrics"][name],
                    )
                    for name in slides.METRIC_NAMES
                },
            }

    summary_path = args.output_dir / "pretrained_vs_phi_median_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Written summary: {summary_path}")
    for mode in slide_modes:
        print(
            f"Written {mode} slides: "
            f"{slides.mode_output_dir(args.output_dir, mode) / 'median_rollout_phi_*.png'}"
        )


if __name__ == "__main__":
    main()
