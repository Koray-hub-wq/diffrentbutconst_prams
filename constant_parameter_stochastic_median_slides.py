"""Create stochastic median rollout comparison slides per constant phi value.

The script is specialized for constant-parameter test windows produced by
resample_constant_parameter_windows.py. It runs repeated stochastic forecasts for
all test trajectories, groups them by their constant normalized phi value, and
writes one comparison slide per phi value.

Selection rule per phi value:
  1. For each test trajectory and model, select the median rollout by a metric.
  2. Among the trajectories with the same phi, select the median trajectory by
     the same metric.

By default phi <= 0 uses DSTSP for selection and phi > 0 uses RMSE. Metrics in
the table are RMSE, MASE, DH, and DSTSP.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


MASTER = Path(
    r"C:\Users\koray\OneDrive\Dokumente\Studium\Master\Heidelberg\Master_Thesis"
)
REPO = MASTER / "DynaMix-python-b-tipping"
RESULT_ROOT = REPO / "results" / "single_halvorsen_constparams"

DEFAULT_WITH_PHI_RUN = RESULT_ROOT / "14to209_noisy-with-phi"
DEFAULT_NO_PHI_RUN = RESULT_ROOT / "14to209_noisy-no-phi"
DEFAULT_DATA_DIR = Path("14to209_noisy") / "windows"
DEFAULT_OUTPUT_DIR = Path("14to209_noisy") / "median_stochastic_slides"

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ROLLOUTS = 50
DEFAULT_PLOT_POINTS = 10000
N_BINS = 30
PS_SMOOTHING = 20
MASE_STEPS = 10
METRIC_NAMES = ("rmse", "mase", "dh", "dstsp")

_REPO_METRICS: dict[str, Callable[..., Any]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-phi-run", type=Path, default=DEFAULT_WITH_PHI_RUN)
    parser.add_argument("--no-phi-run", type=Path, default=DEFAULT_NO_PHI_RUN)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument(
        "--context-steps",
        type=int,
        default=None,
        help="Defaults to context.npy length from data-dir.",
    )
    parser.add_argument(
        "--nonpositive-phi-metric",
        choices=METRIC_NAMES,
        default="dstsp",
        help="Median-selection metric for phi <= 0.",
    )
    parser.add_argument(
        "--positive-phi-metric",
        choices=METRIC_NAMES,
        default="rmse",
        help="Median-selection metric for phi > 0.",
    )
    parser.add_argument(
        "--plot-points",
        type=int,
        default=DEFAULT_PLOT_POINTS,
        help=(
            "Maximum plotted points per 10000-step window. The full time span is "
            "shown; values above the window length plot every step."
        ),
    )
    parser.add_argument(
        "--phi-round-decimals",
        type=int,
        default=8,
        help="Decimals used to group constant phi values.",
    )
    return parser.parse_args()


def ensure_repo_src_on_path() -> None:
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def purge_dynamix_modules() -> None:
    for name in list(sys.modules):
        if name == "dynamix" or name.startswith("dynamix."):
            del sys.modules[name]


def import_dynamix():
    purge_dynamix_modules()
    ensure_repo_src_on_path()
    from dynamix.model.dynamix import DynaMix
    from dynamix.model.forecaster import DynaMixForecaster

    return DynaMix, DynaMixForecaster


def import_repo_metrics() -> dict[str, Callable[..., Any]]:
    global _REPO_METRICS
    if _REPO_METRICS is None:
        ensure_repo_src_on_path()
        from dynamix.metrics.metrics import (
            MASE,
            geometrical_misalignment,
            temporal_misalignment,
        )

        _REPO_METRICS = {
            "MASE": MASE,
            "geometrical_misalignment": geometrical_misalignment,
            "temporal_misalignment": temporal_misalignment,
        }
    return _REPO_METRICS


def find_checkpoint(run_dir: Path) -> Path:
    checkpoint_dir = run_dir / "checkpoints"
    for name in ("final_model.safetensors", "final_model.pt"):
        candidate = checkpoint_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No final_model checkpoint found in {checkpoint_dir}")


def load_state_dict(path: Path, device: str) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        if load_safetensors is None:
            raise ImportError("Install safetensors to load .safetensors checkpoints.")
        return load_safetensors(str(path), device=device)
    return torch.load(path, map_location=device)


def infer_architecture(
    config: dict[str, Any], state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    n_obs, latent_dim = map(int, state["B"].shape)
    expert_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("experts.") and key.split(".")[1].isdigit()
    }
    c_keys = [key for key in state if key.startswith("experts.") and key.endswith(".C")]
    phi_dim = (
        int(state[c_keys[0]].shape[1]) if c_keys else int(config.get("phi_dim", 0) or 0)
    )
    return {
        "M": latent_dim,
        "N": n_obs,
        "Experts": len(expert_indices) or int(config["experts"]),
        "P": int(config.get("pwl_units", 2)),
        "hidden_dim": int(config.get("hidden_dim", 50)),
        "expert_type": config.get("expert_type", "almost_linear_rnn"),
        "probabilistic_expert": bool(config.get("probabilistic_expert", False)),
        "phi_dim": phi_dim,
    }


def load_model(run_dir: Path, device: str):
    DynaMix, DynaMixForecaster = import_dynamix()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint_path = find_checkpoint(run_dir)
    state = load_state_dict(checkpoint_path, device)
    arch = infer_architecture(config, state)
    kwargs = {
        "M": arch["M"],
        "N": arch["N"],
        "Experts": arch["Experts"],
        "P": arch["P"],
        "hidden_dim": arch["hidden_dim"],
        "expert_type": arch["expert_type"],
        "probabilistic_expert": arch["probabilistic_expert"],
    }
    if "phi_dim" in inspect.signature(DynaMix).parameters:
        kwargs["phi_dim"] = arch["phi_dim"]
    model = DynaMix(**kwargs).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, DynaMixForecaster(model), config, arch, checkpoint_path


def load_test_data(data_dir: Path, context_steps_override: int | None):
    test = np.load(data_dir / "test.npy").astype(np.float32)
    test_phi = np.load(data_dir / "test_phi.npy").astype(np.float32)
    context = np.load(data_dir / "context.npy").astype(np.float32)
    context_steps = (
        int(context.shape[0]) if context_steps_override is None else context_steps_override
    )
    if test.ndim != 3:
        raise ValueError(f"Expected test.npy shape (T, S, D), got {test.shape}")
    if test_phi.ndim != 3 or test_phi.shape[:2] != test.shape[:2]:
        raise ValueError(
            f"Expected test_phi.npy shape (T, S, 1) matching test, got {test_phi.shape}"
        )
    if context_steps <= 0 or context_steps >= test.shape[0]:
        raise ValueError(
            f"context_steps must be in (0, {test.shape[0]}), got {context_steps}"
        )

    metadata_path = data_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return test, test_phi, context_steps, metadata


@torch.no_grad()
def forecast_once(
    model,
    forecaster,
    test: np.ndarray,
    test_phi: np.ndarray,
    context_steps: int,
    device: str,
) -> np.ndarray:
    horizon = test.shape[0] - context_steps
    ids = list(range(test.shape[1]))
    context_t = torch.tensor(test[:context_steps, ids, :], device=device)
    kwargs = {}
    if getattr(model, "phi_dim", 0) > 0:
        kwargs["phi_future"] = torch.tensor(
            test_phi[context_steps:, ids, :], device=device
        )
    pred = forecaster.forecast(context_t, horizon, **kwargs)
    return pred.detach().cpu().numpy().astype(np.float32)


def percent_improvement(old: float, new: float) -> float:
    if not np.isfinite(old) or abs(old) < 1e-12:
        return float("nan")
    return 100.0 * (old - new) / old


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    return math.sqrt(float(np.mean((pred - truth) ** 2)))


def metric_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32)


def call_with_supported_signatures(
    fn: Callable[..., Any], attempts: list[tuple[Any, ...]], name: str
) -> float:
    last_error = None
    for args in attempts:
        try:
            return float(fn(*args))
        except (TypeError, AttributeError) as exc:
            last_error = exc
    raise TypeError(f"Could not call {name} with supported signatures: {last_error}")


def repo_mase(truth: np.ndarray, pred: np.ndarray) -> float:
    fn = import_repo_metrics()["MASE"]
    truth_t = metric_tensor(truth)
    pred_t = metric_tensor(pred)
    return call_with_supported_signatures(
        fn,
        [
            (truth_t, pred_t, MASE_STEPS),
            (pred_t, truth_t, MASE_STEPS),
            (truth_t, pred_t),
            (pred_t, truth_t),
        ],
        "MASE",
    )


def repo_dh(truth: np.ndarray, pred: np.ndarray) -> float:
    fn = import_repo_metrics()["temporal_misalignment"]
    truth_t = metric_tensor(truth)
    pred_t = metric_tensor(pred)
    return call_with_supported_signatures(
        fn,
        [
            (pred_t, truth_t, N_BINS, PS_SMOOTHING),
            (truth_t, pred_t, N_BINS, PS_SMOOTHING),
            (pred_t, truth_t, N_BINS),
            (truth_t, pred_t, N_BINS),
            (pred_t, truth_t),
            (truth_t, pred_t),
        ],
        "temporal_misalignment",
    )


def repo_dstsp(truth: np.ndarray, pred: np.ndarray) -> float:
    fn = import_repo_metrics()["geometrical_misalignment"]
    truth_t = metric_tensor(truth)
    pred_t = metric_tensor(pred)
    return call_with_supported_signatures(
        fn,
        [
            (pred_t, truth_t, N_BINS),
            (truth_t, pred_t, N_BINS),
            (pred_t, truth_t),
            (truth_t, pred_t),
        ],
        "geometrical_misalignment",
    )


METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "rmse": rmse,
    "mase": repo_mase,
    "dh": repo_dh,
    "dstsp": repo_dstsp,
}


def compute_metric_scores(truth: np.ndarray, preds: list[np.ndarray]) -> dict[str, np.ndarray]:
    n_rollouts = len(preds)
    n_series = truth.shape[1]
    scores = {
        name: np.empty((n_rollouts, n_series), dtype=np.float64) for name in METRICS
    }
    for rollout_idx, pred in enumerate(preds):
        for series_idx in range(n_series):
            y_true = truth[:, series_idx, :]
            y_pred = pred[:, series_idx, :]
            for name, fn in METRICS.items():
                scores[name][rollout_idx, series_idx] = fn(y_true, y_pred)
    return scores


def median_index(values: np.ndarray) -> int:
    order = np.argsort(np.asarray(values, dtype=float))
    return int(order[len(order) // 2])


def group_series_by_phi(test_phi: np.ndarray, decimals: int) -> dict[float, np.ndarray]:
    phi_values = np.round(test_phi[0, :, 0].astype(float), decimals)
    groups = {}
    for phi in np.unique(phi_values):
        groups[float(phi)] = np.flatnonzero(np.isclose(phi_values, phi))
    return dict(sorted(groups.items(), key=lambda item: item[0]))


def selection_metric_for_phi(
    phi_value: float, nonpositive_metric: str, positive_metric: str
) -> str:
    return nonpositive_metric if phi_value <= 0.0 else positive_metric


def select_group_median(
    group_indices: np.ndarray,
    scores: dict[str, np.ndarray],
    preds: list[np.ndarray],
    metric: str,
) -> dict[str, Any]:
    per_series = []
    for series_idx in group_indices:
        rollout_idx = median_index(scores[metric][:, series_idx])
        per_series.append(
            {
                "series_idx": int(series_idx),
                "rollout_idx": int(rollout_idx),
                "selection_score": float(scores[metric][rollout_idx, series_idx]),
                "metrics": {
                    name: float(values[rollout_idx, series_idx])
                    for name, values in scores.items()
                },
            }
        )
    ordered = sorted(per_series, key=lambda item: item["selection_score"])
    selected = ordered[len(ordered) // 2]
    selected["prediction"] = preds[selected["rollout_idx"]][:, selected["series_idx"], :]
    selected["per_series"] = per_series
    return selected


def make_plot_indices(n_steps: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or max_points >= n_steps:
        return np.arange(n_steps)
    return np.unique(np.linspace(0, n_steps - 1, max_points).astype(int))


def full_prediction(test: np.ndarray, pred_future: np.ndarray, series_idx: int, context_steps: int):
    truth_full = test[:, series_idx, :]
    pred_full = np.full_like(truth_full, np.nan, dtype=np.float32)
    pred_full[context_steps:, :] = pred_future
    return truth_full, pred_full


def metric_line(metric: str, no_value: float, phi_value: float) -> str:
    return (
        f"{metric.upper():<5} no phi={no_value:.4g} | "
        f"phi={phi_value:.4g} | impr={percent_improvement(no_value, phi_value):+.1f}%"
    )


def add_metric_box(
    fig: plt.Figure,
    phi_value: float,
    metric: str,
    no_selected: dict[str, Any],
    phi_selected: dict[str, Any],
) -> None:
    lines = [
        f"phi={phi_value:.6g}, selection metric={metric.upper()}",
        (
            f"with phi: test {phi_selected['series_idx']} rollout {phi_selected['rollout_idx']} | "
            f"no phi: test {no_selected['series_idx']} rollout {no_selected['rollout_idx']}"
        ),
    ]
    lines.extend(
        metric_line(
            name,
            no_selected["metrics"][name],
            phi_selected["metrics"][name],
        )
        for name in METRIC_NAMES
    )
    fig.text(
        0.5,
        0.01,
        "\n".join(lines),
        ha="center",
        va="bottom",
        family="monospace",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.95},
    )


def plot_one_side_3d(ax, truth: np.ndarray, pred: np.ndarray, title: str) -> None:
    ax.plot(
        truth[:, 0],
        truth[:, 1],
        truth[:, 2],
        color="#8f9aa5",
        lw=1.4,
        label="Ground Truth",
    )
    valid = np.isfinite(pred[:, 0])
    ax.plot(
        pred[valid, 0],
        pred[valid, 1],
        pred[valid, 2],
        color="#ff4d4f",
        lw=1.4,
        label="DynaMix",
    )
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")
    ax.legend()


def plot_phi_slide(
    output_path: Path,
    test: np.ndarray,
    phi_value: float,
    metric: str,
    context_steps: int,
    no_selected: dict[str, Any],
    phi_selected: dict[str, Any],
    plot_points: int,
) -> None:
    phi_truth, phi_pred = full_prediction(
        test, phi_selected["prediction"], phi_selected["series_idx"], context_steps
    )
    no_truth, no_pred = full_prediction(
        test, no_selected["prediction"], no_selected["series_idx"], context_steps
    )

    plot_idx = make_plot_indices(test.shape[0], plot_points)
    t = plot_idx
    phi_truth_plot = phi_truth[plot_idx]
    phi_pred_plot = phi_pred[plot_idx]
    no_truth_plot = no_truth[plot_idx]
    no_pred_plot = no_pred[plot_idx]

    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(3, 4, width_ratios=[1.15, 1.15, 1.1, 1.1])
    ax_phi_3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_no_3d = fig.add_subplot(gs[:, 1], projection="3d")
    ts_phi = [fig.add_subplot(gs[i, 2]) for i in range(3)]
    ts_no = [fig.add_subplot(gs[i, 3]) for i in range(3)]

    plot_one_side_3d(
        ax_phi_3d,
        phi_truth_plot,
        phi_pred_plot,
        f"with phi | test {phi_selected['series_idx']}",
    )
    plot_one_side_3d(
        ax_no_3d,
        no_truth_plot,
        no_pred_plot,
        f"no phi | test {no_selected['series_idx']}",
    )

    for dim in range(3):
        for ax, truth, pred, title in (
            (ts_phi[dim], phi_truth_plot, phi_pred_plot, "with phi"),
            (ts_no[dim], no_truth_plot, no_pred_plot, "no phi"),
        ):
            ax.plot(
                t,
                truth[:, dim],
                color="#8f9aa5",
                lw=1.4,
                label="Ground Truth" if dim == 0 else None,
            )
            ax.plot(
                t,
                pred[:, dim],
                color="#ff4d4f",
                lw=1.4,
                label="DynaMix" if dim == 0 else None,
            )
            ax.axvline(context_steps, color="0.25", ls="--", lw=0.8, alpha=0.65)
            ax.set_xlim(0, test.shape[0] - 1)
            ax.set_ylabel(f"Dim {dim + 1}")
            ax.grid(alpha=0.25)
            if dim == 0:
                ax.set_title(title)
                ax.legend()
            if dim == 2:
                ax.set_xlabel("Test-window time")

    improvement = percent_improvement(
        no_selected["metrics"][metric], phi_selected["metrics"][metric]
    )
    fig.suptitle(
        f"Stochastic median rollout | phi {phi_value:.6g} | "
        f"{metric.upper()} improvement {improvement:+.2f}%",
        fontsize=15,
    )
    add_metric_box(fig, phi_value, metric, no_selected, phi_selected)
    fig.tight_layout(rect=[0, 0.13, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def strip_predictions(selection: dict[str, Any]) -> dict[str, Any]:
    def to_jsonable(value):
        if isinstance(value, dict):
            return {
                key: to_jsonable(item)
                for key, item in value.items()
                if key != "prediction"
            }
        if isinstance(value, list):
            return [to_jsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return to_jsonable(selection)


def main() -> None:
    args = parse_args()
    if args.rollouts <= 0:
        raise ValueError("rollouts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data: {args.data_dir}")
    print(f"With phi: {args.with_phi_run}")
    print(f"No phi: {args.no_phi_run}")
    print(f"Device: {args.device}")
    print(f"Rollouts: {args.rollouts}")
    print(
        "Selection metrics: "
        f"phi<=0 -> {args.nonpositive_phi_metric}, phi>0 -> {args.positive_phi_metric}"
    )

    test, test_phi, context_steps, data_metadata = load_test_data(
        args.data_dir, args.context_steps
    )
    truth_future = test[context_steps:, :, :]
    phi_groups = group_series_by_phi(test_phi, args.phi_round_decimals)
    print(f"test: {test.shape}, context_steps={context_steps}")
    print(
        "Phi groups: "
        + ", ".join(f"{phi:g}: {len(indices)}" for phi, indices in phi_groups.items())
    )

    no_model, no_forecaster, *_ = load_model(args.no_phi_run, args.device)
    phi_model, phi_forecaster, *_ = load_model(args.with_phi_run, args.device)

    no_preds: list[np.ndarray] = []
    phi_preds: list[np.ndarray] = []
    for rollout_idx in range(args.rollouts):
        print(f"Rollout {rollout_idx + 1}/{args.rollouts}")
        no_preds.append(
            forecast_once(
                no_model, no_forecaster, test, test_phi, context_steps, args.device
            )
        )
        phi_preds.append(
            forecast_once(
                phi_model, phi_forecaster, test, test_phi, context_steps, args.device
            )
        )

    print("Computing no-phi metrics")
    no_scores = compute_metric_scores(truth_future, no_preds)
    print("Computing with-phi metrics")
    phi_scores = compute_metric_scores(truth_future, phi_preds)

    summary: dict[str, Any] = {
        "rollouts": args.rollouts,
        "data_dir": str(args.data_dir),
        "data_metadata": data_metadata,
        "with_phi_run": str(args.with_phi_run),
        "no_phi_run": str(args.no_phi_run),
        "context_steps": context_steps,
        "test_shape": list(test.shape),
        "selection": {
            "nonpositive_phi_metric": args.nonpositive_phi_metric,
            "positive_phi_metric": args.positive_phi_metric,
        },
        "phi_values": {},
    }

    for phi_value, group_indices in phi_groups.items():
        metric = selection_metric_for_phi(
            phi_value, args.nonpositive_phi_metric, args.positive_phi_metric
        )
        print(f"Selecting phi={phi_value:g} with {metric.upper()}")
        no_selected = select_group_median(group_indices, no_scores, no_preds, metric)
        phi_selected = select_group_median(group_indices, phi_scores, phi_preds, metric)

        safe_phi = f"{phi_value:+.6g}".replace("+", "p").replace("-", "m").replace(".", "p")
        slide_path = args.output_dir / f"median_rollout_phi_{safe_phi}.png"
        plot_phi_slide(
            slide_path,
            test,
            phi_value,
            metric,
            context_steps,
            no_selected,
            phi_selected,
            args.plot_points,
        )

        summary["phi_values"][str(phi_value)] = {
            "series_indices": [int(i) for i in group_indices],
            "selection_metric": metric,
            "slide_path": str(slide_path),
            "no_phi": strip_predictions(no_selected),
            "with_phi": strip_predictions(phi_selected),
            "metric_improvements_pct": {
                name: percent_improvement(
                    no_selected["metrics"][name], phi_selected["metrics"][name]
                )
                for name in METRIC_NAMES
            },
        }

    summary_path = args.output_dir / "constant_parameter_stochastic_median_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Written summary: {summary_path}")
    print(f"Written slides: {args.output_dir / 'median_rollout_phi_*.png'}")


if __name__ == "__main__":
    main()
