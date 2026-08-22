r"""Compare trained DynaMix models with a true bifurcation diagram.

For every value in one shared parameter sweep, the script simulates the chosen
regime with a constant parameter, standardizes that trajectory in the same style
as the training-data generator, forecasts from the initial context with each
model, caches local maxima for all three components, and plots selected
components.

Example:

python compare_model_bifurcation_diagrams.py \
  --repo-root /path/to/DynaMix-python-b-tipping \
  --regime halvorsen \
  --model-runs /path/to/run1 /path/to/run2 \
  --model-labels with_phi no_phi \
  --output outputs/halvorsen_bifurcation_compare.png \
  --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
from scipy.integrate import solve_ivp
from tqdm import tqdm

from constant_parameter_stochastic_median_slides_server import (  # noqa: E402
    configure_runtime,
    load_model,
    set_repo_root,
)


def import_regime_module():
    script_dir = Path(__file__).resolve().parent
    local_file = script_dir / "generate_dynamic_regime_trajectories.py"
    if local_file.exists():
        spec = importlib.util.spec_from_file_location(
            "local_generate_dynamic_regime_trajectories", local_file
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load regime module from {local_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    parent = script_dir.parent
    if str(parent) not in sys.path:
        sys.path.insert(0, str(parent))
    try:
        from time_varying_bifurcation_data import generate_dynamic_regime_trajectories
    except ModuleNotFoundError as exc:
        if exc.name != "time_varying_bifurcation_data":
            raise
        raise ModuleNotFoundError(
            "Could not import the regime definitions. Copy "
            "generate_dynamic_regime_trajectories.py into the same directory as "
            "compare_model_bifurcation_diagrams.py, or provide the original "
            "time_varying_bifurcation_data package next to diffrentbutconst_prams."
        ) from exc
    return generate_dynamic_regime_trajectories


_REGIME_MODULE = import_regime_module()
RHS = _REGIME_MODULE.RHS
Regime = _REGIME_MODULE.Regime
default_regimes = _REGIME_MODULE.default_regimes
normalize_phi = _REGIME_MODULE.normalize_phi
observe_state = _REGIME_MODULE.observe_state
standardize_trajectories = _REGIME_MODULE.standardize_trajectories


COMPONENTS = {"x": 0, "y": 1, "z": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Path to the DynaMix repository whose src directory should be used.",
    )
    parser.add_argument(
        "--model-runs",
        nargs="+",
        type=Path,
        required=True,
        help="One or more trained run directories containing config.json/checkpoints.",
    )
    parser.add_argument(
        "--model-labels",
        nargs="+",
        default=None,
        help="Optional labels for --model-runs. Defaults to the run folder names.",
    )
    parser.add_argument(
        "--regime",
        default="halvorsen_a_chaotic_to_periodic",
        help=(
            "Dynamical-system regime from generate_dynamic_regime_trajectories.py. "
            "Exact names and unique short matches such as 'halvorsen' are accepted."
        ),
    )
    parser.add_argument(
        "--list-regimes",
        action="store_true",
        help="Print available regimes and exit.",
    )
    parser.add_argument(
        "--parameter-min",
        type=float,
        default=None,
        help="Raw parameter sweep minimum. Defaults to the regime start/end range.",
    )
    parser.add_argument(
        "--parameter-max",
        type=float,
        default=None,
        help="Raw parameter sweep maximum. Defaults to the regime start/end range.",
    )
    parser.add_argument(
        "--phi-min",
        type=float,
        default=None,
        help="Normalized phi sweep minimum. Use instead of --parameter-min/max.",
    )
    parser.add_argument(
        "--phi-max",
        type=float,
        default=None,
        help="Normalized phi sweep maximum. Use instead of --parameter-min/max.",
    )
    parser.add_argument("--num-parameters", type=int, default=180)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--context-steps",
        type=int,
        default=500,
        help="Number of true simulated steps used as model context.",
    )
    parser.add_argument(
        "--forecast-steps",
        type=int,
        default=3000,
        help="Number of forecast steps used to extract model local maxima.",
    )
    parser.add_argument(
        "--truth-steps",
        type=int,
        default=None,
        help="Number of post-context true steps used for local maxima. Defaults to forecast-steps.",
    )
    parser.add_argument(
        "--burn-in-steps",
        type=int,
        default=5000,
        help="Burn-in at each constant parameter before recording.",
    )
    parser.add_argument(
        "--components",
        "--component",
        nargs="+",
        choices=tuple(COMPONENTS),
        default=["x"],
        help="One or more state components used for bifurcation diagrams.",
    )
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--max-step-multiplier", type=float, default=5.0)
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "Torch device. Defaults to cuda for Linux server runs and fails if "
            "CUDA is unavailable. Pass cpu explicitly only for debugging."
        ),
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("model_bifurcation_compare.png"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help=(
            "Cache file for computed bifurcation points. Defaults to the output "
            "path with .npz suffix. Existing compatible cache files are reused."
        ),
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore an existing cache and recompute the sweep.",
    )
    return parser.parse_args()


def regime_by_name(name: str) -> Regime:
    regimes = default_regimes()
    exact = [regime for regime in regimes if regime.name == name]
    if exact:
        return exact[0]

    query = name.lower()
    matches = [
        regime
        for regime in regimes
        if query in regime.name.lower()
        or query == regime.family.lower()
        or query == regime.rhs_key.lower()
    ]
    if len(matches) != 1:
        available = ", ".join(regime.name for regime in regimes)
        if matches:
            matching = ", ".join(regime.name for regime in matches)
            raise ValueError(
                f"Regime {name!r} is ambiguous. Matching regimes: {matching}. "
                f"Use one exact name. Available: {available}"
            )
        raise ValueError(f"Unknown regime {name!r}. Available: {available}")
    return matches[0]


def raw_from_phi(regime: Regime, phi: np.ndarray | float) -> np.ndarray:
    scale = max(abs(regime.start - regime.boundary), abs(regime.end - regime.boundary), 1e-8)
    return regime.boundary + np.asarray(phi, dtype=np.float64) * scale


def normalized_phi(regime: Regime, raw_parameter: float) -> float:
    return float(
        normalize_phi(
            np.array([raw_parameter], dtype=np.float32),
            regime.start,
            regime.end,
            regime.boundary,
        )[0]
    )


def local_maxima(values: np.ndarray) -> np.ndarray:
    return values[1:-1][(values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])]


def simulate_standardized_constant_parameter(
    regime: Regime,
    raw_parameter: float,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
    rtol: float,
    atol: float,
    max_step_multiplier: float,
    burn_in_steps: int,
) -> np.ndarray:
    if n_steps <= 2:
        raise ValueError("Need at least three simulated steps for local maxima.")

    rhs = RHS[regime.rhs_key]
    t_eval = np.arange(n_steps, dtype=np.float64) * dt
    max_step = dt * max_step_multiplier
    x0 = rng.normal(loc=np.array(regime.init_center), scale=np.array(regime.init_scale))
    if regime.observation == "phase_sine":
        x0[2] = rng.uniform(0.0, 2.0 * np.pi)

    def ode(_t: float, x: np.ndarray) -> np.ndarray:
        dx = rhs(_t, x, raw_parameter, **regime.params)
        return np.clip(dx, -1e4, 1e4)

    if burn_in_steps > 0:
        burn = solve_ivp(
            ode,
            (0.0, burn_in_steps * dt),
            x0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )
        if not burn.success:
            raise RuntimeError(
                f"Burn-in failed for {regime.name}, {regime.parameter}={raw_parameter:.6g}: {burn.message}"
            )
        x0 = burn.y[:, -1]

    sol = solve_ivp(
        ode,
        (0.0, float(t_eval[-1])),
        x0,
        t_eval=t_eval,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success or sol.y.shape[1] != n_steps:
        raise RuntimeError(
            f"Integration failed for {regime.name}, {regime.parameter}={raw_parameter:.6g}: {sol.message}"
        )

    observed = observe_state(sol.y.T, regime.observation).astype(np.float32)
    standardized, _, _ = standardize_trajectories(observed[None, :, :])
    return standardized[0]


def parameter_values_from_args(args: argparse.Namespace, regime: Regime) -> np.ndarray:
    using_phi = args.phi_min is not None or args.phi_max is not None
    using_raw = args.parameter_min is not None or args.parameter_max is not None
    if using_phi and using_raw:
        raise ValueError("Use either --phi-min/--phi-max or --parameter-min/--parameter-max, not both.")
    if args.num_parameters <= 0:
        raise ValueError("--num-parameters must be positive")

    if using_phi:
        if args.phi_min is None or args.phi_max is None:
            raise ValueError("Pass both --phi-min and --phi-max.")
        return raw_from_phi(regime, np.linspace(args.phi_min, args.phi_max, args.num_parameters))

    low = min(regime.start, regime.end) if args.parameter_min is None else args.parameter_min
    high = max(regime.start, regime.end) if args.parameter_max is None else args.parameter_max
    return np.linspace(low, high, args.num_parameters)


@torch.no_grad()
def forecast_model(
    model: Any,
    forecaster: Any,
    trajectory: np.ndarray,
    phi_value: float,
    context_steps: int,
    forecast_steps: int,
    device: str,
) -> np.ndarray:
    context = torch.tensor(trajectory[:context_steps, None, :], dtype=torch.float32, device=device)
    kwargs: dict[str, torch.Tensor] = {}
    if getattr(model, "phi_dim", 0) > 0:
        kwargs["phi_future"] = torch.full(
            (forecast_steps, 1, 1),
            float(phi_value),
            dtype=torch.float32,
            device=device,
        )
    pred = forecaster.forecast(context, forecast_steps, **kwargs)
    return pred.detach().cpu().numpy().astype(np.float32)[:, 0, :]


def collect_component_points(
    parameter_values: np.ndarray,
    truth_peaks: list[np.ndarray],
    model_peaks: list[list[np.ndarray]],
) -> tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    truth_x = []
    truth_y = []
    for raw, peaks in zip(parameter_values, truth_peaks):
        truth_x.append(np.full(peaks.shape, raw))
        truth_y.append(peaks)
    truth_points = (
        np.concatenate(truth_x) if truth_x else np.empty(0),
        np.concatenate(truth_y) if truth_y else np.empty(0),
    )

    all_model_points = []
    for peaks_by_parameter in model_peaks:
        x_values = []
        y_values = []
        for raw, peaks in zip(parameter_values, peaks_by_parameter):
            x_values.append(np.full(peaks.shape, raw))
            y_values.append(peaks)
        all_model_points.append(
            (
                np.concatenate(x_values) if x_values else np.empty(0),
                np.concatenate(y_values) if y_values else np.empty(0),
            )
        )
    return truth_points, all_model_points


def collect_points_by_component(
    parameter_values: np.ndarray,
    truth_peaks: dict[str, list[np.ndarray]],
    model_peaks: dict[str, list[list[np.ndarray]]],
) -> dict[str, tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]]:
    return {
        component: collect_component_points(
            parameter_values,
            truth_peaks[component],
            model_peaks[component],
        )
        for component in COMPONENTS
    }


def cache_metadata(args: argparse.Namespace, regime: Regime) -> dict[str, Any]:
    truth_steps = args.forecast_steps if args.truth_steps is None else args.truth_steps
    return {
        "regime": regime.name,
        "model_runs": [str(path) for path in args.model_runs],
        "num_parameters": args.num_parameters,
        "parameter_min": args.parameter_min,
        "parameter_max": args.parameter_max,
        "phi_min": args.phi_min,
        "phi_max": args.phi_max,
        "dt": args.dt,
        "context_steps": args.context_steps,
        "forecast_steps": args.forecast_steps,
        "truth_steps": truth_steps,
        "burn_in_steps": args.burn_in_steps,
        "rollouts": args.rollouts,
        "seed": args.seed,
        "rtol": args.rtol,
        "atol": args.atol,
        "max_step_multiplier": args.max_step_multiplier,
    }


def cache_matches(path: Path, expected_metadata: dict[str, Any]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_metadata = json.loads(str(data["metadata"].item()))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return cached_metadata == expected_metadata


def save_points_cache(
    path: Path,
    metadata: dict[str, Any],
    parameter_values: np.ndarray,
    points_by_component: dict[str, tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]],
) -> None:
    payload: dict[str, Any] = {
        "metadata": np.array(json.dumps(metadata), dtype=np.str_),
        "parameter_values": parameter_values.astype(np.float64),
    }
    for component, (truth_points, model_points) in points_by_component.items():
        payload[f"{component}_truth_x"] = truth_points[0].astype(np.float64)
        payload[f"{component}_truth_y"] = truth_points[1].astype(np.float64)
        for model_idx, points in enumerate(model_points):
            payload[f"{component}_model_{model_idx}_x"] = points[0].astype(np.float64)
            payload[f"{component}_model_{model_idx}_y"] = points[1].astype(np.float64)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_points_cache(
    path: Path,
    n_models: int,
) -> tuple[np.ndarray, dict[str, tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]]]:
    with np.load(path, allow_pickle=False) as data:
        parameter_values = data["parameter_values"]
        points_by_component = {}
        for component in COMPONENTS:
            truth_points = (data[f"{component}_truth_x"], data[f"{component}_truth_y"])
            model_points = [
                (data[f"{component}_model_{model_idx}_x"], data[f"{component}_model_{model_idx}_y"])
                for model_idx in range(n_models)
            ]
            points_by_component[component] = (truth_points, model_points)
    return parameter_values, points_by_component


def output_for_component(output: Path, component: str, n_components: int) -> Path:
    if n_components == 1:
        return output
    return output.with_name(f"{output.stem}_{component}{output.suffix}")


def safe_filename_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    return safe.strip("_") or "plot"


def individual_output_path(output: Path, component: str, label: str, n_components: int) -> Path:
    suffix = f"_{component}" if n_components > 1 else ""
    return output.with_name(f"{output.stem}{suffix}_{safe_filename_label(label)}{output.suffix}")


def style_bifurcation_axis(
    ax,
    regime: Regime,
    parameter_values: np.ndarray,
    component: str,
    *,
    show_ylabel: bool,
) -> None:
    ax.set_xlim(float(np.min(parameter_values)), float(np.max(parameter_values)))
    ax.set_xlabel(regime.parameter, fontsize=12)
    if show_ylabel:
        ax.set_ylabel(f"local maxima of {component}(t), standardized", fontsize=12)
    ax.tick_params(labelsize=11)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def plot_single(
    output: Path,
    regime: Regime,
    parameter_values: np.ndarray,
    points: tuple[np.ndarray, np.ndarray],
    label: str,
    component: str,
    point_size: float,
    alpha: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.0), dpi=dpi)
    ax.scatter(points[0], points[1], s=point_size, c="black", alpha=alpha, linewidths=0)
    ax.set_title(f"{label} - {regime.family} {regime.name}", fontsize=14, fontweight="semibold")
    style_bifurcation_axis(ax, regime, parameter_values, component, show_ylabel=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_combined(
    output: Path,
    regime: Regime,
    parameter_values: np.ndarray,
    truth_points: tuple[np.ndarray, np.ndarray],
    model_points: list[tuple[np.ndarray, np.ndarray]],
    model_labels: list[str],
    component: str,
    point_size: float,
    alpha: float,
    dpi: int,
) -> None:
    n_panels = 1 + len(model_points)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(16.0, max(4.0, 3.5 * n_panels)),
        dpi=dpi,
        sharex=True,
        sharey=True,
        squeeze=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()
    labels = ["Ground Truth", *model_labels]
    points = [truth_points, *model_points]

    for ax, label, point_pair in zip(axes_flat, labels, points):
        ax.scatter(point_pair[0], point_pair[1], s=point_size, c="black", alpha=alpha, linewidths=0)
        ax.set_title(label, fontsize=13)
        style_bifurcation_axis(ax, regime, parameter_values, component, show_ylabel=True)

    fig.suptitle(
        f"{regime.family}: {regime.name} bifurcation comparison",
        fontsize=14,
        fontweight="semibold",
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def write_plots(
    output: Path,
    regime: Regime,
    parameter_values: np.ndarray,
    points_by_component: dict[str, tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]],
    labels: list[str],
    components: list[str],
    point_size: float,
    alpha: float,
    dpi: int,
) -> list[Path]:
    written = []
    n_components = len(components)
    for component in components:
        truth_points, model_points = points_by_component[component]
        combined_output = output_for_component(output, component, n_components)
        plot_combined(
            output=combined_output,
            regime=regime,
            parameter_values=parameter_values,
            truth_points=truth_points,
            model_points=model_points,
            model_labels=labels,
            component=component,
            point_size=point_size,
            alpha=alpha,
            dpi=dpi,
        )
        written.append(combined_output)

        all_labels = ["Ground Truth", *labels]
        all_points = [truth_points, *model_points]
        for label, points in zip(all_labels, all_points):
            single_output = individual_output_path(output, component, label, n_components)
            plot_single(
                output=single_output,
                regime=regime,
                parameter_values=parameter_values,
                points=points,
                label=label,
                component=component,
                point_size=point_size,
                alpha=alpha,
                dpi=dpi,
            )
            written.append(single_output)
    return written


def list_regimes_and_exit() -> None:
    for regime in default_regimes():
        print(
            f"{regime.name}: family={regime.family}, parameter={regime.parameter}, "
            f"default_range=[{regime.start:g}, {regime.end:g}], boundary={regime.boundary:g}"
        )


def main() -> None:
    args = parse_args()
    if args.list_regimes:
        list_regimes_and_exit()
        return
    if args.context_steps <= 0 or args.forecast_steps <= 2:
        raise ValueError("--context-steps must be positive and --forecast-steps must be > 2")
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    if args.model_labels is not None and len(args.model_labels) != len(args.model_runs):
        raise ValueError("--model-labels must have the same length as --model-runs")

    regime = regime_by_name(args.regime)
    truth_steps = args.forecast_steps if args.truth_steps is None else args.truth_steps
    if truth_steps <= 2:
        raise ValueError("--truth-steps must be > 2")
    n_steps = args.context_steps + max(args.forecast_steps, truth_steps)
    components = list(dict.fromkeys(args.components))
    parameter_values = parameter_values_from_args(args, regime)
    labels = args.model_labels or [path.name for path in args.model_runs]
    cache_path = args.cache if args.cache is not None else args.output.with_suffix(".npz")
    expected_metadata = cache_metadata(args, regime)

    if cache_path.exists() and not args.force_recompute and cache_matches(cache_path, expected_metadata):
        print(f"Using cached bifurcation points: {cache_path.resolve()}")
        parameter_values, points_by_component = load_points_cache(cache_path, len(args.model_runs))
        written = write_plots(
            output=args.output,
            regime=regime,
            parameter_values=parameter_values,
            points_by_component=points_by_component,
            labels=labels,
            components=components,
            point_size=args.point_size,
            alpha=args.alpha,
            dpi=args.dpi,
        )
        for path in written:
            print(f"Saved plot: {path.resolve()}")
        return

    if cache_path.exists() and not args.force_recompute:
        print(f"Ignoring incompatible cache: {cache_path.resolve()}")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use a CUDA-enabled PyTorch environment or pass --device cpu only for debugging."
        )

    device = configure_runtime(args)
    set_repo_root(args.repo_root)
    models = []
    for run_dir in args.model_runs:
        model, forecaster, config, arch, checkpoint_path = load_model(run_dir, device)
        print(
            f"Loaded {run_dir} from {checkpoint_path.name}; "
            f"phi_dim={arch['phi_dim']}, experts={arch['Experts']}"
        )
        models.append((model, forecaster, config, arch, checkpoint_path))

    print(
        f"Sweeping {len(parameter_values)} values for {regime.name}: "
        f"{regime.parameter}=[{parameter_values[0]:.6g}, {parameter_values[-1]:.6g}]"
    )
    rng = np.random.default_rng(args.seed)
    truth_peaks: dict[str, list[np.ndarray]] = {
        component: [] for component in COMPONENTS
    }
    model_peaks: dict[str, list[list[np.ndarray]]] = {
        component: [[] for _ in models] for component in COMPONENTS
    }

    for raw_parameter in tqdm(parameter_values, desc="Bifurcation sweep"):
        trajectory = simulate_standardized_constant_parameter(
            regime=regime,
            raw_parameter=float(raw_parameter),
            n_steps=n_steps,
            dt=args.dt,
            rng=rng,
            rtol=args.rtol,
            atol=args.atol,
            max_step_multiplier=args.max_step_multiplier,
            burn_in_steps=args.burn_in_steps,
        )
        truth_slice = trajectory[args.context_steps : args.context_steps + truth_steps]
        for component, component_idx in COMPONENTS.items():
            truth_peaks[component].append(local_maxima(truth_slice[:, component_idx]))

        phi_value = normalized_phi(regime, float(raw_parameter))
        for model_idx, (model, forecaster, *_rest) in enumerate(models):
            rollout_peaks = {component: [] for component in COMPONENTS}
            for _ in range(args.rollouts):
                pred = forecast_model(
                    model=model,
                    forecaster=forecaster,
                    trajectory=trajectory,
                    phi_value=phi_value,
                    context_steps=args.context_steps,
                    forecast_steps=args.forecast_steps,
                    device=device,
                )
                for component, component_idx in COMPONENTS.items():
                    rollout_peaks[component].append(local_maxima(pred[:, component_idx]))
            for component in COMPONENTS:
                peaks = rollout_peaks[component]
                model_peaks[component][model_idx].append(
                    np.concatenate(peaks) if peaks else np.empty(0)
                )

    points_by_component = collect_points_by_component(parameter_values, truth_peaks, model_peaks)
    save_points_cache(cache_path, expected_metadata, parameter_values, points_by_component)
    print(f"Saved cache: {cache_path.resolve()}")

    written = write_plots(
        output=args.output,
        regime=regime,
        parameter_values=parameter_values,
        points_by_component=points_by_component,
        labels=labels,
        components=components,
        point_size=args.point_size,
        alpha=args.alpha,
        dpi=args.dpi,
    )

    for path in written:
        print(f"Saved plot: {path.resolve()}")
    for component in components:
        truth_points, model_points = points_by_component[component]
        print(f"{component}: Ground-truth maxima: {len(truth_points[1])}")
        for label, points in zip(labels, model_points):
            print(f"{component}: {label} maxima: {len(points[1])}")


if __name__ == "__main__":
    main()
