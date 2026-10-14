r"""Compare trained DynaMix models with a true bifurcation diagram.

For every value in one shared parameter sweep, the script simulates the chosen
regime with a constant parameter, standardizes that trajectory in the same style
as the training-data generator, forecasts from the initial context with each
model, caches local maxima for all three components, and plots selected
components.

Faint vertical lines mark the constant parameter values represented in each
model's training dataset. Dataset paths are followed from the run metadata;
use --training-data-dirs to override moved datasets, or --no-training-markers
to disable the lines. Marker discovery also runs when replotting cached points.
Use --plot-only to require an existing cache and never recompute the sweep.

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
import inspect
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        default=None,
        help="Path to the DynaMix repository whose src directory should be used.",
    )
    parser.add_argument(
        "--model-type",
        choices=("auto", "dynamix", "alrnn", "pretrained"),
        default="auto",
        help=(
            "Model run format. 'auto' treats runs with model.pt and metrics.json "
            "as single ALRNN runs, otherwise DynaMix runs. Use 'pretrained' with "
            "--pretrained-model-names for original DynaMix HF-style pretrained models."
        ),
    )
    parser.add_argument(
        "--model-runs",
        nargs="+",
        type=Path,
        default=[],
        help="One or more trained run directories containing config.json/checkpoints.",
    )
    parser.add_argument(
        "--pretrained-repo-root",
        type=Path,
        default=None,
        help="Original DynaMix repo root used for load_hf_model pretrained models.",
    )
    parser.add_argument(
        "--pretrained-model-names",
        nargs="+",
        default=[],
        help="One or more pretrained model names, e.g. dynamix-3d-alrnn-v1.0.",
    )
    parser.add_argument(
        "--pretrained-forecast-preprocessing-method",
        default=None,
        help="Optional preprocessing_method passed to the pretrained forecaster if supported.",
    )
    parser.add_argument(
        "--pretrained-standardize",
        choices=("none", "true", "false"),
        default="none",
        help="Optional standardize argument passed to the pretrained forecaster if supported.",
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
        default=15000,
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
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--output", type=Path, default=Path("model_bifurcation_compare.png")
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help=(
            "Cache file for computed bifurcation points. Defaults to the output "
            "path with .npz suffix. Existing compatible cache files are reused."
        ),
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--force-recompute",
        action="store_true",
        help="Ignore an existing cache and recompute the sweep.",
    )
    cache_mode.add_argument(
        "--plot-only",
        action="store_true",
        help=(
            "Only redraw an existing cache; never simulate, load model weights, or write "
            "the cache. Requires matching regime and model order, but ignores current "
            "simulation settings such as --num-parameters."
        ),
    )
    parser.add_argument(
        "--training-data-dirs",
        nargs="+",
        type=Path,
        help=(
            "Override training dataset locations, one directory per --model-runs entry. "
            "Otherwise follow data_path/data_dir in config.json or metrics.json. "
            "Use this when datasets have moved since training."
        ),
    )
    parser.add_argument(
        "--no-training-markers",
        action="store_true",
        help="Disable the faint vertical lines marking training parameter values.",
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
    scale = max(
        abs(regime.start - regime.boundary), abs(regime.end - regime.boundary), 1e-8
    )
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


def find_training_data_dir(
    reference: str, run_dir: Path, repo_root: Path | None
) -> Path:
    # Training paths are normally relative to the repository, not the run folder.
    path = Path(reference.replace("\\", "/")).expanduser()
    if path.suffix.lower() in (".npy", ".json"):
        path = path.parent
    bases = ([repo_root] if repo_root is not None else []) + [
        run_dir, *run_dir.resolve().parents, Path.cwd(), Path(__file__).resolve().parent
    ]
    candidates = [path] if path.is_absolute() else [base / path for base in bases]
    for candidate in candidates:
        for directory in (candidate, candidate / "windows"):
            if (directory / "metadata.json").is_file():
                return directory.resolve()
    raise FileNotFoundError(
        f"Cannot locate training metadata for {reference!r}. "
        "Pass --training-data-dirs with the current dataset locations."
    )


def training_parameter_values(
    data_dir: Path,
    metadata: dict[str, Any],
    regime: Regime,
    settings: dict[str, Any],
    model_type: str,
) -> np.ndarray:
    """Resolve actual constant training values, applying model-side selection."""
    regimes = metadata.get("regimes", [])
    matches = [item for item in regimes if item.get("name") == regime.name]
    if not matches:
        raise ValueError(f"Training dataset has no metadata for regime {regime.name!r}")
    dataset_regime = matches[0]
    boundary = float(dataset_regime["boundary"])
    scale = max(
        abs(float(dataset_regime["start"]) - boundary),
        abs(float(dataset_regime["end"]) - boundary),
        1e-8,
    )
    parameter_filter = metadata.get("parameter_filter", {})
    values = None
    per_series = False
    names_path = None
    # Prefer the actual training windows. The arrays can be memory-mapped so no
    # trajectory/state data need to be loaded, even when replotting a cache.
    array_names = (
        ("phi_raw.npy", "trajectory_phi_raw_constants.npy", "trajectory_phi_raw.npy")
        if model_type == "dynamix"
        else ("trajectory_phi_raw_constants.npy", "trajectory_phi_raw.npy")
    )
    for name in array_names:
        path = data_dir / name
        if not path.is_file():
            continue
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if name.endswith("constants.npy"):
            values = np.asarray(array, dtype=np.float64).reshape(-1)
        else:
            if array.ndim != 3 or array.shape[2] != 1:
                raise ValueError(f"Unexpected parameter array shape in {path}: {array.shape}")
            # Window arrays are (time, windows, 1); trajectories are (series, time, 1).
            by_series = array[:, :, 0].T if name == "phi_raw.npy" else array[:, :, 0]
            values = np.asarray(by_series[:, 0], dtype=np.float64)
            for start in range(0, len(values), 1024):
                block = by_series[start : start + 1024]
                if not np.all(np.isclose(block, values[start : start + 1024, None],
                                         atol=1e-6, rtol=0.0)):
                    raise ValueError("Training parameters vary over time; discrete markers are unavailable")
        per_series = True
        names_path = data_dir / (
            "window_regime_names.npy" if name == "phi_raw.npy" else "trajectory_regime_names.npy"
        )
        break

    if values is None:
        if len(regimes) != 1:
            raise ValueError("Multiple training regimes require parameter arrays with regime labels")
        raw_values = parameter_filter.get("selected_raw_parameter_values")
        if raw_values is not None:
            values = np.asarray(raw_values, dtype=np.float64)
        else:
            phi_values = parameter_filter.get("selected_phi_values")
            if phi_values is None:
                phi_values = metadata.get("requested_phi_values")
            if phi_values is None:
                raise ValueError("No constant training parameter values found in dataset")
            values = boundary + np.asarray(phi_values, dtype=np.float64) * scale

    keep = np.ones(values.shape, dtype=bool)
    if per_series and names_path is not None and names_path.is_file():
        names = np.load(names_path, allow_pickle=False).astype(str).reshape(-1)
        if names.shape != values.shape:
            raise ValueError("Training regime labels do not match the parameter array")
        keep &= names == regime.name
    elif len(regimes) != 1:
        raise ValueError("Cannot associate training parameter values with the plotted regime")

    if model_type == "alrnn" and settings.get("multi_trajectory") is False:
        if not per_series:
            raise ValueError("Single-trajectory ALRNN training requires per-trajectory parameter arrays")
        selected = np.zeros(values.shape, dtype=bool)
        trajectory_id = int(settings.get("trajectory_id", 0))
        if not 0 <= trajectory_id < len(values):
            raise ValueError("ALRNN trajectory_id is outside the training dataset")
        selected[trajectory_id] = True
        keep &= selected
    else:
        requested_phi = settings.get("train_phi_values")
        requested_raw = settings.get("train_raw_parameter_values")
        if requested_phi is not None:
            phi = (values - boundary) / scale
            keep &= np.any(np.isclose(phi[:, None], requested_phi, atol=1e-5), axis=1)
        if requested_raw is not None:
            keep &= np.any(np.isclose(values[:, None], requested_raw, atol=1e-5), axis=1)

    values = values[keep]
    if not np.all(np.isfinite(values)):
        raise ValueError("Training parameter values contain non-finite entries")
    if not len(values):
        raise ValueError("No training parameters remain for this regime and model selection")
    # Collapse float32 rounding duplicates without merging distinct sampled values.
    unique = np.unique(values)
    return unique[np.r_[True, ~np.isclose(unique[1:], unique[:-1], atol=1e-7, rtol=0.0)]]


def resolve_training_parameters(args: argparse.Namespace, regime: Regime) -> list[np.ndarray]:
    markers = [np.empty(0) for _ in args.model_runs + args.pretrained_model_names]
    if args.no_training_markers:
        return markers
    for index, run_dir in enumerate(args.model_runs):
        try:
            model_type = detect_model_type(run_dir, args.model_type)
            filename = "metrics.json" if model_type == "alrnn" else "config.json"
            run_metadata = json.loads((run_dir / filename).read_text(encoding="utf-8-sig"))
            settings = run_metadata.get("args", run_metadata)
            if args.training_data_dirs is not None:
                data_dir = args.training_data_dirs[index].expanduser().resolve()
            else:
                reference = settings.get("data_path") or settings.get("data_dir")
                if not reference:
                    raise ValueError(f"No training dataset reference in {filename}")
                data_dir = find_training_data_dir(reference, run_dir, args.repo_root)
            metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8-sig"))
            markers[index] = training_parameter_values(data_dir, metadata, regime, settings, model_type)
            print(
                f"Training parameters for {run_dir.name}: "
                f"{np.array2string(markers[index], precision=6)}; dataset={data_dir}"
            )
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            print(f"Warning: no training markers for {run_dir.name}: {exc}")
    return markers


def add_training_markers(ax, values: np.ndarray, parameter_values: np.ndarray) -> None:
    low, high = float(np.min(parameter_values)), float(np.max(parameter_values))
    visible = values[(values >= low - 1e-7) & (values <= high + 1e-7)]
    for index, value in enumerate(visible):
        ax.axvline(
            float(np.clip(value, low, high)), color="tab:blue", alpha=0.4,
            linewidth=1.1, zorder=0,
            label="Training parameters" if index == 0 else "_nolegend_",
        )
    if len(visible):
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)


def local_maxima(values: np.ndarray) -> np.ndarray:
    return values[1:-1][(values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])]


class AL_RNN(nn.Module):
    """Single ALRNN model definition used by ALRNN-DSR/alrnn_python runs."""

    def __init__(self, M: int, P: int, N: int, phi_dim: int = 0):
        super().__init__()
        self.M = M
        self.P = P
        self.N = N
        self.phi_dim = phi_dim
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B = self.init_uniform((self.N, self.M))
        if self.phi_dim > 0:
            self.C = nn.Parameter(torch.zeros(self.M, self.phi_dim))
        else:
            self.register_parameter("C", None)

    def forward(
        self, z: torch.Tensor, phi_t: torch.Tensor | None = None
    ) -> torch.Tensor:
        z_unactivated = torch.clone(z)
        if self.P > 0:
            z[:, -self.P :] = F.relu(z[:, -self.P :])
        out = self.A * z_unactivated + z @ self.W.t() + self.h
        if self.C is not None:
            if phi_t is None:
                raise ValueError("phi_t must be provided when phi_dim > 0")
            out = out + phi_t @ self.C.t()
        return out

    def initialize_AWh_random(self) -> tuple[nn.Parameter, nn.Parameter, nn.Parameter]:
        A = nn.Parameter(torch.diagonal(self.normalized_positive_definite(self.M), 0))
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.zeros(self.M))
        return A, W, h

    def normalized_positive_definite(self, M: int) -> torch.Tensor:
        R = np.random.randn(M, M).astype(np.float32)
        K = np.matmul(R.T, R) / M + np.eye(M)
        lambda_max = np.max(np.abs(np.linalg.eigvals(K)))
        return torch.tensor(K / lambda_max).float()

    def init_uniform(self, shape: tuple[int, int]) -> nn.Parameter:
        tensor = torch.empty(*shape)
        radius = 1 / math.sqrt(shape[0])
        torch.nn.init.uniform_(tensor, -radius, radius)
        return nn.Parameter(tensor, requires_grad=True)


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
        raise ValueError(
            "Use either --phi-min/--phi-max or --parameter-min/--parameter-max, not both."
        )
    if args.num_parameters <= 0:
        raise ValueError("--num-parameters must be positive")

    if using_phi:
        if args.phi_min is None or args.phi_max is None:
            raise ValueError("Pass both --phi-min and --phi-max.")
        return raw_from_phi(
            regime, np.linspace(args.phi_min, args.phi_max, args.num_parameters)
        )

    low = (
        min(regime.start, regime.end)
        if args.parameter_min is None
        else args.parameter_min
    )
    high = (
        max(regime.start, regime.end)
        if args.parameter_max is None
        else args.parameter_max
    )
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
    context = torch.tensor(
        trajectory[:context_steps, None, :], dtype=torch.float32, device=device
    )
    kwargs: dict[str, torch.Tensor] = {}
    if getattr(model, "phi_dim", 0) > 0:
        forecast_params = inspect.signature(forecaster.forecast).parameters
        if "context_phi" in forecast_params:
            kwargs["context_phi"] = torch.full(
                (context_steps, 1, 1),
                float(phi_value),
                dtype=torch.float32,
                device=device,
            )
        kwargs["phi_future"] = torch.full(
            (forecast_steps, 1, 1),
            float(phi_value),
            dtype=torch.float32,
            device=device,
        )
    pred = forecaster.forecast(context, forecast_steps, **kwargs)
    return pred.detach().cpu().numpy().astype(np.float32)[:, 0, :]


def purge_dynamix_modules() -> None:
    for name in list(sys.modules):
        if name == "dynamix" or name.startswith("dynamix."):
            del sys.modules[name]


def put_repo_src_first(repo_root: Path) -> None:
    src = repo_root.resolve() / "src"
    if not src.exists():
        raise FileNotFoundError(f"Expected DynaMix src directory at {src}")
    src_str = str(src)
    while src_str in sys.path:
        sys.path.remove(src_str)
    sys.path.insert(0, src_str)


def load_pretrained_model(
    repo_root: Path,
    model_name: str,
    device: str,
) -> tuple[Any, Any]:
    purge_dynamix_modules()
    put_repo_src_first(repo_root)
    from dynamix.model.forecaster import DynaMixForecaster
    from dynamix.utilities.utilities import load_hf_model

    model = load_hf_model(model_name).to(device)
    model.eval()
    return model, DynaMixForecaster(model)


def optional_bool(value: str) -> bool | None:
    if value == "none":
        return None
    return value == "true"


@torch.no_grad()
def forecast_pretrained_model(
    forecaster: Any,
    trajectory: np.ndarray,
    context_steps: int,
    forecast_steps: int,
    device: str,
    preprocessing_method: str | None,
    standardize: bool | None,
) -> np.ndarray:
    context = torch.tensor(trajectory[:context_steps, None, :], dtype=torch.float32, device=device)
    kwargs: dict[str, Any] = {}
    forecast_params = inspect.signature(forecaster.forecast).parameters
    if preprocessing_method is not None and "preprocessing_method" in forecast_params:
        kwargs["preprocessing_method"] = preprocessing_method
    if standardize is not None and "standardize" in forecast_params:
        kwargs["standardize"] = standardize
    pred = forecaster.forecast(context, forecast_steps, **kwargs)
    return pred.detach().cpu().numpy().astype(np.float32)[:, 0, :]


def detect_model_type(run_dir: Path, requested_type: str) -> str:
    if requested_type != "auto":
        return requested_type
    if (run_dir / "model.pt").exists() and (run_dir / "metrics.json").exists():
        return "alrnn"
    return "dynamix"


def load_alrnn_model(run_dir: Path, device: str) -> tuple[AL_RNN, dict[str, Any], Path]:
    metrics_path = run_dir / "metrics.json"
    model_path = run_dir / "model.pt"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected ALRNN metrics.json at {metrics_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Expected ALRNN model.pt at {model_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    run_args = metrics.get("args", {})
    latent_dim_value = run_args.get("latent_dim", metrics.get("latent_dim"))
    pwl_units_value = run_args.get("pwl_units", metrics.get("pwl_units"))
    if latent_dim_value is None or pwl_units_value is None:
        raise ValueError(
            f"Could not infer ALRNN architecture from {metrics_path}. "
            "Expected args.latent_dim and args.pwl_units."
        )
    latent_dim = int(latent_dim_value)
    pwl_units = int(pwl_units_value)
    n_dims = int(metrics.get("N", 3))
    phi_dim = int(metrics.get("phi_dim", 1 if run_args.get("use_phi") else 0))

    model = AL_RNN(M=latent_dim, P=pwl_units, N=n_dims, phi_dim=phi_dim).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, metrics, model_path


@torch.no_grad()
def forecast_alrnn_model(
    model: AL_RNN,
    trajectory: np.ndarray,
    phi_value: float,
    context_steps: int,
    forecast_steps: int,
    device: str,
) -> np.ndarray:
    # The single ALRNN has no encoder/context assimilation. Match its training
    # evaluation: seed the latent state from one observed x and then free-run.
    seed_idx = max(context_steps - 1, 0)
    x = torch.tensor(
        trajectory[seed_idx : seed_idx + 1, :], dtype=torch.float32, device=device
    )
    z = x @ model.B
    z[:, 0 : model.N] = x
    phi = None
    if model.phi_dim > 0:
        phi = torch.full(
            (1, model.phi_dim), float(phi_value), dtype=torch.float32, device=device
        )

    out = torch.empty((forecast_steps, model.N), dtype=torch.float32, device=device)
    for t in range(forecast_steps):
        z = model(z, phi)
        out[t] = z[0, 0 : model.N]
    return out.detach().cpu().numpy().astype(np.float32)


def forecast_loaded_model(
    loaded: dict[str, Any],
    trajectory: np.ndarray,
    phi_value: float,
    context_steps: int,
    forecast_steps: int,
    device: str,
) -> np.ndarray:
    if loaded["type"] == "alrnn":
        return forecast_alrnn_model(
            model=loaded["model"],
            trajectory=trajectory,
            phi_value=phi_value,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            device=device,
        )
    if loaded["type"] == "pretrained":
        return forecast_pretrained_model(
            forecaster=loaded["forecaster"],
            trajectory=trajectory,
            context_steps=context_steps,
            forecast_steps=forecast_steps,
            device=device,
            preprocessing_method=loaded["preprocessing_method"],
            standardize=loaded["standardize"],
        )
    return forecast_model(
        model=loaded["model"],
        forecaster=loaded["forecaster"],
        trajectory=trajectory,
        phi_value=phi_value,
        context_steps=context_steps,
        forecast_steps=forecast_steps,
        device=device,
    )


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
) -> dict[
    str, tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]
]:
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
        "model_type": args.model_type,
        "model_runs": [str(path) for path in args.model_runs],
        "pretrained_repo_root": (
            str(args.pretrained_repo_root) if args.pretrained_repo_root is not None else None
        ),
        "pretrained_model_names": list(args.pretrained_model_names),
        "pretrained_forecast_preprocessing_method": args.pretrained_forecast_preprocessing_method,
        "pretrained_standardize": args.pretrained_standardize,
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


def validate_plot_only_cache(path: Path, expected_metadata: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"--plot-only requires an existing cache: {path}")
    with np.load(path, allow_pickle=False) as data:
        cached = json.loads(str(data["metadata"].item()))
    if cached.get("regime") != expected_metadata["regime"]:
        raise ValueError(
            f"Cache regime is {cached.get('regime')!r}; requested "
            f"{expected_metadata['regime']!r}. No plots were written."
        )
    # Accept absolute/relative spellings of the same run without accepting a
    # different model or model order, which would mislabel the cached panels.
    def resolved_runs(metadata: dict[str, Any]) -> list[Path]:
        return [Path(value).expanduser().resolve() for value in metadata.get("model_runs", [])]

    if (
        resolved_runs(cached) != resolved_runs(expected_metadata)
        or cached.get("pretrained_model_names", []) != expected_metadata["pretrained_model_names"]
    ):
        raise ValueError(
            "Cache models differ from the requested models or their order. "
            "Pass --model-runs/--pretrained-model-names in the original cache order. "
            "No plots were written."
        )


def save_points_cache(
    path: Path,
    metadata: dict[str, Any],
    parameter_values: np.ndarray,
    points_by_component: dict[
        str, tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]
    ],
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
    components: list[str] | None = None,
) -> tuple[
    np.ndarray,
    dict[
        str, tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]
    ],
]:
    with np.load(path, allow_pickle=False) as data:
        parameter_values = data["parameter_values"]
        points_by_component = {}
        for component in COMPONENTS if components is None else components:
            truth_points = (data[f"{component}_truth_x"], data[f"{component}_truth_y"])
            model_points = [
                (
                    data[f"{component}_model_{model_idx}_x"],
                    data[f"{component}_model_{model_idx}_y"],
                )
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


def individual_output_path(
    output: Path, component: str, label: str, n_components: int
) -> Path:
    suffix = f"_{component}" if n_components > 1 else ""
    return output.with_name(
        f"{output.stem}{suffix}_{safe_filename_label(label)}{output.suffix}"
    )


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
    training_parameters: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.0), dpi=dpi)
    ax.scatter(points[0], points[1], s=point_size, c="black", alpha=alpha, linewidths=0)
    ax.set_title(
        f"{label} - {regime.family} {regime.name}", fontsize=14, fontweight="semibold"
    )
    style_bifurcation_axis(ax, regime, parameter_values, component, show_ylabel=True)
    if training_parameters is not None:
        add_training_markers(ax, training_parameters, parameter_values)
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
    training_parameters: list[np.ndarray] | None = None,
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

    for panel_index, (ax, label, point_pair) in enumerate(zip(axes_flat, labels, points)):
        ax.scatter(
            point_pair[0],
            point_pair[1],
            s=point_size,
            c="black",
            alpha=alpha,
            linewidths=0,
        )
        ax.set_title(label, fontsize=13)
        style_bifurcation_axis(
            ax, regime, parameter_values, component, show_ylabel=True
        )
        if panel_index > 0 and training_parameters is not None:
            add_training_markers(ax, training_parameters[panel_index - 1], parameter_values)

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
    points_by_component: dict[
        str, tuple[tuple[np.ndarray, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]
    ],
    labels: list[str],
    components: list[str],
    point_size: float,
    alpha: float,
    dpi: int,
    training_parameters: list[np.ndarray] | None = None,
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
            training_parameters=training_parameters,
        )
        written.append(combined_output)

        all_labels = ["Ground Truth", *labels]
        all_points = [truth_points, *model_points]
        for panel_index, (label, points) in enumerate(zip(all_labels, all_points)):
            single_output = individual_output_path(
                output, component, label, n_components
            )
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
                training_parameters=(
                    training_parameters[panel_index - 1]
                    if panel_index > 0 and training_parameters is not None else None
                ),
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
        raise ValueError(
            "--context-steps must be positive and --forecast-steps must be > 2"
        )
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive")
    n_models_requested = len(args.model_runs) + len(args.pretrained_model_names)
    if n_models_requested == 0:
        raise ValueError("Pass at least one --model-runs entry or --pretrained-model-names entry")
    if args.model_type == "pretrained" and args.model_runs:
        raise ValueError("Use --pretrained-model-names, not --model-runs, with --model-type pretrained")
    if args.pretrained_model_names and args.pretrained_repo_root is None:
        raise ValueError("--pretrained-repo-root is required with --pretrained-model-names")
    if args.model_labels is not None and len(args.model_labels) != n_models_requested:
        raise ValueError(
            "--model-labels must have the same length as local runs plus pretrained models"
        )
    if args.training_data_dirs is not None and len(args.training_data_dirs) != len(args.model_runs):
        raise ValueError("--training-data-dirs must have one directory per --model-runs entry")

    regime = regime_by_name(args.regime)
    truth_steps = args.forecast_steps if args.truth_steps is None else args.truth_steps
    if truth_steps <= 2:
        raise ValueError("--truth-steps must be > 2")
    n_steps = args.context_steps + max(args.forecast_steps, truth_steps)
    components = list(dict.fromkeys(args.components))
    parameter_values = parameter_values_from_args(args, regime)
    default_labels = [path.name for path in args.model_runs] + [
        f"pretrained:{name}" for name in args.pretrained_model_names
    ]
    labels = args.model_labels or default_labels
    training_parameters = resolve_training_parameters(args, regime)
    cache_path = (
        args.cache if args.cache is not None else args.output.with_suffix(".npz")
    )
    expected_metadata = cache_metadata(args, regime)

    if args.plot_only:
        validate_plot_only_cache(cache_path, expected_metadata)

    if (
        args.plot_only
        or (
            cache_path.exists()
            and not args.force_recompute
            and cache_matches(cache_path, expected_metadata)
        )
    ):
        print(f"Using cached bifurcation points: {cache_path.resolve()}")
        parameter_values, points_by_component = load_points_cache(
            cache_path, n_models_requested, components
        )
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
            training_parameters=training_parameters,
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
    models = []
    local_model_type = "auto" if args.model_type == "pretrained" else args.model_type
    run_types = [
        detect_model_type(run_dir, local_model_type) for run_dir in args.model_runs
    ]
    if any(run_type == "dynamix" for run_type in run_types):
        if args.repo_root is None:
            raise ValueError("--repo-root is required for DynaMix model runs")
        set_repo_root(args.repo_root)

    for run_dir, run_type in zip(args.model_runs, run_types):
        if run_type == "alrnn":
            model, metrics, model_path = load_alrnn_model(run_dir, device)
            run_args = metrics.get("args", {})
            print(
                f"Loaded ALRNN {run_dir} from {model_path.name}; "
                f"phi_dim={model.phi_dim}, M={model.M}, P={model.P}, "
                f"train_phi_values={run_args.get('train_phi_values')}"
            )
            models.append(
                {
                    "type": "alrnn",
                    "model": model,
                    "metrics": metrics,
                    "checkpoint_path": model_path,
                }
            )
        else:
            model, forecaster, config, arch, checkpoint_path = load_model(
                run_dir, device
            )
            print(
                f"Loaded DynaMix {run_dir} from {checkpoint_path.name}; "
                f"phi_dim={arch['phi_dim']}, experts={arch['Experts']}"
            )
            models.append(
                {
                    "type": "dynamix",
                    "model": model,
                    "forecaster": forecaster,
                    "config": config,
                    "arch": arch,
                    "checkpoint_path": checkpoint_path,
                }
            )

    for model_name in args.pretrained_model_names:
        if args.pretrained_repo_root is None:
            raise ValueError("--pretrained-repo-root is required for pretrained models")
        model, forecaster = load_pretrained_model(
            args.pretrained_repo_root,
            model_name,
            device,
        )
        print(
            f"Loaded pretrained {model_name} from {args.pretrained_repo_root}; "
            f"phi_dim={getattr(model, 'phi_dim', 0)}"
        )
        models.append(
            {
                "type": "pretrained",
                "model": model,
                "forecaster": forecaster,
                "model_name": model_name,
                "repo_root": args.pretrained_repo_root,
                "preprocessing_method": args.pretrained_forecast_preprocessing_method,
                "standardize": optional_bool(args.pretrained_standardize),
            }
        )

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
        for model_idx, loaded_model in enumerate(models):
            rollout_peaks = {component: [] for component in COMPONENTS}
            for _ in range(args.rollouts):
                pred = forecast_loaded_model(
                    loaded=loaded_model,
                    trajectory=trajectory,
                    phi_value=phi_value,
                    context_steps=args.context_steps,
                    forecast_steps=args.forecast_steps,
                    device=device,
                )
                for component, component_idx in COMPONENTS.items():
                    rollout_peaks[component].append(
                        local_maxima(pred[:, component_idx])
                    )
            for component in COMPONENTS:
                peaks = rollout_peaks[component]
                model_peaks[component][model_idx].append(
                    np.concatenate(peaks) if peaks else np.empty(0)
                )

    points_by_component = collect_points_by_component(
        parameter_values, truth_peaks, model_peaks
    )
    save_points_cache(
        cache_path, expected_metadata, parameter_values, points_by_component
    )
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
        training_parameters=training_parameters,
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
