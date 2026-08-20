"""Simulate dynamic 3D regime-change trajectories for DynaMix experiments.

This script intentionally only simulates and stores long trajectories. Training
windows should be sampled afterwards with resample_time_varying_bifurcation_windows.py.

The output layout is compatible with that resampler:

    trajectories.npy              (S, T, 3)
    trajectory_phi.npy            (S, T, 1)
    trajectory_phi_raw.npy        (S, T, 1)
    trajectory_regime_index.npy   (S,)
    trajectory_regime_names.npy   (S,)
    metadata.json

The regimes below are chosen to be more dynamically informative than fixed-point
crossings: periodic-to-periodic, periodic-to-chaotic, chaotic-to-chaotic, and
chaotic-to-periodic transitions. The boundary values are practical numerical
transition markers for window sampling, not exact analytical bifurcation proofs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

Array = np.ndarray
RhsFn = Callable[[float, Array, float], Array]


@dataclass(frozen=True)
class Regime:
    name: str
    family: str
    parameter: str
    boundary: float
    start: float
    end: float
    transition_type: str
    description: str
    rhs_key: str
    params: dict
    init_center: tuple[float, float, float]
    init_scale: tuple[float, float, float]
    observation: str = "identity"


def linear_parameter(t: float, t_end: float, start: float, end: float) -> float:
    if t_end <= 0.0:
        return start
    s = np.clip(t / t_end, 0.0, 1.0)
    return start + (end - start) * s


def constant_parameter_value(regime: Regime, position: str) -> float:
    if position == "start":
        return regime.start
    if position == "boundary":
        return regime.boundary
    if position == "end":
        return regime.end
    if position == "midpoint":
        return 0.5 * (regime.start + regime.end)
    raise ValueError(f"Unknown constant parameter position: {position}")


def parameter_value(
    t: float, t_end: float, regime: Regime, mode: str, constant_position: str
) -> float:
    if mode == "linear":
        return linear_parameter(t, t_end, regime.start, regime.end)
    if mode == "constant":
        return constant_parameter_value(regime, constant_position)
    raise ValueError(f"Unknown parameter mode: {mode}")


def normalize_phi(phi_raw: Array, start: float, end: float, boundary: float) -> Array:
    scale = max(abs(start - boundary), abs(end - boundary), 1e-8)
    return (phi_raw - boundary) / scale


def rossler_rhs(t: float, x: Array, p: float, *, a: float, b: float) -> Array:
    return np.array(
        [
            -x[1] - x[2],
            x[0] + a * x[1],
            b + x[2] * (x[0] - p),
        ],
        dtype=np.float64,
    )


def lorenz_rhs(t: float, x: Array, p: float, *, sigma: float, beta: float) -> Array:
    return np.array(
        [
            sigma * (x[1] - x[0]),
            x[0] * (p - x[2]) - x[1],
            x[0] * x[1] - beta * x[2],
        ],
        dtype=np.float64,
    )


def forced_duffing_rhs(
    t: float,
    x: Array,
    p: float,
    *,
    damping: float,
    alpha: float,
    beta: float,
    omega: float,
) -> Array:
    return np.array(
        [
            x[1],
            -damping * x[1] - alpha * x[0] - beta * x[0] ** 3 + p * np.cos(x[2]),
            omega,
        ],
        dtype=np.float64,
    )


def forced_vdp_rhs(
    t: float,
    x: Array,
    p: float,
    *,
    mu: float,
    omega: float,
) -> Array:
    return np.array(
        [
            x[1],
            mu * (1.0 - x[0] * x[0]) * x[1] - x[0] + p * np.cos(x[2]),
            omega,
        ],
        dtype=np.float64,
    )


def halvorsen_rhs(t: float, x: Array, p: float, *, drive: float) -> Array:
    a = p
    return np.array(
        [
            -a * x[0] - 4.0 * x[1] - 4.0 * x[2] - x[1] * x[1] + drive,
            -a * x[1] - 4.0 * x[2] - 4.0 * x[0] - x[2] * x[2] + drive,
            -a * x[2] - 4.0 * x[0] - 4.0 * x[1] - x[0] * x[0] + drive,
        ],
        dtype=np.float64,
    )


RHS: dict[str, RhsFn] = {
    "rossler": rossler_rhs,
    "lorenz": lorenz_rhs,
    "forced_duffing": forced_duffing_rhs,
    "forced_vdp": forced_vdp_rhs,
    "halvorsen": halvorsen_rhs,
}


def default_regimes() -> list[Regime]:
    sigma = 10.0
    beta = 8.0 / 3.0
    return [
        Regime(
            name="rossler_c_periodic_to_chaotic",
            family="Rossler",
            parameter="c",
            boundary=4.2,
            start=3.5,
            end=5.7,
            transition_type="periodic_to_chaotic",
            description="Rossler route from mostly periodic oscillations toward chaotic spiral dynamics.",
            rhs_key="rossler",
            params={"a": 0.2, "b": 0.2},
            init_center=(0.0, -5.0, 0.5),
            init_scale=(1.0, 1.0, 0.2),
        ),
        Regime(
            name="rossler_c_chaotic_to_periodic",
            family="Rossler",
            parameter="c",
            boundary=4.2,
            start=5.7,
            end=3.5,
            transition_type="chaotic_to_periodic",
            description="Reverse Rossler sweep from chaotic spiral dynamics toward more regular oscillations.",
            rhs_key="rossler",
            params={"a": 0.2, "b": 0.2},
            init_center=(0.0, -5.0, 0.5),
            init_scale=(1.0, 1.0, 0.2),
        ),
        Regime(
            name="duffing_gamma_periodic_to_chaotic",
            family="ForcedDuffing",
            parameter="gamma",
            boundary=0.29,
            start=0.20,
            end=0.38,
            transition_type="periodic_to_chaotic",
            description="Driven double-well Duffing oscillator with increasing forcing amplitude.",
            rhs_key="forced_duffing",
            params={"damping": 0.2, "alpha": -1.0, "beta": 1.0, "omega": 1.2},
            init_center=(0.0, 0.0, 0.0),
            init_scale=(0.6, 0.4, 3.141592653589793),
            observation="phase_sine",
        ),
        Regime(
            name="duffing_gamma_chaotic_to_periodic",
            family="ForcedDuffing",
            parameter="gamma",
            boundary=0.29,
            start=0.38,
            end=0.20,
            transition_type="chaotic_to_periodic",
            description="Reverse driven Duffing sweep from irregular motion back toward regular oscillation.",
            rhs_key="forced_duffing",
            params={"damping": 0.2, "alpha": -1.0, "beta": 1.0, "omega": 1.2},
            init_center=(0.0, 0.0, 0.0),
            init_scale=(0.6, 0.4, 3.141592653589793),
            observation="phase_sine",
        ),
        Regime(
            name="forced_vdp_amp_periodic_to_periodic",
            family="ForcedVanDerPol",
            parameter="forcing_amplitude",
            boundary=0.25,
            start=0.05,
            end=0.45,
            transition_type="periodic_to_periodic",
            description="Forced Van der Pol oscillator with regular oscillations on both sides but changing amplitude and phase locking.",
            rhs_key="forced_vdp",
            params={"mu": 3.0, "omega": 1.7},
            init_center=(2.0, 0.0, 0.0),
            init_scale=(0.4, 0.4, 3.141592653589793),
            observation="phase_sine",
        ),
        Regime(
            name="forced_vdp_amp_periodic_to_mixed",
            family="ForcedVanDerPol",
            parameter="forcing_amplitude",
            boundary=0.75,
            start=0.20,
            end=1.20,
            transition_type="periodic_to_chaotic",
            description="Forced Van der Pol oscillator from regular limit-cycle locking toward mixed irregular response.",
            rhs_key="forced_vdp",
            params={"mu": 5.0, "omega": 2.466},
            init_center=(2.0, 0.0, 0.0),
            init_scale=(0.4, 0.4, 3.141592653589793),
            observation="phase_sine",
        ),
        Regime(
            name="lorenz_rho_chaotic_attractor_change",
            family="Lorenz",
            parameter="rho",
            boundary=60.0,
            start=28.0,
            end=100.0,
            transition_type="chaotic_to_chaotic",
            description="Lorenz sweep through chaotic regimes with changing attractor scale and lobe switching statistics.",
            rhs_key="lorenz",
            params={"sigma": sigma, "beta": beta},
            init_center=(1.0, 1.0, 25.0),
            init_scale=(2.0, 2.0, 3.0),
        ),
        Regime(
            name="lorenz_rho_high_to_classic_chaos",
            family="Lorenz",
            parameter="rho",
            boundary=60.0,
            start=100.0,
            end=28.0,
            transition_type="chaotic_to_chaotic",
            description="Reverse Lorenz chaotic sweep from large-amplitude chaos toward the classical rho=28 attractor.",
            rhs_key="lorenz",
            params={"sigma": sigma, "beta": beta},
            init_center=(1.0, 1.0, 40.0),
            init_scale=(3.0, 3.0, 5.0),
        ),
        Regime(  # eigentlicher standard 1.4 to 2.09
            name="halvorsen_a_chaotic_to_periodic",
            family="Halvorsen",
            parameter="a",
            boundary=1.75,
            start=1.4,
            end=2.09,
            transition_type="chaotic_to_periodic",
            description="Halvorsen-type flow with damping sweep from richer chaotic motion toward simpler bounded motion.",
            rhs_key="halvorsen",
            params={"drive": 0.0},
            init_center=(-4.0, 0.0, 0.0),
            init_scale=(0.8, 0.8, 0.8),
        ),
    ]


def observe_state(states: Array, observation: str) -> Array:
    if observation == "identity":
        return states
    if observation == "phase_sine":
        observed = states.copy()
        observed[:, 2] = np.sin(states[:, 2])
        return observed
    raise ValueError(f"Unknown observation mode: {observation}")


def standardize_trajectories(trajectories: Array) -> tuple[Array, Array, Array]:
    mean = trajectories.mean(axis=1, keepdims=True)
    std = trajectories.std(axis=1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((trajectories - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def add_gaussian_noise(
    trajectories: Array, rng: np.random.Generator, noise_std_fraction: float
) -> Array:
    if noise_std_fraction < 0.0:
        raise ValueError("noise_std_fraction must be non-negative")
    if noise_std_fraction == 0.0:
        return trajectories

    data_std = trajectories.std(axis=1, keepdims=True)
    data_std = np.where(data_std < 1e-6, 1.0, data_std)
    noise = rng.normal(
        loc=0.0,
        scale=noise_std_fraction * data_std,
        size=trajectories.shape,
    )
    return (trajectories + noise).astype(np.float32)


def simulate_regime(
    regime: Regime,
    n_steps: int,
    dt: float,
    n_trajectories: int,
    rng: np.random.Generator,
    rtol: float,
    atol: float,
    max_step_multiplier: float,
    burn_in_steps: int,
    parameter_mode: str,
    constant_parameter: str,
) -> tuple[Array, Array, Array]:
    t_eval = np.arange(n_steps, dtype=np.float64) * dt
    t_end = float(t_eval[-1])
    rhs = RHS[regime.rhs_key]

    phi_raw = np.array(
        [
            parameter_value(t, t_end, regime, parameter_mode, constant_parameter)
            for t in t_eval
        ],
        dtype=np.float32,
    )
    phi = normalize_phi(phi_raw, regime.start, regime.end, regime.boundary).astype(
        np.float32
    )
    trajectories = np.empty((n_trajectories, n_steps, 3), dtype=np.float32)

    def ode_main(t: float, x: Array) -> Array:
        p = parameter_value(t, t_end, regime, parameter_mode, constant_parameter)
        dx = rhs(t, x, p, **regime.params)
        return np.clip(dx, -1e4, 1e4)

    def ode_burn_in(t: float, x: Array) -> Array:
        dx = rhs(t, x, regime.start, **regime.params)
        return np.clip(dx, -1e4, 1e4)

    for k in range(n_trajectories):
        x0 = rng.normal(
            loc=np.array(regime.init_center), scale=np.array(regime.init_scale)
        )
        if regime.observation == "phase_sine":
            x0[2] = rng.uniform(0.0, 2.0 * np.pi)

        if burn_in_steps > 0:
            burn_end = burn_in_steps * dt
            burn = solve_ivp(
                ode_burn_in,
                (0.0, burn_end),
                x0,
                method="DOP853",
                rtol=rtol,
                atol=atol,
                max_step=dt * max_step_multiplier,
            )
            if not burn.success:
                raise RuntimeError(f"Burn-in failed for {regime.name}: {burn.message}")
            x0 = burn.y[:, -1]

        sol = solve_ivp(
            ode_main,
            (0.0, t_end),
            x0,
            t_eval=t_eval,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=dt * max_step_multiplier,
        )
        if not sol.success or sol.y.shape[1] != n_steps:
            raise RuntimeError(f"Integration failed for {regime.name}: {sol.message}")

        traj = observe_state(sol.y.T, regime.observation)
        if not np.all(np.isfinite(traj)):
            raise RuntimeError(f"Non-finite trajectory generated for {regime.name}")
        trajectories[k] = traj.astype(np.float32)

    return trajectories, phi[:, None], phi_raw[:, None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("generated_dynamic_regime_trajectories")
    )
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--trajectories-per-regime", type=int, default=8)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--max-step-multiplier", type=float, default=5.0)
    parser.add_argument(
        "--noise-std-fraction",
        type=float,
        default=0.0,
        help=(
            "Add Gaussian observation noise after standardization. The value is "
            "the noise std as a fraction of each trajectory/dimension std; use "
            "0.05 for 5%% noise."
        ),
    )
    parser.add_argument(
        "--burn-in-steps",
        type=int,
        default=5000,
        help="Integrate at the start parameter before recording, so trajectories begin near the initial attractor.",
    )
    parser.add_argument(
        "--parameter-mode",
        choices=("linear", "constant"),
        default="linear",
        help="Use linear parameter drift or hold the selected parameter constant.",
    )
    parser.add_argument(
        "--constant-parameter",
        choices=("start", "boundary", "end", "midpoint"),
        default="start",
        help="Constant parameter value used when --parameter-mode constant is selected.",
    )
    parser.add_argument(
        "--regime-filter",
        nargs="*",
        default=None,
        help="Optional list of regime names to simulate. Defaults to all regimes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    regimes = default_regimes()
    if args.regime_filter:
        requested = set(args.regime_filter)
        regimes = [regime for regime in regimes if regime.name in requested]
        missing = requested.difference(regime.name for regime in regimes)
        if missing:
            raise ValueError(f"Unknown regime names: {sorted(missing)}")
    if not regimes:
        raise ValueError("No regimes selected.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_traj = []
    all_phi = []
    all_phi_raw = []
    series_regime_names = []
    regime_index = []

    for regime_idx, regime in enumerate(
        tqdm(regimes, desc="Simulating dynamic regimes")
    ):
        traj, phi_one, phi_raw_one = simulate_regime(
            regime,
            n_steps=args.n_steps,
            dt=args.dt,
            n_trajectories=args.trajectories_per_regime,
            rng=rng,
            rtol=args.rtol,
            atol=args.atol,
            max_step_multiplier=args.max_step_multiplier,
            burn_in_steps=args.burn_in_steps,
            parameter_mode=args.parameter_mode,
            constant_parameter=args.constant_parameter,
        )
        traj, mean, std = standardize_trajectories(traj)
        traj = add_gaussian_noise(traj, rng, args.noise_std_fraction)
        all_traj.append(traj)
        all_phi.append(
            np.repeat(phi_one[None, :, :], args.trajectories_per_regime, axis=0)
        )
        all_phi_raw.append(
            np.repeat(phi_raw_one[None, :, :], args.trajectories_per_regime, axis=0)
        )
        series_regime_names.extend([regime.name] * args.trajectories_per_regime)
        regime_index.extend([regime_idx] * args.trajectories_per_regime)
        np.save(args.output_dir / f"{regime.name}_mean.npy", mean)
        np.save(args.output_dir / f"{regime.name}_std.npy", std)

    trajectories = np.concatenate(all_traj, axis=0)
    phi = np.concatenate(all_phi, axis=0)
    phi_raw = np.concatenate(all_phi_raw, axis=0)

    np.save(args.output_dir / "trajectories.npy", trajectories)
    np.save(args.output_dir / "trajectory_phi.npy", phi)
    np.save(args.output_dir / "trajectory_phi_raw.npy", phi_raw)
    np.save(
        args.output_dir / "trajectory_regime_index.npy",
        np.array(regime_index, dtype=np.int64),
    )
    np.save(
        args.output_dir / "trajectory_regime_names.npy", np.array(series_regime_names)
    )

    metadata = {
        "description": (
            "Dynamic regime-change trajectories with known normalized phi_t. "
            "No training windows are sampled by this script; use "
            "resample_time_varying_bifurcation_windows.py afterwards."
        ),
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "shapes": {
            "trajectories": list(trajectories.shape),
            "trajectory_phi": list(phi.shape),
            "trajectory_phi_raw": list(phi_raw.shape),
        },
        "regimes": [asdict(regime) for regime in regimes],
        "phi_convention": (
            "phi arrays are normalized as (raw_parameter - transition_boundary) / "
            "max(abs(start-boundary), abs(end-boundary)); boundary is therefore phi=0. "
            "For these regimes the boundary is a practical numerical transition marker."
        ),
        "observation_convention": (
            "identity regimes save the three ODE states. phase_sine regimes integrate "
            "[x, velocity, theta] but save [x, velocity, sin(theta)] to keep observations "
            "bounded and three-dimensional."
        ),
        "resampling_note": (
            "This output is compatible with resample_time_varying_bifurcation_windows.py. "
            "Use --window-sampling mixed or crossing for linear parameter sweeps, and random "
            "for constant-parameter experiments."
        ),
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved trajectories to {args.output_dir.resolve()}")
    for key, shape in metadata["shapes"].items():
        print(f"{key}: {shape}")


if __name__ == "__main__":
    main()
