"""Simulate trajectories with fixed parameter values for each trajectory.

This mirrors time_varying_bifurcation_data/generate_dynamic_regime_trajectories.py
but does not sweep parameters over time. Instead, each trajectory is simulated at
one constant normalized phi value selected by --phi-values.

Output layout:

    trajectories.npy                  (S, T, 3)
    trajectories_clean.npy            (S, T, 3)  only when noise is requested
    ../<output-dir-name>_clean/       clean dataset mirror for resampling
    trajectory_phi.npy                (S, T, 1)
    trajectory_phi_raw.npy            (S, T, 1)
    trajectory_regime_index.npy       (S,)
    trajectory_regime_names.npy       (S,)
    trajectory_phi_value_index.npy    (S,)
    trajectory_phi_constants.npy      (S,)
    trajectory_phi_raw_constants.npy  (S,)
    metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from time_varying_bifurcation_data.generate_dynamic_regime_trajectories import (  # noqa: E402
    RHS,
    Regime,
    add_gaussian_noise,
    default_regimes,
    normalize_phi,
    observe_state,
    standardize_trajectories,
)

Array = np.ndarray


def raw_parameter_from_normalized_phi(regime: Regime, phi: float) -> float:
    scale = max(abs(regime.start - regime.boundary), abs(regime.end - regime.boundary), 1e-8)
    return float(regime.boundary + phi * scale)


def phi_label(phi: float) -> str:
    return f"{phi:+.6g}".replace("+", "p").replace("-", "m").replace(".", "p")


def simulate_constant_parameter(
    regime: Regime,
    phi_value: float,
    n_steps: int,
    dt: float,
    n_trajectories: int,
    rng: np.random.Generator,
    rtol: float,
    atol: float,
    max_step_multiplier: float,
    burn_in_steps: int,
) -> tuple[Array, Array, Array, float]:
    t_eval = np.arange(n_steps, dtype=np.float64) * dt
    t_end = float(t_eval[-1])
    rhs = RHS[regime.rhs_key]
    raw_parameter = raw_parameter_from_normalized_phi(regime, phi_value)
    normalized_phi = float(
        normalize_phi(
            np.array([raw_parameter], dtype=np.float32),
            regime.start,
            regime.end,
            regime.boundary,
        )[0]
    )

    trajectories = np.empty((n_trajectories, n_steps, 3), dtype=np.float32)

    def ode(t: float, x: Array) -> Array:
        dx = rhs(t, x, raw_parameter, **regime.params)
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
                ode,
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
            ode,
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

    phi = np.full((n_steps, 1), normalized_phi, dtype=np.float32)
    phi_raw = np.full((n_steps, 1), raw_parameter, dtype=np.float32)
    return trajectories, phi, phi_raw, raw_parameter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated_constant_parameter_trajectories"),
    )
    parser.add_argument("--n-steps", type=int, default=100000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--phi-values",
        nargs="+",
        type=float,
        default=[-1.0, 0.0, 1.0],
        help="Normalized phi values to simulate for every selected regime.",
    )
    parser.add_argument("--trajectories-per-phi", type=int, default=8)
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
        help="Integrate at the selected constant parameter before recording.",
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
    if args.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if args.trajectories_per_phi <= 0:
        raise ValueError("trajectories_per_phi must be positive")
    if not args.phi_values:
        raise ValueError("At least one --phi-values entry is required")

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
    clean_output_dir = (
        args.output_dir.with_name(f"{args.output_dir.name}_clean")
        if args.noise_std_fraction > 0.0
        else None
    )

    all_traj = []
    all_traj_clean = []
    all_phi = []
    all_phi_raw = []
    series_regime_names = []
    regime_index = []
    phi_value_index = []
    phi_constants = []
    phi_raw_constants = []

    jobs = [(regime_idx, regime, phi_idx, phi) for regime_idx, regime in enumerate(regimes) for phi_idx, phi in enumerate(args.phi_values)]
    for regime_idx, regime, phi_idx, phi in tqdm(jobs, desc="Simulating constant-parameter regimes"):
        traj, phi_one, phi_raw_one, raw_parameter = simulate_constant_parameter(
            regime,
            phi_value=phi,
            n_steps=args.n_steps,
            dt=args.dt,
            n_trajectories=args.trajectories_per_phi,
            rng=rng,
            rtol=args.rtol,
            atol=args.atol,
            max_step_multiplier=args.max_step_multiplier,
            burn_in_steps=args.burn_in_steps,
        )
        traj_clean, mean, std = standardize_trajectories(traj)
        if args.noise_std_fraction > 0.0:
            traj = add_gaussian_noise(traj_clean.copy(), rng, args.noise_std_fraction)
            all_traj_clean.append(traj_clean)
        else:
            traj = traj_clean
        all_traj.append(traj)
        all_phi.append(np.repeat(phi_one[None, :, :], args.trajectories_per_phi, axis=0))
        all_phi_raw.append(np.repeat(phi_raw_one[None, :, :], args.trajectories_per_phi, axis=0))
        series_regime_names.extend([regime.name] * args.trajectories_per_phi)
        regime_index.extend([regime_idx] * args.trajectories_per_phi)
        phi_value_index.extend([phi_idx] * args.trajectories_per_phi)
        phi_constants.extend([float(phi_one[0, 0])] * args.trajectories_per_phi)
        phi_raw_constants.extend([raw_parameter] * args.trajectories_per_phi)

        stem = f"{regime.name}_phi_{phi_label(float(phi_one[0, 0]))}"
        np.save(args.output_dir / f"{stem}_mean.npy", mean)
        np.save(args.output_dir / f"{stem}_std.npy", std)

    trajectories = np.concatenate(all_traj, axis=0)
    trajectories_clean = (
        np.concatenate(all_traj_clean, axis=0)
        if args.noise_std_fraction > 0.0
        else None
    )
    phi = np.concatenate(all_phi, axis=0)
    phi_raw = np.concatenate(all_phi_raw, axis=0)

    np.save(args.output_dir / "trajectories.npy", trajectories)
    if trajectories_clean is not None:
        np.save(args.output_dir / "trajectories_clean.npy", trajectories_clean)
    np.save(args.output_dir / "trajectory_phi.npy", phi)
    np.save(args.output_dir / "trajectory_phi_raw.npy", phi_raw)
    regime_index_array = np.array(regime_index, dtype=np.int64)
    regime_names_array = np.array(series_regime_names)
    phi_value_index_array = np.array(phi_value_index, dtype=np.int64)
    phi_constants_array = np.array(phi_constants, dtype=np.float32)
    phi_raw_constants_array = np.array(phi_raw_constants, dtype=np.float32)

    np.save(args.output_dir / "trajectory_regime_index.npy", regime_index_array)
    np.save(args.output_dir / "trajectory_regime_names.npy", regime_names_array)
    np.save(args.output_dir / "trajectory_phi_value_index.npy", phi_value_index_array)
    np.save(args.output_dir / "trajectory_phi_constants.npy", phi_constants_array)
    np.save(args.output_dir / "trajectory_phi_raw_constants.npy", phi_raw_constants_array)

    metadata = {
        "description": (
            "Constant-parameter trajectories. Each trajectory uses one fixed raw "
            "parameter value derived from a requested normalized phi value."
        ),
        "args": vars(args) | {"output_dir": str(args.output_dir)},
        "shapes": {
            "trajectories": list(trajectories.shape),
            **(
                {"trajectories_clean": list(trajectories_clean.shape)}
                if trajectories_clean is not None
                else {}
            ),
            "trajectory_phi": list(phi.shape),
            "trajectory_phi_raw": list(phi_raw.shape),
        },
        "trajectory_files": {
            "trajectories": (
                "trajectories.npy contains standardized trajectories with Gaussian "
                "noise when noise_std_fraction > 0, otherwise standardized clean trajectories."
            ),
            **(
                {
                    "trajectories_clean": (
                        "trajectories_clean.npy contains the same standardized trajectories "
                        "before observation noise was added."
                    )
                }
                if trajectories_clean is not None
                else {}
            ),
            **(
                {
                    "clean_dataset_dir": (
                        f"{clean_output_dir} mirrors this dataset for resampling; "
                        "its trajectories.npy is the clean standardized trajectory array."
                    )
                }
                if clean_output_dir is not None
                else {}
            ),
        },
        "regimes": [asdict(regime) for regime in regimes],
        "requested_phi_values": [float(v) for v in args.phi_values],
        "phi_convention": (
            "phi arrays are normalized as (raw_parameter - transition_boundary) / "
            "max(abs(start-boundary), abs(end-boundary)); boundary is therefore phi=0. "
            "For this dataset phi is constant over each trajectory."
        ),
        "observation_convention": (
            "identity regimes save the three ODE states. phase_sine regimes integrate "
            "[x, velocity, theta] but save [x, velocity, sin(theta)] to keep observations "
            "bounded and three-dimensional."
        ),
        "resampling_note": (
            "Use diffrentbutconst_prams/resample_constant_parameter_windows.py. "
            "The time-varying resampler only works for this data when --window-sampling random is used."
        ),
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if clean_output_dir is not None and trajectories_clean is not None:
        clean_output_dir.mkdir(parents=True, exist_ok=True)
        np.save(clean_output_dir / "trajectories.npy", trajectories_clean)
        np.save(clean_output_dir / "trajectory_phi.npy", phi)
        np.save(clean_output_dir / "trajectory_phi_raw.npy", phi_raw)
        np.save(clean_output_dir / "trajectory_regime_index.npy", regime_index_array)
        np.save(clean_output_dir / "trajectory_regime_names.npy", regime_names_array)
        np.save(clean_output_dir / "trajectory_phi_value_index.npy", phi_value_index_array)
        np.save(clean_output_dir / "trajectory_phi_constants.npy", phi_constants_array)
        np.save(clean_output_dir / "trajectory_phi_raw_constants.npy", phi_raw_constants_array)

        clean_metadata = metadata | {
            "description": (
                "Clean constant-parameter trajectories mirrored from a noisy run. "
                "Each trajectory uses one fixed raw parameter value derived from a "
                "requested normalized phi value."
            ),
            "args": metadata["args"] | {"output_dir": str(clean_output_dir)},
            "shapes": {
                "trajectories": list(trajectories_clean.shape),
                "trajectory_phi": list(phi.shape),
                "trajectory_phi_raw": list(phi_raw.shape),
            },
            "trajectory_files": {
                "trajectories": (
                    "trajectories.npy contains standardized clean trajectories before "
                    "Gaussian observation noise was added in the paired noisy dataset."
                )
            },
            "paired_noisy_dir": str(args.output_dir),
        }
        with open(clean_output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(clean_metadata, f, indent=2)

    print(f"Saved trajectories to {args.output_dir.resolve()}")
    if clean_output_dir is not None:
        print(f"Saved clean trajectories to {clean_output_dir.resolve()}")
    for key, shape in metadata["shapes"].items():
        print(f"{key}: {shape}")


if __name__ == "__main__":
    main()
