from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from with_phi_summary_plot_utils import plot_prediction_grid, resolve_requested_phi_values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a 3D with-phi prediction summary image from a saved .npz prediction cache."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--params",
        nargs="+",
        type=float,
        default=None,
        help="Subset of cached parameter values to plot. Interpreted as phi unless --parameter-space raw is set.",
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


def main() -> None:
    args = parse_args()
    cache = np.load(args.cache)
    truth_future = cache["truth_future"]
    pred_future = cache["pred_future"]
    phi_values = cache["phi_values"].astype(float)
    raw_values = cache["raw_values"].astype(float)
    selected_indices = resolve_requested_phi_values(
        phi_values, raw_values, args.params, args.parameter_space, args.parameter_atol
    )
    plot_prediction_grid(
        args.output,
        truth_future,
        pred_future,
        phi_values,
        raw_values,
        selected_indices=selected_indices,
        title=args.title,
        label_space=args.label_space,
        raw_name=args.raw_name,
        single_row=args.single_row,
        dpi=args.dpi,
        view_elev=args.view_elev,
        view_azim=args.view_azim,
    )
    print(f"Written plot: {args.output}")


if __name__ == "__main__":
    main()
