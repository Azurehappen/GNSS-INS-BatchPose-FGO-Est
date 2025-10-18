"""Plotting helpers for visualising position errors."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _to_float_list(values: Sequence[float | None]) -> list[float | None]:
    """Convert a sequence to a list while keeping optional None entries."""
    return [None if v is None else float(v) for v in values]


def plot_position_errors(
    timestamps,
    east_errors_m: Sequence[float],
    north_errors_m: Sequence[float],
    up_errors_m: Sequence[float],
    horizontal_errors_m: Sequence[float],
    horizontal_std_m: Sequence[float | None],
    vertical_std_m: Sequence[float | None],
    *,
    output_html: str | Path = "plotting/position_errors.html",
) -> Path:
    """Create a 2x2 Plotly figure showing ENU position errors and std bands."""

    if not timestamps:
        raise ValueError("timestamps must not be empty")

    if not (
        len(timestamps)
        == len(east_errors_m)
        == len(north_errors_m)
        == len(up_errors_m)
        == len(horizontal_errors_m)
        == len(horizontal_std_m)
        == len(vertical_std_m)
    ):
        raise ValueError("All error and std sequences must have the same length")

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ts = list(timestamps)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "East Error",
            "North Error",
            "Horizontal Error",
            "Vertical Error",
        ),
        shared_xaxes=True,
    )

    fig.add_trace(
        go.Scatter(x=ts, y=east_errors_m, name="East Error", line=dict(color="#1f77b4")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=ts, y=north_errors_m, name="North Error", line=dict(color="#ff7f0e")),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=horizontal_errors_m,
            name="Horizontal Error",
            line=dict(color="#2ca02c"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=_to_float_list(horizontal_std_m),
            name="Horizontal 1σ",
            line=dict(color="#9467bd", dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ts,
            y=[abs(v) for v in up_errors_m],
            name="Vertical Error",
            line=dict(color="#d62728"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=_to_float_list(vertical_std_m),
            name="Vertical 1σ",
            line=dict(color="#8c564b", dash="dash"),
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(matches="x")
    fig.update_layout(
        height=900,
        width=1400,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=30, t=80, b=50),
        title="Position Errors vs Time",
    )

    fig.update_yaxes(title_text="Error (m)", row=1, col=1)
    fig.update_yaxes(title_text="Error (m)", row=1, col=2)
    fig.update_yaxes(title_text="Error (m)", row=2, col=1)
    fig.update_yaxes(title_text="Error (m)", row=2, col=2)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=2)

    fig.write_html(output_path)
    return output_path


def plot_enu_trajectory(
    est_east_m: Sequence[float],
    est_north_m: Sequence[float],
    gt_east_m: Sequence[float],
    gt_north_m: Sequence[float],
    timestamps=None,
    *,
    output_html: str | Path = "plotting/trajectory_comparison.html",
):
    """Plot estimated vs ground-truth trajectory in the ENU plane."""

    if not est_east_m:
        raise ValueError("Trajectory sequences must not be empty")

    if not (
        len(est_east_m)
        == len(est_north_m)
        == len(gt_east_m)
        == len(gt_north_m)
    ):
        raise ValueError("Trajectory sequences must have the same length")

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=est_east_m,
            y=est_north_m,
            mode="markers+lines",
            name="Estimated",
            marker=dict(color="#1f77b4", size=6),
            line=dict(color="#1f77b4"),
            text=timestamps,
            hovertemplate="East: %{x:.3f} m<br>North: %{y:.3f} m<extra>%{text}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gt_east_m,
            y=gt_north_m,
            mode="markers+lines",
            name="Ground Truth",
            marker=dict(color="#d62728", size=6, symbol="diamond"),
            line=dict(color="#d62728"),
            text=timestamps,
            hovertemplate="East: %{x:.3f} m<br>North: %{y:.3f} m<extra>%{text}</extra>",
        )
    )

    fig.update_layout(
        title="ENU Trajectory Comparison",
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=900,
        height=800,
        margin=dict(l=60, r=40, t=80, b=60),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    fig.write_html(output_path)
    return output_path
