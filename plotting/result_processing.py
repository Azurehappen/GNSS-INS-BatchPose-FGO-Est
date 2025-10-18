"""Processing utilities for solver results and plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from fgo_solver.gnss_ins_fgo import EpochLogEntry
from gnss_utils.time_utils import GpsTime
from imu_utils.imu_data_utils import GroundTruthSingleEpoch

from .error_plots import plot_position_errors, plot_enu_trajectory


@dataclass
class PositionErrorSummary:
    """Aggregated statistics for position errors."""

    processed_epochs: int
    evaluated_epochs: int
    horizontal_threshold_m: float
    horizontal_within_threshold_pct: float
    horizontal_rms: float
    horizontal_max: float
    vertical_rms: float
    cov_trace_mean: float
    cov_trace_max: float
    plot_path: Path
    trajectory_plot_path: Path


def analyze_position_results(
    results: Iterable[EpochLogEntry],
    ground_truth_data: Dict[GpsTime, GroundTruthSingleEpoch],
    imu_params,
    *,
    horizontal_threshold_m: float = 1.5,
    output_html: str | Path = "plotting/position_errors.html",
) -> Optional[PositionErrorSummary]:
    """Compute position error metrics and generate plots.

    Returns None if no ground-truth-aligned epochs are available.
    """

    results = list(results)
    if not results:
        return None

    lever_arm_b = np.asarray(imu_params.t_imu_to_ant_in_b)

    timestamps_utc = []
    timestamp_labels = []
    east_errors = []
    north_errors = []
    up_errors = []
    horizontal_errors = []
    vertical_errors = []
    horizontal_stds = []
    vertical_stds = []
    cov_traces = []
    est_east = []
    est_north = []
    gt_east = []
    gt_north = []

    evaluated_epochs = 0

    for entry in results:
        gt = ground_truth_data.get(entry.epoch)
        if gt is None:
            continue
        evaluated_epochs += 1

        rot_enu_from_body = entry.pose.rotation().matrix()
        lever_enu = rot_enu_from_body @ lever_arm_b
        est_ant_enu = entry.pose_enu_m + lever_enu
        diff_enu = est_ant_enu - gt.pos_world_enu_m

        timestamp_utc = entry.epoch.toDatetimeInUtc()
        timestamps_utc.append(timestamp_utc.to_pydatetime())
        timestamp_labels.append(timestamp_utc.strftime("%Y-%m-%d %H:%M:%S"))
        est_east.append(float(est_ant_enu[0]))
        est_north.append(float(est_ant_enu[1]))
        gt_east.append(float(gt.pos_world_enu_m[0]))
        gt_north.append(float(gt.pos_world_enu_m[1]))
        east_errors.append(float(diff_enu[0]))
        north_errors.append(float(diff_enu[1]))
        up_errors.append(float(diff_enu[2]))
        horizontal_errors.append(float(np.linalg.norm(diff_enu[:2])))
        vertical_errors.append(float(diff_enu[2]))

        cov_traces.append(float(np.trace(entry.pose_cov_6x6)))
        pos_cov = entry.pose_cov_6x6[3:6, 3:6]
        if np.all(np.isfinite(pos_cov)):
            east_var = max(float(pos_cov[0, 0]), 0.0)
            north_var = max(float(pos_cov[1, 1]), 0.0)
            up_var = max(float(pos_cov[2, 2]), 0.0)
            horizontal_stds.append(float(np.sqrt(east_var + north_var)))
            vertical_stds.append(float(np.sqrt(up_var)))
        else:
            horizontal_stds.append(None)
            vertical_stds.append(None)

    if not horizontal_errors:
        return None

    horizontal_errors_arr = np.asarray(horizontal_errors)
    vertical_errors_arr = np.asarray(vertical_errors)
    cov_traces_arr = np.asarray(cov_traces)

    within_threshold = np.mean(horizontal_errors_arr < horizontal_threshold_m) * 100.0
    horizontal_rms = float(np.sqrt(np.mean(horizontal_errors_arr**2)))
    vertical_rms = float(np.sqrt(np.mean(vertical_errors_arr**2)))
    horizontal_max = float(horizontal_errors_arr.max())
    cov_trace_mean = float(cov_traces_arr.mean())
    cov_trace_max = float(cov_traces_arr.max())

    plot_path = plot_position_errors(
        timestamps_utc,
        east_errors,
        north_errors,
        up_errors,
        horizontal_errors,
        horizontal_stds,
        vertical_stds,
        output_html=output_html,
    )

    trajectory_plot_path = plot_enu_trajectory(
        est_east,
        est_north,
        gt_east,
        gt_north,
        timestamps=timestamp_labels,
        output_html=Path(output_html).with_name("trajectory_comparison.html"),
    )

    return PositionErrorSummary(
        processed_epochs=len(results),
        evaluated_epochs=evaluated_epochs,
        horizontal_threshold_m=float(horizontal_threshold_m),
        horizontal_within_threshold_pct=within_threshold,
        horizontal_rms=horizontal_rms,
        horizontal_max=horizontal_max,
        vertical_rms=vertical_rms,
        cov_trace_mean=cov_trace_mean,
        cov_trace_max=cov_trace_max,
        plot_path=plot_path,
        trajectory_plot_path=trajectory_plot_path,
    )
