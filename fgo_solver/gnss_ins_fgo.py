"""GNSS/INS factor graph optimizer utilities."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import gtsam
import numpy as np
import pymap3d as pm

from constants.gnss_constants import Constellation, CycleSlipType
from constants.parameters import *
from fgo_solver import logger_utils, meas_error_models, outlier_helper, utils
from fgo_solver.wls_solver import solve_wls_position
from gnss_utils.gnss_data_utils import (
    GnssMeasurementChannel,
    SignalChannelId,
    SignalType,
)
from gnss_utils.model_utils import (
    compute_ecef_enu_rot_mat,
    compute_world_frame_coord_from_ecef,
    elevation_based_noise_var,
    select_pivot_satellite,
)
from gnss_utils.satellite_utils import compute_sat_elev_az
from gnss_utils.time_utils import GpsTime
from imu_utils.imu_data_utils import ImuSingleEpoch

try:  # Progress display (optional)
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None


POSE_KEY = lambda idx: gtsam.symbol("x", idx)
VEL_KEY = lambda idx: gtsam.symbol("v", idx)
BIAS_KEY = lambda idx: gtsam.symbol("b", idx)
AMB_KEY = lambda idx: gtsam.symbol("a", idx)


@dataclass
class BatchMeasFactor:
    """Container for double-differenced GNSS measurements within a signal type."""

    pivot: GnssMeasurementChannel
    meas: List[GnssMeasurementChannel]
    code_mask: List[bool]
    phase_mask: List[bool]


@dataclass
class AmbiguityState:
    """Tracks metadata for a carrier-phase ambiguity estimate."""

    key: int
    signal_type: SignalType
    pivot_id: SignalChannelId
    last_update_sec: float
    last_epoch_idx: int


@dataclass
class EpochLogEntry:
    """Stores solver outputs for an epoch."""

    epoch: GpsTime
    pose_key: int
    vel_key: int
    bias_key: int
    pose: gtsam.Pose3
    pose_enu_m: np.ndarray
    vel_enu_mps: np.ndarray
    pose_cov_6x6: np.ndarray
    vel_cov_3x3: np.ndarray


@dataclass
class EpochUpdateData:
    """Container for per-epoch factor graph updates."""

    factors: gtsam.NonlinearFactorGraph
    values: gtsam.Values
    pose_key: int
    vel_key: int
    bias_key: int
    rel_time: float
    dd_sat_ids: Set[Tuple[Constellation, int]]
    epoch_idx: int


class RtkInsFgo:
    """GNSS/INS factor graph optimizer built on GTSAM."""

    _lever_arm_b = np.zeros(3)

    def __init__(
        self,
        imu_params: Any,
        imu_data_list: Sequence[ImuSingleEpoch],
        *,
        use_isam: bool = True,
        final_batch_opt: bool = False,
        isam_relinearize_skip: int = 10,
        show_progress: bool = False,
        logger: Optional[logging.Logger] = None,
        debug_times: Optional[Iterable[GpsTime]] = None,
    ) -> None:
        self.imu_params = imu_params
        self.use_isam = use_isam
        self.final_batch_opt = final_batch_opt
        self.isam_relinearize_skip = isam_relinearize_skip
        self.t_imu_to_ant_in_b = np.asarray(imu_params.t_imu_to_ant_in_b)
        RtkInsFgo._lever_arm_b = self.t_imu_to_ant_in_b
        self.logger = logger or logging.getLogger(__name__)
        self.debug_times: Set[GpsTime] = set(debug_times or [])

        self.preint_params = self._make_preintegration_params()
        self.bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate((imu_params.acc_bias_std, imu_params.gyro_bias_std))
        )

        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.isam: Optional[gtsam.ISAM2] = None
        self._current_dd_sat_ids: Set[Tuple[Constellation, int]] = set()
        self._latest_estimate: Optional[gtsam.Values] = None
        self._force_new_ambiguity_next_epoch = False

        self.smoother_init_time: Optional[GpsTime] = None
        self.prev_epoch_time: Optional[GpsTime] = None

        self.current_epoch_idx = -1
        self.prev_pose: Optional[gtsam.Pose3] = None
        self.prev_pose_cov: Optional[np.ndarray] = None
        self.prev_velocity = np.zeros(3)
        self.prev_velocity_cov: Optional[np.ndarray] = None
        self.prev_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        self._last_successful_wls_enu: Optional[np.ndarray] = None
        self._epoch_keys: List[Tuple[GpsTime, int, int, int]] = []

        self.ambiguities: Dict[SignalChannelId, AmbiguityState] = {}
        self.next_amb_idx = 0
        self.current_pivot_by_type: Dict[SignalType, SignalChannelId] = {}

        self.imu_data_list = list(imu_data_list)
        self._imu_idx = 0  # index of next IMU sample to process
        self._last_imu_sample: Optional[ImuSingleEpoch] = None
        self.show_progress = show_progress

        self.result_log: List[EpochLogEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the optimizer state."""
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_values = gtsam.Values()
        self.isam = None
        self._current_dd_sat_ids.clear()
        self._force_new_ambiguity_next_epoch = False

        self.smoother_init_time = None
        self.prev_epoch_time = None
        self.current_epoch_idx = -1
        self.prev_pose = None
        self.prev_pose_cov = None
        self.prev_velocity = np.zeros(3)
        self.prev_velocity_cov = None
        self.prev_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))
        self._last_successful_wls_enu = None
        self._latest_estimate = None
        self._epoch_keys: List[Tuple[GpsTime, int, int, int]] = []

        self.ambiguities.clear()
        self.next_amb_idx = 0
        self.current_pivot_by_type.clear()
        self._imu_idx = 0
        self._last_imu_sample = None
        self.result_log.clear()

    def process_epoch(
        self,
        epoch: GpsTime,
        channels: Dict[SignalChannelId, GnssMeasurementChannel],
    ) -> Optional[EpochLogEntry]:
        """Ingest a GNSS epoch and integrate IMU data internally."""

        if not channels:
            return None

        if self.smoother_init_time is None:
            return self._initialize_first_epoch(epoch, channels)

        if self.prev_epoch_time and epoch <= self.prev_epoch_time:
            raise ValueError("Epoch timestamps must be strictly increasing.")

        rel_time = self._relative_time(epoch)
        self.current_epoch_idx += 1
        pose_key = POSE_KEY(self.current_epoch_idx)
        vel_key = VEL_KEY(self.current_epoch_idx)
        bias_key = BIAS_KEY(self.current_epoch_idx)

        preint = self._integrate_imu(epoch)
        predicted_state = preint.predict(
            gtsam.NavState(self.prev_pose, self.prev_velocity), self.prev_bias
        )

        pose_guess = self._pose_guess_from_wls(predicted_state, channels)
        velocity_guess = np.asarray(predicted_state.velocity(), dtype=float)

        update = self._build_epoch_update(
            epoch,
            pose_key,
            vel_key,
            bias_key,
            pose_guess,
            velocity_guess,
            preint,
            channels,
            rel_time,
        )

        debug_enabled = epoch in self.debug_times
        if debug_enabled:
            pose_std_vec = (
                np.sqrt(np.diag(self.prev_pose_cov))
                if self.prev_pose_cov is not None
                else None
            )
            vel_std_vec = (
                np.sqrt(np.diag(self.prev_velocity_cov))
                if self.prev_velocity_cov is not None
                else None
            )
            logger_utils.log_epoch_debug(
                self.logger,
                epoch,
                pose_guess,
                predicted_velocity=velocity_guess,
                pose_std=pose_std_vec,
                vel_std=vel_std_vec,
                measurements=[],
            )

        self._append_update(update)
        self._merge_initial_values(update.values)
        self._epoch_keys.append((epoch, pose_key, vel_key, bias_key))

        self.prev_epoch_time = epoch

        if self.use_isam:
            self._ensure_isam()
            self.isam.update(update.factors, update.values)
            estimate = self.isam.calculateEstimate()
            self._latest_estimate = estimate
            entry = self._extract_epoch_result(
                epoch, pose_key, vel_key, bias_key, estimate
            )
            self.prev_pose = estimate.atPose3(pose_key)
            self.prev_pose_cov = entry.pose_cov_6x6
            self.prev_velocity_cov = entry.vel_cov_3x3
            self.prev_velocity = estimate.atVector(vel_key)
            self.prev_bias = estimate.atConstantBias(bias_key)

            if debug_enabled:
                logger_utils.log_epoch_debug(
                    self.logger,
                    epoch,
                    entry.pose,
                    predicted_velocity=entry.vel_enu_mps,
                    pose_std=(
                        np.sqrt(np.diag(entry.pose_cov_6x6))
                        if np.all(np.isfinite(entry.pose_cov_6x6))
                        else None
                    ),
                    vel_std=(
                        np.sqrt(np.diag(entry.vel_cov_3x3))
                        if np.all(np.isfinite(entry.vel_cov_3x3))
                        else None
                    ),
                    measurements=[],
                )
            return entry

        self.prev_pose = pose_guess
        self.prev_pose_cov = None
        self.prev_velocity_cov = None
        self.prev_velocity = velocity_guess
        return None

    def get_results(self) -> List[EpochLogEntry]:
        """Return accumulated solver log entries."""
        return self.result_log

    def run(
        self,
        rover_obs: Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]],
        *,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        show_progress: Optional[bool] = None,
    ) -> List[EpochLogEntry]:
        """Process a range of GNSS epochs using the factor graph optimizer."""

        self.reset()

        epochs_sorted = sorted(rover_obs.keys())
        if not epochs_sorted:
            return []

        total_epochs = len(epochs_sorted)
        if start_idx < 0 or start_idx >= total_epochs:
            raise ValueError("start_idx is outside the available epochs range")
        if end_idx is None or end_idx >= total_epochs:
            end_idx = total_epochs - 1
        if end_idx < start_idx:
            raise ValueError("end_idx must be greater than or equal to start_idx")

        epoch_subset = epochs_sorted[start_idx : end_idx + 1]
        progress_enabled = (
            self.show_progress if show_progress is None else show_progress
        )

        iterator: Iterable[GpsTime]
        progress_bar = None
        if progress_enabled and tqdm is not None:
            progress_bar = tqdm(epoch_subset, desc="RTK-INS FGO", unit="epoch")
            iterator = progress_bar
        else:
            iterator = epoch_subset

        try:
            for epoch in iterator:
                channels = rover_obs.get(epoch)
                if not channels:
                    continue
                self.process_epoch(epoch, channels)

            if self.final_batch_opt and self._epoch_keys:
                initial_guess = self._prepare_initial_guess(gtsam.Values())
                self._assert_all_values_connected(self.graph, initial_guess)
                estimate = self._optimize_with_lm(initial_guess)
                self._latest_estimate = estimate

                self.result_log.clear()
                for epoch, pose_key, vel_key, bias_key in self._epoch_keys:
                    self._extract_epoch_result(
                        epoch, pose_key, vel_key, bias_key, estimate
                    )
            elif (
                self.use_isam and self._epoch_keys and self._latest_estimate is not None
            ):
                # Reconstruct log from the final ISAM estimate to return a full-trajectory snapshot.
                self.result_log.clear()
                for epoch, pose_key, vel_key, bias_key in self._epoch_keys:
                    self._extract_epoch_result(
                        epoch, pose_key, vel_key, bias_key, self._latest_estimate
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        return self.get_results()

    # ------------------------------------------------------------------
    # Static measurement model
    # ------------------------------------------------------------------

    @staticmethod
    def error_meas_batch(
        measurement: BatchMeasFactor,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]],
    ) -> np.ndarray:
        """Non-linear error function for double-differenced GNSS factors."""

        pos_key = this.keys()[0]
        meas_list = measurement.meas
        code_mask = measurement.code_mask
        phase_mask = measurement.phase_mask
        if not meas_list:
            return np.zeros(0)
        if len(code_mask) != len(meas_list) or len(phase_mask) != len(meas_list):
            raise ValueError("Mask length does not align with measurement list.")

        pose = values.atPose3(pos_key)
        lever_b = RtkInsFgo._lever_arm()

        amb_keys = [this.keys()[i + 1] for i in range(len(meas_list))]
        amb_vals = []
        for key in amb_keys:
            if not values.exists(key):
                raise KeyError(
                    "GNSS ambiguity key missing from linearization point; "
                    "ensure it is inserted before evaluating the factor."
                )
            amb_vals.append(values.atDouble(key))

        residual, J_pose, J_amb_list = (
            meas_error_models.double_difference_batch_residual_and_jacobian(
                measurement.pivot,
                meas_list,
                pose,
                lever_b,
                amb_vals=amb_vals,
                compute_jacobian=jacobians is not None,
            )
        )

        code_indices = [idx for idx, use in enumerate(code_mask) if use]
        phase_indices = [idx for idx, use in enumerate(phase_mask) if use]
        row_indices = code_indices + [len(meas_list) + idx for idx in phase_indices]

        if not row_indices:
            return np.zeros(0)

        residual = residual[row_indices]

        if jacobians is not None:
            if J_pose is None or J_amb_list is None:
                raise RuntimeError("Analytic Jacobian computation failed")
            J_pose = J_pose[row_indices, :]
            if jacobians[0].size == 0:
                jacobians[0] = J_pose
            else:
                jacobians[0][:] = J_pose
            for i, J_amb in enumerate(J_amb_list):
                J_sel = J_amb[row_indices, :]
                if jacobians[i + 1].size == 0:
                    jacobians[i + 1] = J_sel
                else:
                    jacobians[i + 1][:] = J_sel

        return residual

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_first_epoch(
        self,
        epoch: GpsTime,
        channels: Dict[SignalChannelId, GnssMeasurementChannel],
    ) -> EpochLogEntry:
        if len(channels) < 4:
            raise ValueError("Need at least 4 satellites for initialization.")

        self.smoother_init_time = epoch
        self.prev_epoch_time = epoch
        self.current_epoch_idx = 0

        initial_ecef, _ = solve_wls_position(channels)
        initial_enu = compute_world_frame_coord_from_ecef(initial_ecef)
        self._last_successful_wls_enu = initial_enu

        pose_key = POSE_KEY(0)
        vel_key = VEL_KEY(0)
        bias_key = BIAS_KEY(0)

        initial_pose = gtsam.Pose3(gtsam.Rot3.Ypr(INIT_YAW_RAD, 0.0, 0.0), initial_enu)
        initial_velocity = np.zeros(3, dtype=float)
        initial_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

        restart_attitude_std_rad = np.array(
            [np.deg2rad(50.0), np.deg2rad(50.0), 2 * np.pi], dtype=float
        )
        pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate(
                (
                    restart_attitude_std_rad,
                    np.asarray(
                        [INIT_POS_HOR_STD_M, INIT_POS_HOR_STD_M, INIT_POS_VER_STD_M],
                        dtype=float,
                    ),
                )
            )
        )
        velocity_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, INIT_VEL_STD_MPS)
        bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate((INIT_ACC_BIAS_STD, INIT_GYRO_BIAS_STD))
        )

        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        graph.push_back(
            gtsam.PriorFactorPose3(pose_key, initial_pose, pose_prior_noise)
        )
        graph.push_back(
            gtsam.PriorFactorVector(vel_key, initial_velocity, velocity_prior_noise)
        )
        graph.push_back(
            gtsam.PriorFactorConstantBias(bias_key, initial_bias, bias_prior_noise)
        )

        values.insert(pose_key, initial_pose)
        values.insert(vel_key, initial_velocity)
        values.insert(bias_key, initial_bias)

        self._current_dd_sat_ids.clear()
        self._add_gnss_factors(
            epoch,
            pose_key,
            channels,
            initial_pose,
            initial_velocity,
            graph,
            values,
        )
        self._prime_imu_index(epoch)

        self.graph = graph
        self.initial_values = values
        self._latest_estimate = self.initial_values
        self._epoch_keys = [(epoch, pose_key, vel_key, bias_key)]

        if self.use_isam:
            self._ensure_isam()
            self.isam.update(self.graph, self.initial_values)
            estimate = self.isam.calculateEstimate()
            self._latest_estimate = estimate
            entry = self._extract_epoch_result(
                epoch, pose_key, vel_key, bias_key, estimate
            )
            self.prev_pose = estimate.atPose3(pose_key)
            self.prev_pose_cov = entry.pose_cov_6x6
            self.prev_velocity_cov = entry.vel_cov_3x3
            self.prev_velocity = estimate.atVector(vel_key)
            self.prev_bias = estimate.atConstantBias(bias_key)
            return entry

        self.prev_pose = initial_pose
        self.prev_velocity = initial_velocity
        return None

    def _pose_guess_from_wls(
        self,
        predicted_state: gtsam.NavState,
        channels: Dict[SignalChannelId, GnssMeasurementChannel],
    ) -> gtsam.Pose3:
        rotation = predicted_state.pose().rotation()
        translation = np.asarray(predicted_state.pose().translation(), dtype=float)

        dd_count = self._count_available_dd_signals(channels)
        if dd_count >= 6:
            try:
                wls_ecef, _ = solve_wls_position(channels)
                translation = compute_world_frame_coord_from_ecef(wls_ecef)
                self._last_successful_wls_enu = translation
            except Exception as exc:  # pragma: no cover - defensive fallback
                if self._last_successful_wls_enu is not None:
                    translation = self._last_successful_wls_enu
                self.logger.debug(
                    "WLS initialization failed (dd_count=%s): %s",
                    dd_count,
                    exc,
                )
        elif self._last_successful_wls_enu is not None:
            translation = self._last_successful_wls_enu

        return gtsam.Pose3(rotation, translation)

    def _count_available_dd_signals(
        self, channels: Dict[SignalChannelId, GnssMeasurementChannel]
    ) -> int:
        per_signal_count: Dict[SignalType, int] = defaultdict(int)
        for ch in channels.values():
            if (
                ch.code_m is None
                or ch.phase_m is None
                or ch.sat_pos_ecef_m is None
                or ch.wavelength_m is None
                or ch.wavelength_m == 0.0
                or ch.signal_id is None
                or ch.signal_id.signal_type is None
            ):
                continue
            per_signal_count[ch.signal_id.signal_type] += 1

        dd_count = 0
        for count in per_signal_count.values():
            if count >= 2:
                dd_count += count - 1
        return dd_count

    def _add_gnss_factors(
        self,
        epoch: GpsTime,
        pose_key: int,
        channels: Dict[SignalChannelId, GnssMeasurementChannel],
        pose_guess: gtsam.Pose3,
        velocity_guess: Optional[np.ndarray],
        graph: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        pose_cov_prior: Optional[np.ndarray] = None,
    ) -> None:
        if not channels:
            return

        receiver_ecef = self._antenna_ecef(pose_guess)
        lat_rad, lon_rad, _ = pm.ecef2geodetic(*receiver_ecef, deg=False)
        ecef_to_enu = compute_ecef_enu_rot_mat(lat_rad, lon_rad)
        rel_time = self._relative_time(epoch)
        lever_arm_b = self._lever_arm()
        pose_cov_for_gate = (
            None
            if pose_cov_prior is None or not np.all(np.isfinite(pose_cov_prior))
            else pose_cov_prior
        )

        debug_enabled = epoch in self.debug_times
        debug_records: List[logger_utils.MeasurementDebugRecord] = []

        grouped: Dict[SignalType, Dict[SignalChannelId, GnssMeasurementChannel]] = (
            defaultdict(dict)
        )
        for scid, ch in channels.items():
            if scid.signal_type in SIGNAL_TO_EXCLUDED_FROM_SOL:
                continue
            if ch.sat_pos_ecef_m is None:
                continue
            elev_deg, _ = compute_sat_elev_az(
                ecef_to_enu, receiver_ecef, ch.sat_pos_ecef_m
            )
            ch.elevation_deg = elev_deg
            if elev_deg < GnssParameters.ELEVATION_MASK_DEG:
                continue
            grouped[scid.signal_type][scid] = ch

        def emit_debug(skipped: bool) -> None:
            if not debug_enabled:
                return
            pose_std_vec = (
                np.sqrt(np.diag(pose_cov_for_gate))
                if pose_cov_for_gate is not None
                else None
            )
            vel_std_vec = (
                np.sqrt(np.diag(self.prev_velocity_cov))
                if self.prev_velocity_cov is not None
                else None
            )
            logger_utils.log_epoch_debug(
                self.logger,
                epoch,
                pose_guess,
                velocity_guess,
                pose_std_vec,
                vel_std_vec,
                debug_records,
                skipped=skipped,
            )

        pending_batches: List[Dict[str, Any]] = []
        total_code_sat_count = 0

        for signal_type, signal_channels in grouped.items():
            if len(signal_channels) < GnssParameters.MIN_NUM_SIGNALS_FOR_DD:
                continue
            pivot_info = select_pivot_satellite(
                signal_channels,
                receiver_ecef,
                ecef_to_enu,
            )
            if pivot_info is None:
                continue

            pivot_id, pivot_ch, _ = pivot_info
            pivot_changed = self._update_pivot(signal_type, pivot_id, pivot_ch)

            pivot_code_std, pivot_phase_std = self._measurement_std(
                signal_type, pivot_ch.elevation_deg
            )
            pivot_ch.sigma_code_m = pivot_code_std
            pivot_ch.sigma_phase_m = pivot_phase_std

            force_new_ambiguity = (
                pivot_changed
                or GnssParameters.ambiguity_mode == AmbiguityMode.INSTANTANEOUS
                or self._force_new_ambiguity_next_epoch
            )

            candidate_infos: List[Dict[str, Any]] = []
            for scid, ch in signal_channels.items():
                if scid == pivot_id:
                    continue

                code_std, phase_std = self._measurement_std(
                    signal_type, ch.elevation_deg
                )

                state = self.ambiguities.get(scid)
                slip_not_detected = (
                    ch.cycle_slip_status is not None
                    and ch.cycle_slip_status == CycleSlipType.NOT_DETECTED
                )
                used_last_epoch = (
                    state is not None
                    and state.last_epoch_idx == self.current_epoch_idx - 1
                )
                will_initialize = (
                    state is None
                    or force_new_ambiguity
                    or not slip_not_detected
                    or not used_last_epoch
                )

                prior_ambiguity: Optional[float] = None
                if state is not None and not will_initialize:
                    if values.exists(state.key):
                        prior_ambiguity = values.atDouble(state.key)
                    elif (
                        self._latest_estimate is not None
                        and self._latest_estimate.exists(state.key)
                    ):
                        prior_ambiguity = self._latest_estimate.atDouble(state.key)

                amb_guess = prior_ambiguity if prior_ambiguity is not None else 0.0
                residual_single, J_pose_single, _ = (
                    meas_error_models.double_difference_batch_residual_and_jacobian(
                        pivot_ch,
                        [ch],  # process single signal
                        pose_guess,
                        lever_arm_b,
                        amb_vals=[amb_guess],
                        compute_jacobian=True,
                    )
                )
                code_residual = float(residual_single[0])
                phase_residual = (
                    float(residual_single[1]) if residual_single.shape[0] > 1 else None
                )
                if will_initialize:
                    phase_residual = None

                use_code = True
                note: Optional[str] = None
                if (
                    pose_cov_for_gate is not None
                    and not pivot_changed
                    and J_pose_single is not None
                ):
                    noise_code_var = np.array(
                        [pivot_code_std**2 + code_std**2], dtype=float
                    )
                    mask_single = outlier_helper.neyman_pearson_residual_test(
                        OutlierOptParam.RESIDUAL_GATE_THRESHOLD,
                        pose_cov_for_gate,
                        residual_single[:1],
                        J_pose_single[:1, :],
                        noise_code_var,
                    )
                    use_code = bool(mask_single[0])
                    if not use_code:
                        note = "code gate failed"

                debug_record: Optional[logger_utils.MeasurementDebugRecord] = None
                if debug_enabled:
                    debug_record = logger_utils.MeasurementDebugRecord(
                        constellation=scid.signal_type.constellation.name,
                        prn=scid.satellite_id.prn,
                        signal_type=scid.signal_type.obs_code,
                        code_residual=code_residual,
                        phase_residual=phase_residual,
                        ambiguity_forced=will_initialize,
                        prior_ambiguity=prior_ambiguity,
                        note=note,
                    )

                if not use_code and not OutlierOptParam.KEEP_PHASE_WHEN_CODE_OUT:
                    if debug_record is not None:
                        debug_record.code_used = False
                        debug_record.phase_used = False
                        debug_record.rejected = True
                        if not debug_record.note:
                            debug_record.note = "measurement dropped"
                        debug_records.append(debug_record)
                    continue

                candidate_infos.append(
                    {
                        "scid": scid,
                        "channel": ch,
                        "code_std": code_std,
                        "phase_std": phase_std,
                        "use_code": use_code,
                        "will_initialize": will_initialize,
                        "debug_record": debug_record,
                    }
                )
                if debug_record is not None:
                    debug_record.code_used = use_code
                    debug_record.phase_used = True
                    if not use_code:
                        debug_record.note = (
                            f"{debug_record.note}; phase only"
                            if debug_record.note
                            else "code rejected, phase kept"
                        )
                    debug_records.append(debug_record)
                if use_code:
                    total_code_sat_count += 1

            if not candidate_infos:
                continue

            pending_batches.append(
                {
                    "signal_type": signal_type,
                    "pivot_ch": pivot_ch,
                    "pivot_id": pivot_id,
                    "pivot_code_std": pivot_code_std,
                    "pivot_phase_std": pivot_phase_std,
                    "force_new": force_new_ambiguity,
                    "candidates": candidate_infos,
                }
            )

        if total_code_sat_count < GnssParameters.MIN_NUM_DD_SATS_FOR_SOL:
            self._force_new_ambiguity_next_epoch = True
            self._current_dd_sat_ids.clear()
            emit_debug(skipped=True)
            return

        self._force_new_ambiguity_next_epoch = False
        prior_noise = gtsam.noiseModel.Isotropic.Sigma(1, 1e7)
        prior_added: Set[int] = set()

        for batch in pending_batches:
            pivot_ch = batch["pivot_ch"]
            signal_type = batch["signal_type"]
            pivot_code_std = batch["pivot_code_std"]
            pivot_phase_std = batch["pivot_phase_std"]
            force_new = batch["force_new"]

            meas_list: List[GnssMeasurementChannel] = []
            amb_keys: List[int] = []
            code_mask: List[bool] = []
            phase_mask: List[bool] = []
            code_std_list: List[float] = []
            phase_std_list: List[float] = []

            for cand in batch["candidates"]:
                scid = cand["scid"]
                ch = cand["channel"]
                code_std = cand["code_std"]
                phase_std = cand["phase_std"]
                use_code = cand["use_code"]
                debug_record = cand["debug_record"]

                (
                    amb_key,
                    was_new,
                    prior_val,
                    prev_amb_key,
                ) = self._resolve_ambiguity_key(
                    signal_type,
                    ch,
                    pivot_ch,
                    rel_time,
                    force_new,
                    values,
                )
                if amb_key is None:
                    raise RuntimeError("Failed to resolve ambiguity key")

                if debug_record is not None:
                    debug_record.ambiguity_forced = was_new
                    if prior_val is not None:
                        debug_record.prior_ambiguity = prior_val

                ch.sigma_code_m = code_std
                ch.sigma_phase_m = phase_std

                meas_list.append(ch)
                amb_keys.append(amb_key)
                cand["amb_key"] = amb_key
                cand["prev_amb_key"] = prev_amb_key
                code_mask.append(use_code)
                phase_mask.append(True)
                code_std_list.append(code_std)
                phase_std_list.append(phase_std)

            if not meas_list:
                continue

            n = len(meas_list)
            code_active_idx = [idx for idx, use in enumerate(code_mask) if use]
            phase_active_idx = [idx for idx, use in enumerate(phase_mask) if use]
            if not code_active_idx and not phase_active_idx:
                continue

            if code_active_idx:
                self._current_dd_sat_ids.update(
                    utils.unique_satellite_ids(
                        [meas_list[idx] for idx in code_active_idx]
                    )
                )

            for cand in batch["candidates"]:
                amb_key = cand["amb_key"] if "amb_key" in cand else None
                prev_amb_key = cand.get("prev_amb_key")
                if amb_key is None:
                    continue
                if (
                    GnssParameters.amb_propagation_mode
                    == AmbPropagationMode.RANDOM_WALK
                    and prev_amb_key is not None
                ):
                    graph.push_back(
                        gtsam.BetweenFactorDouble(
                            prev_amb_key,
                            amb_key,
                            0.0,
                            GnssParameters.amb_random_walk_noise,
                        )
                    )
                    continue
                if amb_key in prior_added:
                    continue
                if values.exists(amb_key):
                    prior_val = values.atDouble(amb_key)
                elif self._latest_estimate is not None and self._latest_estimate.exists(
                    amb_key
                ):
                    prior_val = self._latest_estimate.atDouble(amb_key)
                else:
                    prior_val = 0.0
                graph.push_back(
                    gtsam.PriorFactorDouble(amb_key, prior_val, prior_noise)
                )
                prior_added.add(amb_key)

            amb_vals_log = []
            for amb_key in amb_keys:
                if values.exists(amb_key):
                    amb_vals_log.append(values.atDouble(amb_key))
                elif self._latest_estimate is not None and self._latest_estimate.exists(
                    amb_key
                ):
                    amb_vals_log.append(self._latest_estimate.atDouble(amb_key))
                else:
                    amb_vals_log.append(0.0)

            residual_full, _, _ = (
                meas_error_models.double_difference_batch_residual_and_jacobian(
                    pivot_ch,
                    meas_list,
                    pose_guess,
                    lever_arm_b,
                    amb_vals=amb_vals_log,
                    compute_jacobian=False,
                )
            )

            for idx, cand in enumerate(batch["candidates"]):
                debug_record = cand["debug_record"]
                if debug_record is None:
                    continue
                debug_record.code_residual = float(residual_full[idx])
                debug_record.phase_residual = float(residual_full[n + idx])

            code_std_arr = np.asarray(code_std_list, dtype=float)
            phase_std_arr = np.asarray(phase_std_list, dtype=float)
            code_cov_full = np.full((n, n), pivot_code_std**2)
            phase_cov_full = np.full((n, n), pivot_phase_std**2)
            np.fill_diagonal(code_cov_full, pivot_code_std**2 + code_std_arr**2)
            np.fill_diagonal(phase_cov_full, pivot_phase_std**2 + phase_std_arr**2)

            row_indices = code_active_idx + [n + idx for idx in phase_active_idx]
            if not row_indices:
                continue

            dim_code = len(code_active_idx)
            dim_phase = len(phase_active_idx)
            total_dim = dim_code + dim_phase

            noise_cov = np.zeros((total_dim, total_dim))
            if dim_code:
                code_cov_active = code_cov_full[
                    np.ix_(code_active_idx, code_active_idx)
                ]
                noise_cov[:dim_code, :dim_code] = code_cov_active
            if dim_phase:
                phase_cov_active = phase_cov_full[
                    np.ix_(phase_active_idx, phase_active_idx)
                ]
                start = dim_code
                noise_cov[start:, start:] = phase_cov_active

            noise = gtsam.noiseModel.Gaussian.Covariance(noise_cov)
            huber = gtsam.noiseModel.mEstimator.Huber.Create(GnssParameters.HUBER_K)
            noise = gtsam.noiseModel.Robust.Create(huber, noise)

            meas_batch = BatchMeasFactor(
                pivot=pivot_ch,
                meas=meas_list,
                code_mask=code_mask,
                phase_mask=phase_mask,
            )
            keys = [pose_key] + amb_keys
            factor = gtsam.CustomFactor(
                noise,
                gtsam.KeyVector(keys),
                partial(RtkInsFgo.error_meas_batch, meas_batch),
            )
            graph.push_back(factor)

        emit_debug(skipped=False)

    def _build_epoch_update(
        self,
        epoch: GpsTime,
        pose_key: int,
        vel_key: int,
        bias_key: int,
        pose_guess: gtsam.Pose3,
        velocity_guess: np.ndarray,
        preint: gtsam.PreintegratedImuMeasurements,
        channels: Dict[SignalChannelId, GnssMeasurementChannel],
        rel_time: float,
    ) -> EpochUpdateData:
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()

        values.insert(pose_key, pose_guess)
        values.insert(vel_key, velocity_guess)
        values.insert(bias_key, self.prev_bias)

        graph.push_back(
            gtsam.ImuFactor(
                POSE_KEY(self.current_epoch_idx - 1),
                VEL_KEY(self.current_epoch_idx - 1),
                pose_key,
                vel_key,
                BIAS_KEY(self.current_epoch_idx - 1),
                preint,
            )
        )
        graph.push_back(
            gtsam.BetweenFactorConstantBias(
                BIAS_KEY(self.current_epoch_idx - 1),
                bias_key,
                gtsam.imuBias.ConstantBias(),
                self.bias_between_noise,
            )
        )

        self._current_dd_sat_ids.clear()
        self._add_gnss_factors(
            epoch,
            pose_key,
            channels,
            pose_guess,
            velocity_guess,
            graph,
            values,
            pose_cov_prior=self.prev_pose_cov,
        )

        return EpochUpdateData(
            factors=graph,
            values=values,
            pose_key=pose_key,
            vel_key=vel_key,
            bias_key=bias_key,
            rel_time=rel_time,
            dd_sat_ids=set(self._current_dd_sat_ids),
            epoch_idx=self.current_epoch_idx,
        )

    @staticmethod
    def _assert_all_values_connected(
        graph: gtsam.NonlinearFactorGraph, values: gtsam.Values
    ) -> None:
        """Ensure every variable supplied in ``values`` appears in at least one factor."""

        factor_key_sets = []
        for idx in range(graph.size()):
            factor = graph.at(idx)
            if factor is None:
                continue
            factor_key_sets.append(set(factor.keys()))
        all_factor_keys: Set[int] = (
            set().union(*factor_key_sets) if factor_key_sets else set()
        )

        dangling = [key for key in values.keys() if key not in all_factor_keys]
        if dangling:
            names = ", ".join(
                f"{key} ({gtsam.DefaultKeyFormatter(key)})" for key in dangling
            )
            raise RuntimeError(
                f"Dangling variables detected with no associated factor: {names}"
            )

    @staticmethod
    def _format_key(key: int) -> str:
        return gtsam.DefaultKeyFormatter(key)

    @staticmethod
    def _factor_key_sets(graph: gtsam.NonlinearFactorGraph) -> List[List[str]]:
        key_sets: List[List[str]] = []
        for idx in range(graph.size()):
            factor = graph.at(idx)
            if factor is None:
                continue
            key_sets.append([gtsam.DefaultKeyFormatter(k) for k in factor.keys()])
        return key_sets

    @staticmethod
    def _pose_indices_from_graph(graph: gtsam.NonlinearFactorGraph) -> Set[int]:
        pose_indices: Set[int] = set()
        for idx in range(graph.size()):
            factor = graph.at(idx)
            if factor is None:
                continue
            for key in factor.keys():
                if gtsam.symbolChr(key) == ord("x"):
                    pose_indices.add(gtsam.symbolIndex(key))
        return pose_indices

    def _append_update(self, update: EpochUpdateData) -> None:
        for idx in range(update.factors.size()):
            factor = update.factors.at(idx)
            if factor is not None:
                self.graph.push_back(factor)

    def _merge_initial_values(self, new_values: gtsam.Values) -> None:
        for key in new_values.keys():
            if self.initial_values.exists(key):
                continue
            self._copy_value(key, new_values, self.initial_values)

    def _copy_value(self, key: int, source: gtsam.Values, target: gtsam.Values) -> None:
        if target.exists(key):
            return
        symbol = chr(gtsam.symbolChr(key))
        if symbol == "x":
            target.insert(key, source.atPose3(key))
        elif symbol == "v":
            target.insert(key, source.atVector(key))
        elif symbol == "b":
            target.insert(key, source.atConstantBias(key))
        elif symbol == "a":
            target.insert(key, source.atDouble(key))
        else:
            raise ValueError(f"Unsupported key symbol: {symbol}")

    def _prepare_initial_guess(self, new_values: gtsam.Values) -> gtsam.Values:
        initial = gtsam.Values()
        for key in self.initial_values.keys():
            self._copy_value(key, self.initial_values, initial)
        for key in new_values.keys():
            self._copy_value(key, new_values, initial)
        return initial

    def _ensure_isam(self) -> None:
        if self.isam is not None:
            return
        params = gtsam.ISAM2Params()
        params.setFactorization("CHOLESKY")
        params.relinearizeSkip = max(1, int(self.isam_relinearize_skip))
        self.isam = gtsam.ISAM2(params)

    def _optimize_with_lm(self, initial_guess: gtsam.Values) -> gtsam.Values:
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph,
            initial_guess,
            params,
        )
        return optimizer.optimize()

    def _prime_imu_index(self, epoch: GpsTime) -> None:
        if not self.imu_data_list:
            return

        epoch_time = epoch.gps_timestamp + 1e-9
        while self._imu_idx < len(self.imu_data_list):
            sample = self.imu_data_list[self._imu_idx]
            if sample.epoch.gps_timestamp > epoch_time:
                break
            self._last_imu_sample = sample
            self._imu_idx += 1

    def _update_pivot(
        self,
        signal_type: SignalType,
        pivot_id: SignalChannelId,
        pivot_ch: GnssMeasurementChannel,
    ) -> bool:
        prev_pivot = self.current_pivot_by_type.get(signal_type)
        pivot_changed = prev_pivot is None or prev_pivot != pivot_id
        if (
            pivot_ch.cycle_slip_status is not None
            and pivot_ch.cycle_slip_status != CycleSlipType.NOT_DETECTED
        ):
            pivot_changed = True
        if pivot_changed:
            keys_to_delete = [
                scid
                for scid, state in self.ambiguities.items()
                if state.signal_type == signal_type
            ]
            for scid in keys_to_delete:
                self.ambiguities.pop(scid, None)
        self.current_pivot_by_type[signal_type] = pivot_id
        return pivot_changed

    def _resolve_ambiguity_key(
        self,
        signal_type: SignalType,
        ch: GnssMeasurementChannel,
        pivot_ch: GnssMeasurementChannel,
        rel_time: float,
        force_new: bool,
        values: gtsam.Values,
    ) -> Tuple[int, bool, Optional[float], Optional[int]]:
        scid = ch.signal_id
        state = self.ambiguities.get(scid)
        slip_not_detected = (
            ch.cycle_slip_status is not None
            and ch.cycle_slip_status == CycleSlipType.NOT_DETECTED
        )
        is_stale = (
            state is not None and rel_time - state.last_update_sec > self.window_size_s
        )

        prev_key_for_between: Optional[int] = None

        if state is None or force_new or not slip_not_detected or is_stale:
            key = AMB_KEY(self.next_amb_idx)
            self.next_amb_idx += 1
            init_val = self._initialize_ambiguity(ch, pivot_ch)
            values.insert(key, init_val)
            self.ambiguities[scid] = AmbiguityState(
                key=key,
                signal_type=signal_type,
                pivot_id=pivot_ch.signal_id,
                last_update_sec=rel_time,
                last_epoch_idx=self.current_epoch_idx,
            )
            return key, True, None, None

        used_last_epoch = state.last_epoch_idx == self.current_epoch_idx - 1
        if not used_last_epoch:
            key = AMB_KEY(self.next_amb_idx)
            self.next_amb_idx += 1
            init_val = self._initialize_ambiguity(ch, pivot_ch)
            values.insert(key, init_val)
            self.ambiguities[scid] = AmbiguityState(
                key=key,
                signal_type=signal_type,
                pivot_id=pivot_ch.signal_id,
                last_update_sec=rel_time,
                last_epoch_idx=self.current_epoch_idx,
            )
            return key, True, None, None

        prior_val = None
        if values.exists(state.key):
            prior_val = values.atDouble(state.key)
        elif self._latest_estimate is not None and self._latest_estimate.exists(
            state.key
        ):
            prior_val = self._latest_estimate.atDouble(state.key)

        if GnssParameters.amb_propagation_mode == AmbPropagationMode.RANDOM_WALK:
            prev_key_for_between = state.key
            key = AMB_KEY(self.next_amb_idx)
            self.next_amb_idx += 1
            init_val = prior_val if prior_val is not None else 0.0
            values.insert(key, init_val)
            self.ambiguities[scid].key = key
            self.ambiguities[scid].last_update_sec = rel_time
            self.ambiguities[scid].last_epoch_idx = self.current_epoch_idx
            return key, False, prior_val, prev_key_for_between

        self.ambiguities[scid].last_update_sec = rel_time
        self.ambiguities[scid].last_epoch_idx = self.current_epoch_idx
        return state.key, False, prior_val, None

    def _measurement_std(
        self, signal_type: SignalType, elev_deg: float
    ) -> Tuple[float, float]:
        params = GNSS_ELEV_MODEL_PARAMS.get(signal_type)
        if params is None:
            raise ValueError(f"No elevation model sigma parameters for {signal_type}")
        code_var = elevation_based_noise_var(elev_deg, params[0], params[1])
        phase_var = elevation_based_noise_var(
            elev_deg, phase_sigma_a, 2.0 * phase_sigma_b
        )
        return float(np.sqrt(code_var)), float(np.sqrt(phase_var))

    def _initialize_ambiguity(
        self,
        ch: GnssMeasurementChannel,
        pivot_ch: GnssMeasurementChannel,
    ) -> float:
        if ch.wavelength_m is None or ch.wavelength_m == 0:
            raise ValueError("Invalid wavelength for ambiguity initialization")
        code_diff = ch.code_m - pivot_ch.code_m
        phase_diff = ch.phase_m - pivot_ch.phase_m
        return (phase_diff - code_diff) / ch.wavelength_m

    def _integrate_imu(
        self, target_epoch: GpsTime
    ) -> gtsam.PreintegratedImuMeasurements:
        preint = gtsam.PreintegratedImuMeasurements(self.preint_params, self.prev_bias)
        if not self.imu_data_list or self.prev_epoch_time is None:
            return preint

        target_time = target_epoch.gps_timestamp
        last_time = self.prev_epoch_time.gps_timestamp
        idx = self._imu_idx
        last_sample = self._last_imu_sample

        while idx < len(self.imu_data_list):
            sample = self.imu_data_list[idx]
            sample_time = sample.epoch.gps_timestamp
            if sample_time > target_time + 1e-9:
                break

            dt = sample_time - last_time
            if dt > 0:
                preint.integrateMeasurement(sample.acc_mps2, sample.gyro_rps, dt)
                last_time = sample_time

            last_sample = sample
            idx += 1

        self._imu_idx = idx
        self._last_imu_sample = last_sample

        if last_time < target_time and last_sample is not None:
            dt = target_time - last_time
            if dt > 0:
                preint.integrateMeasurement(
                    last_sample.acc_mps2, last_sample.gyro_rps, dt
                )

        return preint

    def _antenna_ecef(self, pose: gtsam.Pose3) -> np.ndarray:
        enu_to_ecef = BASE_ECEF_TO_ENU_ROT_MAT().T
        pos_enu = pose.translation()
        rot = pose.rotation().matrix()
        return enu_to_ecef @ (pos_enu + rot @ self._lever_arm()) + np.asarray(
            BASE_POS_ECEF
        )

    def _extract_epoch_result(
        self,
        epoch: GpsTime,
        pose_key: int,
        vel_key: int,
        bias_key: int,
        estimate: gtsam.Values,
    ) -> EpochLogEntry:
        pose = estimate.atPose3(pose_key)
        vel = estimate.atVector(vel_key)
        try:
            marginals = gtsam.Marginals(self.graph, estimate)
            cov = marginals.marginalCovariance(pose_key)
            vel_cov = marginals.marginalCovariance(vel_key)
        except (RuntimeError, IndexError) as exc:
            text = str(exc)
            if (
                isinstance(exc, RuntimeError)
                and "IndeterminantLinearSystemException" in text
            ):
                cov = np.full((6, 6), np.nan)
                vel_cov = np.full((3, 3), np.nan)
            elif "Requested the BayesTree clique" in text or isinstance(
                exc, IndexError
            ):
                cov = np.full((6, 6), np.nan)
                vel_cov = np.full((3, 3), np.nan)
            else:
                raise
        entry = EpochLogEntry(
            epoch=epoch,
            pose_key=pose_key,
            vel_key=vel_key,
            bias_key=bias_key,
            pose=pose,
            pose_enu_m=pose.translation(),
            vel_enu_mps=vel,
            pose_cov_6x6=cov,
            vel_cov_3x3=vel_cov,
        )
        self.result_log.append(entry)
        return entry

    def _make_preintegration_params(self) -> gtsam.PreintegrationParams:
        params = gtsam.PreintegrationParams.MakeSharedU(self.imu_params.gravity)
        params.setAccelerometerCovariance(np.diag(self.imu_params.acc_noise_std**2))
        params.setGyroscopeCovariance(np.diag(self.imu_params.gyro_noise_std**2))
        params.setIntegrationCovariance(np.eye(3) * 1e-4)
        return params

    def _relative_time(self, epoch: GpsTime) -> float:
        if self.smoother_init_time is None:
            return 0.0
        return float(epoch - self.smoother_init_time)

    @staticmethod
    def _lever_arm() -> np.ndarray:
        return RtkInsFgo._lever_arm_b
