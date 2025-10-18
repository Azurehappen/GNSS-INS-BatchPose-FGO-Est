"""GNSS/INS factor graph optimizer utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import gtsam
import numpy as np
import pymap3d as pm

from constants.gnss_constants import CycleSlipType
from constants.parameters import (
    AmbiguityMode,
    BASE_ECEF_TO_ENU_ROT_MAT,
    BASE_POS_ECEF,
    GNSS_ELEV_MODEL_PARAMS,
    GnssParameters,
    INIT_ACC_BIAS_STD,
    INIT_ATTITUDE_STD_RAD,
    INIT_GYRO_BIAS_STD,
    INIT_VEL_ENU,
    INIT_YAW_RAD,
    phase_sigma_a,
    phase_sigma_b,
)
from fgo_solver import utils, meas_error_models
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


@dataclass
class AmbiguityState:
    """Tracks metadata for a carrier-phase ambiguity estimate."""

    key: int
    signal_type: SignalType
    pivot_id: SignalChannelId
    last_update_sec: float


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


class RtkInsFgo:
    """Fixed-lag GNSS/INS factor graph optimizer built on GTSAM."""

    _lever_arm_b = np.zeros(3)

    def __init__(
        self,
        imu_params: Any,
        imu_data_list: Sequence[ImuSingleEpoch],
        *,
        window_size_s: float = 5.0,
        show_progress: bool = False,
    ) -> None:
        self.imu_params = imu_params
        self.window_size_s = window_size_s
        self.t_imu_to_ant_in_b = np.asarray(imu_params.t_imu_to_ant_in_b)
        RtkInsFgo._lever_arm_b = self.t_imu_to_ant_in_b

        self.preint_params = self._make_preintegration_params()
        self.bias_between_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate((imu_params.acc_bias_std, imu_params.gyro_bias_std))
        )

        self.smoother = gtsam.BatchFixedLagSmoother(window_size_s)
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        self.new_timestamps: Dict[int, float] = {}

        self.base_time: Optional[GpsTime] = None
        self.prev_epoch_time: Optional[GpsTime] = None

        self.current_epoch_idx = -1
        self.prev_pose: Optional[gtsam.Pose3] = None
        self.prev_velocity = np.zeros(3)
        self.prev_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

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
        self.smoother = gtsam.BatchFixedLagSmoother(self.window_size_s)
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        self.new_timestamps.clear()

        self.base_time = None
        self.prev_epoch_time = None
        self.current_epoch_idx = -1
        self.prev_pose = None
        self.prev_velocity = np.zeros(3)
        self.prev_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

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

        if self.base_time is None:
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

        self.new_values.insert(pose_key, predicted_state.pose())
        self.new_values.insert(vel_key, predicted_state.velocity())
        self.new_values.insert(bias_key, self.prev_bias)

        self.new_timestamps[pose_key] = rel_time
        self.new_timestamps[vel_key] = rel_time
        self.new_timestamps[bias_key] = rel_time

        self.new_factors.push_back(
            gtsam.ImuFactor(
                POSE_KEY(self.current_epoch_idx - 1),
                VEL_KEY(self.current_epoch_idx - 1),
                pose_key,
                vel_key,
                BIAS_KEY(self.current_epoch_idx - 1),
                preint,
            )
        )
        self.new_factors.push_back(
            gtsam.BetweenFactorConstantBias(
                BIAS_KEY(self.current_epoch_idx - 1),
                bias_key,
                gtsam.imuBias.ConstantBias(),
                self.bias_between_noise,
            )
        )

        self._add_gnss_factors(epoch, pose_key, channels, predicted_state.pose())
        self._update_smoother()
        estimate = self.smoother.calculateEstimate()

        entry = self._extract_epoch_result(epoch, pose_key, vel_key, bias_key, estimate)

        self.prev_epoch_time = epoch
        self.prev_pose = estimate.atPose3(pose_key)
        self.prev_velocity = estimate.atVector(vel_key)
        self.prev_bias = estimate.atConstantBias(bias_key)

        return entry

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
        """Process a range of GNSS epochs using the internal smoother."""

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
            progress_bar = tqdm(epoch_subset, desc="GNSS-INS FGO", unit="epoch")
            iterator = progress_bar
        else:
            iterator = epoch_subset

        try:
            for epoch in iterator:
                channels = rover_obs.get(epoch)
                if not channels:
                    continue
                self.process_epoch(epoch, channels)
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
        if not meas_list:
            return np.zeros(0)

        pose = values.atPose3(pos_key)
        lever_b = RtkInsFgo._lever_arm()

        amb_keys = [this.keys()[i + 1] for i in range(len(meas_list))]
        amb_vals = [values.atDouble(k) if values.exists(k) else 0.0 for k in amb_keys]

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

        if jacobians is not None:
            if J_pose is None or J_amb_list is None:
                raise RuntimeError("Analytic Jacobian computation failed")
            if jacobians[0].size == 0:
                jacobians[0] = J_pose
            else:
                jacobians[0][:] = J_pose
            for i, J_amb in enumerate(J_amb_list):
                if jacobians[i + 1].size == 0:
                    jacobians[i + 1] = J_amb
                else:
                    jacobians[i + 1][:] = J_amb

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

        self.base_time = epoch
        self.prev_epoch_time = epoch
        self.current_epoch_idx = 0

        initial_ecef, _ = self._solve_wls_position(channels)
        initial_enu = compute_world_frame_coord_from_ecef(initial_ecef)

        pose_key = POSE_KEY(0)
        vel_key = VEL_KEY(0)
        bias_key = BIAS_KEY(0)

        initial_pose = gtsam.Pose3(gtsam.Rot3.Ypr(INIT_YAW_RAD, 0.0, 0.0), initial_enu)
        initial_velocity = np.asarray(INIT_VEL_ENU, dtype=float)
        initial_bias = gtsam.imuBias.ConstantBias(np.zeros(3), np.zeros(3))

        pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate((INIT_ATTITUDE_STD_RAD, np.full(3, 40.0)))
        )
        velocity_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 5.0)
        bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.concatenate((INIT_ACC_BIAS_STD, INIT_GYRO_BIAS_STD))
        )

        self.new_factors.push_back(
            gtsam.PriorFactorPose3(pose_key, initial_pose, pose_prior_noise)
        )
        self.new_factors.push_back(
            gtsam.PriorFactorVector(vel_key, initial_velocity, velocity_prior_noise)
        )
        self.new_factors.push_back(
            gtsam.PriorFactorConstantBias(bias_key, initial_bias, bias_prior_noise)
        )

        self.new_values.insert(pose_key, initial_pose)
        self.new_values.insert(vel_key, initial_velocity)
        self.new_values.insert(bias_key, initial_bias)

        rel_time = 0.0
        self.new_timestamps[pose_key] = rel_time
        self.new_timestamps[vel_key] = rel_time
        self.new_timestamps[bias_key] = rel_time

        self._add_gnss_factors(epoch, pose_key, channels, initial_pose)
        self._prime_imu_index(epoch)
        self._update_smoother()
        estimate = self.smoother.calculateEstimate()

        self.prev_pose = estimate.atPose3(pose_key)
        self.prev_velocity = estimate.atVector(vel_key)
        self.prev_bias = estimate.atConstantBias(bias_key)

        entry = self._extract_epoch_result(epoch, pose_key, vel_key, bias_key, estimate)
        return entry

    def _add_gnss_factors(
        self,
        epoch: GpsTime,
        pose_key: int,
        channels: Dict[SignalChannelId, GnssMeasurementChannel],
        pose_guess: gtsam.Pose3,
    ) -> None:
        if not channels:
            return

        receiver_ecef = self._antenna_ecef(pose_guess)
        lat_rad, lon_rad, _ = pm.ecef2geodetic(*receiver_ecef, deg=False)
        ecef_to_enu = compute_ecef_enu_rot_mat(lat_rad, lon_rad)
        rel_time = self._relative_time(epoch)

        grouped: Dict[SignalType, Dict[SignalChannelId, GnssMeasurementChannel]] = (
            defaultdict(dict)
        )
        for scid, ch in channels.items():
            if ch.sat_pos_ecef_m is None:
                continue
            elev_deg, _ = compute_sat_elev_az(
                ecef_to_enu, receiver_ecef, ch.sat_pos_ecef_m
            )
            ch.elevation_deg = elev_deg
            if elev_deg < GnssParameters.ELEVATION_MASK_DEG:
                continue
            grouped[scid.signal_type][scid] = ch

        for signal_type, signal_channels in grouped.items():
            if len(signal_channels) < 3:
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

            meas_list: List[GnssMeasurementChannel] = []
            amb_keys: List[int] = []

            pivot_code_std, pivot_phase_std = self._measurement_std(
                signal_type, pivot_ch.elevation_deg
            )
            pivot_ch.sigma_code_m = pivot_code_std
            pivot_ch.sigma_phase_m = pivot_phase_std

            force_new_ambiguity = (
                pivot_changed
                or GnssParameters.ambiguity_mode == AmbiguityMode.INSTANTANEOUS
            )

            for scid, ch in signal_channels.items():
                if scid == pivot_id:
                    continue
                amb_key = self._resolve_ambiguity_key(
                    signal_type,
                    ch,
                    pivot_ch,
                    rel_time,
                    force_new_ambiguity,
                )
                if amb_key is None:
                    raise RuntimeError("Failed to resolve ambiguity key")

                code_std, phase_std = self._measurement_std(
                    signal_type, ch.elevation_deg
                )
                ch.sigma_code_m = code_std
                ch.sigma_phase_m = phase_std
                meas_list.append(ch)
                amb_keys.append(amb_key)

            if not meas_list:
                continue

            n = len(meas_list)
            code_std_arr = np.array([ch.sigma_code_m for ch in meas_list])
            phase_std_arr = np.array([ch.sigma_phase_m for ch in meas_list])

            code_cov = np.full((n, n), pivot_code_std**2)
            phase_cov = np.full((n, n), pivot_phase_std**2)
            np.fill_diagonal(code_cov, pivot_code_std**2 + code_std_arr**2)
            np.fill_diagonal(phase_cov, pivot_phase_std**2 + phase_std_arr**2)

            noise_cov = np.zeros((2 * n, 2 * n))
            noise_cov[:n, :n] = code_cov
            noise_cov[n:, n:] = phase_cov
            noise = gtsam.noiseModel.Gaussian.Covariance(noise_cov)

            meas_batch = BatchMeasFactor(pivot=pivot_ch, meas=meas_list)
            keys = [pose_key] + amb_keys
            factor = gtsam.CustomFactor(
                noise,
                gtsam.KeyVector(keys),
                partial(RtkInsFgo.error_meas_batch, meas_batch),
            )
            self.new_factors.push_back(factor)

            for amb_key in amb_keys:
                self.new_timestamps[amb_key] = rel_time

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
    ) -> Optional[int]:
        scid = ch.signal_id
        state = self.ambiguities.get(scid)
        slip_not_detected = (
            ch.cycle_slip_status is not None
            and ch.cycle_slip_status == CycleSlipType.NOT_DETECTED
        )
        is_stale = (
            state is not None and rel_time - state.last_update_sec > self.window_size_s
        )

        if state is None or force_new or not slip_not_detected or is_stale:
            key = AMB_KEY(self.next_amb_idx)
            self.next_amb_idx += 1
            init_val = self._initialize_ambiguity(ch, pivot_ch)
            self.new_values.insert(key, init_val)
            self.ambiguities[scid] = AmbiguityState(
                key=key,
                signal_type=signal_type,
                pivot_id=pivot_ch.signal_id,
                last_update_sec=rel_time,
            )
            return key

        self.ambiguities[scid].last_update_sec = rel_time
        return state.key

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

    def _solve_wls_position(
        self,
        channels: Dict[SignalChannelId, GnssMeasurementChannel],
    ) -> Tuple[np.ndarray, float]:
        est = np.asarray(BASE_POS_ECEF, dtype=float)
        clock_bias = 0.0
        for _ in range(10):
            A = []
            b = []
            for ch in channels.values():
                if ch.sat_pos_ecef_m is None or ch.code_m is None:
                    continue
                los = ch.sat_pos_ecef_m - est
                range_est = np.linalg.norm(los)
                if range_est < 1.0:
                    continue
                unit = los / range_est
                A.append(np.append(-unit, 1.0))
                b.append(ch.code_m - range_est - clock_bias)
            if len(A) < 4:
                raise ValueError("Insufficient valid satellites for WLS initialization")
            A = np.vstack(A)
            b = np.asarray(b)
            dx, *_ = np.linalg.lstsq(A, b, rcond=None)
            est += dx[:3]
            clock_bias += dx[3]
            if np.linalg.norm(dx[:3]) < 1e-3:
                break
        return est, clock_bias

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
            marginals = gtsam.Marginals(self.smoother.getFactors(), estimate)
            cov = marginals.marginalCovariance(pose_key)
        except RuntimeError as exc:
            if "IndeterminantLinearSystemException" in str(exc):
                cov = np.full((6, 6), np.nan)
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
        )
        self.result_log.append(entry)
        return entry

    def _update_smoother(self) -> None:
        self.smoother.update(self.new_factors, self.new_values, self.new_timestamps)
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.new_values = gtsam.Values()
        self.new_timestamps = {}

    def _make_preintegration_params(self) -> gtsam.PreintegrationParams:
        params = gtsam.PreintegrationParams.MakeSharedU(self.imu_params.gravity)
        params.setAccelerometerCovariance(np.diag(self.imu_params.acc_noise_std**2))
        params.setGyroscopeCovariance(np.diag(self.imu_params.gyro_noise_std**2))
        params.setIntegrationCovariance(np.eye(3) * 1e-4)
        return params

    def _relative_time(self, epoch: GpsTime) -> float:
        if self.base_time is None:
            return 0.0
        return float(epoch - self.base_time)

    @staticmethod
    def _lever_arm() -> np.ndarray:
        return RtkInsFgo._lever_arm_b
