import gtsam
import numpy as np
from typing import Any
from dataclasses import dataclass
from typing import List, Optional
from constants.parameters import BASE_POS_ECEF, BASE_ECEF_TO_NED_ROT_MAT
from gnss_utils.gnss_data_utils import GnssMeasurementChannel
from gnss_utils.satellite_utils import sagnac_correction
from fgo_solver import utils


@dataclass
class SingleMeasFactor:
    """Custom factor for single GNSS measurements."""

    def __init__(self, meas: GnssMeasurementChannel, pivot: GnssMeasurementChannel):
        self.code_m = meas.code_m
        self.pivot_code_m = pivot.code_m
        self.phase_m = meas.phase_m
        self.pivot_phase_m = pivot.phase_m
        self.sat_pos_ecef = meas.sat_pos_ecef_m
        self.pivot_sat_pos_ecef = pivot.sat_pos_ecef_m
        self.wavelength_m = meas.wavelength_m


class RtkInsFgo:

    def __init__(self, imu_params: Any):
        self.imu_params = imu_params
        global t_imu_to_ant_in_b
        t_imu_to_ant_in_b = imu_params.t_imu_to_ant_in_b

    # Define the error function for the GNSS measurement factor
    @staticmethod
    def error_meas(
        measurement: SingleMeasFactor,
        this: gtsam.CustomFactor,
        values: gtsam.Values,
        jacobians: Optional[List[np.ndarray]],
    ) -> np.ndarray:
        """
        Nonlinear error function.
        Computes the residual.

        Args:
            measurement: GNSS measurements.
            this (CustomFactor): GTSAM custom factor instance.
            values (Values): Current state estimates.
            jacobians (Optional[List[np.ndarray]]): Optional list to store Jacobian matrices.

        Returns:
            np.ndarray: Residual error array of shape (1,).
        """

        pos_key = this.keys()[0]
        dd_amb_key = this.keys()[1]  # Between satellite ambiguity key
        pos_imu_ned = values.atPose3(pos_key).translation()
        rotation_imu_to_ned = (
            values.atPose3(pos_key).rotation().matrix()
        )  # Body to World

        # Convert antenna position from NED frame (relative to base station) to ECEF frame
        ant_pos_ecef = (
            BASE_ECEF_TO_NED_ROT_MAT().T
            @ (pos_imu_ned + rotation_imu_to_ned @ t_imu_to_ant_in_b)
            + BASE_POS_ECEF
        )

        est_geo_range_i = np.linalg.norm(
            ant_pos_ecef - measurement.sat_pos_ecef
        ) + sagnac_correction(ant_pos_ecef, measurement.sat_pos_ecef)
        est_geo_range_p = np.linalg.norm(
            ant_pos_ecef - measurement.pivot_sat_pos_ecef
        ) + sagnac_correction(ant_pos_ecef, measurement.pivot_sat_pos_ecef)

        error_code = (
            measurement.code_m
            - est_geo_range_i
            - (measurement.pivot_code_m - est_geo_range_p)
        )
        error_phase = (
            measurement.phase_m
            - est_geo_range_i
            - (measurement.pivot_phase_m - est_geo_range_p)
            + measurement.wavelength_m * values.atConstantBias(dd_amb_key)
        )

        # Line-of-sight vector from satellite to receiver
        range_i = np.linalg.norm(ant_pos_ecef - measurement.sat_pos_ecef)
        range_p = np.linalg.norm(ant_pos_ecef - measurement.pivot_sat_pos_ecef)
        los_i = (ant_pos_ecef - measurement.sat_pos_ecef) / range_i
        los_p = (ant_pos_ecef - measurement.pivot_sat_pos_ecef) / range_p

        # Package residuals (code and carrier-phase in meters)
        residual = np.array([error_code, error_phase], dtype=float)

        # Optionally provide Jacobians
        if jacobians is not None:
            # Common geometric term: derivative w.r.t. receiver ECEF position
            # for the double-differenced range.
            los_diff = (los_i - los_p).reshape(1, 3)  # 1x3

            # Map to NED position state using base rotation (NED->ECEF is R^T)
            J_pos = (los_diff @ BASE_ECEF_TO_NED_ROT_MAT().T).astype(float)  # 1x3

            # Attitude component from lever-arm contribution.
            # d p_ecef / d theta ~= R_en^T * [ R_nb * t_ib ]_x
            lever_in_ned = rotation_imu_to_ned @ t_imu_to_ant_in_b
            J_theta_map = BASE_ECEF_TO_NED_ROT_MAT().T @ utils.skew_symmetric(
                lever_in_ned
            )  # 3x3
            J_rot = (los_diff @ J_theta_map).astype(float)  # 1x3

            # Compose Pose3 Jacobian in GTSAM local coordinates order [rot, trans]
            J_pose_row = np.hstack([J_rot, J_pos])  # 1x6
            J_pose = np.vstack([J_pose_row, J_pose_row]).astype(float)  # 2x6

            # Ambiguity Jacobian: only affects phase, coefficient = wavelength
            J_amb = np.array([[0.0], [measurement.wavelength_m]], dtype=float)  # 2x1

            # Assign to output containers following `this.keys()` ordering
            jacobians[0][:] = J_pose
            jacobians[1][:] = J_amb

        return residual
