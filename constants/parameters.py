from dataclasses import dataclass, field
from enum import Enum
import gtsam
import numpy as np
import pymap3d as pm
from gnss_utils.gnss_dataclass import SignalType
from constants.gnss_constants import Constellation
import constants.common_utils as comm_utils


class AmbiguityMode(Enum):
    CONTINUOUS = 1
    INSTANTANEOUS = 2


class AmbPropagationMode(Enum):
    CONSTANT = 1
    RANDOM_WALK = 2


@dataclass
class GnssParameters:
    ELEVATION_MASK_DEG: float = 15.0  # Minimum elevation angle in degrees
    CNO_THRESHOLD: float = (
        20.0  # Minimum C/N0 in dB-Hz for a satellite to be considered valid
    )
    PIVOT_SAT_ELEVATION_MASK_DEG: float = (
        45.0  # Minimum elevation angle for pivot satellite
    )
    PIVOT_SAT_CNO_THRESHOLD: float = 30.0  # Minimum C/N0 in dB-Hz for pivot satellite
    CYCLE_SLIP_THRESHOLD_M: float = 0.1  # Geometry-free cycle slip threshold

    MIN_NUM_SIGNALS_FOR_DD: int = (
        3  # Minimum number of common signals for double-difference
    )

    MIN_NUM_DD_SATS_FOR_SOL: int = (
        5  # Minimum number of double-difference satellites to enable a GNSS solution
    )

    MIN_NUM_DD_SATS_FOR_OPENSKY: int = (
        14  # Minimum number of double-difference satellites for open-sky
    )
    HUBER_K: float = 1.5  # Breakpoint for robust GNSS factors

    enable_gps: bool = True
    enable_galileo: bool = True
    enable_glonass: bool = True
    enable_beidou: bool = True

    ambiguity_mode: AmbiguityMode = AmbiguityMode.CONTINUOUS
    amb_propagation_mode: AmbPropagationMode = AmbPropagationMode.CONSTANT
    amb_random_walk_noise = gtsam.noiseModel.Isotropic.Sigma(
        1, 0.1
    )  # Ambiguity random walk noise (cycles/sqrt(s))


phase_sigma_a = 0.003  # meters
phase_sigma_b = 0.003  # meters


@dataclass
class OutlierOptParam:
    RESIDUAL_GATE_THRESHOLD: float = 1.5  # Neyman–Pearson gate multiplier
    KEEP_PHASE_WHEN_CODE_OUT: bool = (
        False  # Keep phase measurements when code is outlier
    )


SIGNAL_TO_EXCLUDED_FROM_SOL: set[SignalType] = {
    # GPS L2W is a half-wavelength signal (~12 cm), which is too small.
    SignalType(Constellation.GPS, 2, "W"),
}


# SignalType -> (fact_a, fact_b)
GNSS_ELEV_MODEL_PARAMS: dict[SignalType, tuple[float, float]] = {
    SignalType(Constellation.GPS, 1, "C"): (300 * phase_sigma_a, 650 * phase_sigma_b),
    SignalType(Constellation.GPS, 2, "W"): (500 * phase_sigma_a, 650 * phase_sigma_b),
    SignalType(Constellation.GLO, 1, "C"): (700 * phase_sigma_a, 800 * phase_sigma_b),
    SignalType(Constellation.GLO, 2, "C"): (700 * phase_sigma_a, 800 * phase_sigma_b),
    SignalType(Constellation.GAL, 1, "C"): (500 * phase_sigma_a, 650 * phase_sigma_b),
    SignalType(Constellation.GAL, 7, "Q"): (500 * phase_sigma_a, 800 * phase_sigma_b),
    SignalType(Constellation.BDS, 2, "I"): (300 * phase_sigma_a, 450 * phase_sigma_b),
}

RINEX_OBS_CHANNEL_TO_USE: dict[str, set[str]] = {
    "G": {"1C", "2W"},
    "R": {"1C", "2C"},
    "E": {"1C", "7Q"},
    "C": {"2I"},
}

BASE_POS_ECEF = [
    -742080.4125,
    -5462031.7412,
    3198339.6909,
]  # Base station position in ECEF coordinates (meters)

# Lazy-loaded constants - computed only once when first accessed
_base_pos_lla_rad = None
_ecef_to_enu_rot = None


def BASE_POS_LLA_RAD():
    global _base_pos_lla_rad
    if _base_pos_lla_rad is None:
        _base_pos_lla_rad = pm.ecef2geodetic(
            BASE_POS_ECEF[0], BASE_POS_ECEF[1], BASE_POS_ECEF[2], deg=False
        )  # Base station position in LLA (lat, lon, alt)
    return _base_pos_lla_rad


def BASE_ECEF_TO_ENU_ROT_MAT():
    """
    Rotation matrix from ECEF to ENU at base position.
    """
    global _ecef_to_enu_rot
    if _ecef_to_enu_rot is None:
        lla = BASE_POS_LLA_RAD()
        _ecef_to_enu_rot = comm_utils.compute_ecef_enu_rot_mat(lla[0], lla[1])
    return _ecef_to_enu_rot


INIT_VEL_ENU = np.array([-0.527, -0.246, 0.0204])
# The y-axis of IMU point to the heading of the vehicle.
# Rotates from ENU (nav to body) = -90 + math.atan2(INIT_VEL_ENU[1], INIT_VEL_ENU[0]) ~= -90-155 degrees
# Nav to body = 360 + ( -90 - 155) = 115 degrees
# Body to Nav = - 115 degrees
INIT_YAW_RAD = np.deg2rad(-115.0)
INIT_ATTITUDE_STD_RAD = np.deg2rad(np.array([10.0, 10.0, 20.0]))
INIT_ACC_BIAS_STD = np.full(3, 2.5e-3)
INIT_GYRO_BIAS_STD = np.full(3, 4.0e-6)
INIT_POS_HOR_STD_M = 40.0  # Initial horizontal position standard deviation in meters
INIT_POS_VER_STD_M = 10.0  # Initial vertical position standard deviation in meters
INIT_VEL_STD_MPS = 5.0  # Initial velocity standard deviation in m/s


INCH_TO_METER = 0.0254  # 1 inch = 0.0254 meters


@dataclass(frozen=True)
class TexCupBoschImuParams:
    # IMU to GNSS antenna translation (meters) in IMU body frame.
    t_imu_to_ant_in_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [16.1811 * INCH_TO_METER, -6.2992 * INCH_TO_METER, 4.8425 * INCH_TO_METER]
        )
    )

    gravity: float = comm_utils.computeGravityConst(BASE_POS_LLA_RAD()[0])
    z_up: bool = True  # IMU z-axis points upward (GTSAM assumes Z axis pointing up)

    acc_noise_std = np.full(
        3, np.sqrt(0.1) * gravity / 1e3
    )  # 0.1 mg^2/Hz > m/s^2/sqrt(Hz)
    acc_bias_std = np.full(3, 10 * gravity / 1e3)  # 10 milli-g > m/s^2
    gyro_noise_std = np.full(
        3,
        np.sqrt(0.02) * np.pi / 180,
    )  # 0.02 (deg/s)^2/Hz -> rad/s/sqrt(Hz)
    gyro_bias_std = np.full(
        3,
        200 * np.pi / 180 / 3600,
    )  # 200 deg/h -> rad/s


@dataclass(frozen=True)
class TexCupLordImuParams:
    # IMU to GNSS antenna translation (meters) in IMU body frame.
    t_imu_to_ant_in_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [-0.461 * INCH_TO_METER, -0.125 * INCH_TO_METER, -0.119 * INCH_TO_METER]
        )
    )

    gravity: float = comm_utils.computeGravityConst(BASE_POS_LLA_RAD()[0])
    z_up: bool = False

    acc_noise_std = np.full(3, 0.1 * gravity / 1e3)  # 0.1 mg/sqrt(Hz) > m/s^2/sqrt(Hz)
    acc_bias_std = np.full(3, 0.5 * gravity / 1e3)  # 0.5 milli-g > m/s^2
    gyro_noise_std = np.full(
        3,
        0.01 * np.pi / 180,
    )  # 0.01 deg/s/sqrt(Hz) -> rad/s/sqrt(Hz)
    gyro_bias_std = np.full(
        3,
        8 * np.pi / 180 / 3600,
    )  # 8 deg/h -> rad/s
