from dataclasses import dataclass, field
import numpy as np
import pymap3d as pm
from gnss_utils.gnss_dataclass import SignalType
from constants.gnss_constants import Constellation


def compute_ecef_ned_rot_mat(lat_rad: float, lon_rad: float) -> np.ndarray:
    """
    Compute the rotation matrix from ECEF to NED frame.

    Args:
        lat_rad: Latitude in radians
        lon_rad: Longitude in radians

    Returns:
        3x3 rotation matrix from ECEF to NED
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    return np.array(
        [
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [-sin_lon, cos_lon, 0],
            [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
        ]
    )


def computeGravityConst(lat_rad: float) -> float:
    g_h = (
        9.7803267715
        * (1 + 0.001931851353 * (np.sin(lat_rad)) ** 2)
        / np.sqrt(1 - 0.0066943800229 * (np.sin(lat_rad)) ** 2)
    )

    return g_h


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

    enable_gps: bool = True
    enable_galileo: bool = True
    enable_glonass: bool = True
    enable_beidou: bool = True


phase_sigma_a = 0.003  # meters
phase_sigma_b = 0.003  # meters

# SignalType -> (fact_a, fact_b)
GNSS_ELEV_MODEL_PARAMS: dict[SignalType, tuple[float, float]] = {
    SignalType(Constellation.GPS, 1, "C"): (300 * phase_sigma_a, 450 * phase_sigma_b),
    SignalType(Constellation.GPS, 2, "W"): (300 * phase_sigma_a, 450 * phase_sigma_b),
    SignalType(Constellation.GLO, 1, "C"): (700 * phase_sigma_a, 800 * phase_sigma_b),
    SignalType(Constellation.GLO, 2, "C"): (700 * phase_sigma_a, 800 * phase_sigma_b),
    SignalType(Constellation.GAL, 1, "C"): (350 * phase_sigma_a, 500 * phase_sigma_b),
    SignalType(Constellation.GAL, 7, "Q"): (350 * phase_sigma_a, 500 * phase_sigma_b),
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
_ecef_to_ned_rot = None


def BASE_POS_LLA_RAD():
    global _base_pos_lla_rad
    if _base_pos_lla_rad is None:
        _base_pos_lla_rad = pm.ecef2geodetic(
            BASE_POS_ECEF[0], BASE_POS_ECEF[1], BASE_POS_ECEF[2], deg=False
        )  # Base station position in LLA (lat, lon, alt)
    return _base_pos_lla_rad


def BASE_ECEF_TO_NED_ROT_MAT():
    """
    Rotation matrix from ECEF to NED at base position.
    C_e_n = [   -sin_lat*cos_lng -sin_lat*sin_lng   cos_lat; % Farrell eqn. 2.32
                -sin_lng          cos_lng           0;
                -cos_lat*cos_lng -cos_lat*sin_lng  -sin_lat];
    """
    global _ecef_to_ned_rot
    if _ecef_to_ned_rot is None:
        lla = BASE_POS_LLA_RAD()
        _ecef_to_ned_rot = compute_ecef_ned_rot_mat(lla[0], lla[1])
    return _ecef_to_ned_rot


INCH_TO_METER = 0.0254  # 1 inch = 0.0254 meters


@dataclass(frozen=True)
class TexCupBoschImuParams:
    # IMU to GNSS antenna translation (meters) in IMU body frame.
    t_imu_to_ant_in_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [16.1811 * INCH_TO_METER, -6.2992 * INCH_TO_METER, 4.8425 * INCH_TO_METER]
        )
    )

    gravity: float = computeGravityConst(BASE_POS_LLA_RAD()[0])
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

    gravity: float = computeGravityConst(BASE_POS_LLA_RAD()[0])
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
