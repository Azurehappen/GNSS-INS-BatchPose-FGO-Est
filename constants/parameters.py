from dataclasses import dataclass, field
import numpy as np
import pymap3d as pm


def computeGravityConst(lat: float) -> float:
    g_h = (
        9.7803267715
        * (1 + 0.001931851353 * (np.sin(lat)) ** 2)
        / np.sqrt(1 - 0.0066943800229 * (np.sin(lat)) ** 2)
    )

    return g_h


@dataclass
class GnssParameters:
    ELEVATION_MASK: float = 15.0  # Minimum elevation angle in degrees
    CNO_THRESHOLD: float = (
        20.0  # Minimum C/N0 in dB-Hz for a satellite to be considered valid
    )

    enable_gps: bool = True
    enable_galileo: bool = True
    enable_glonass: bool = True
    enable_beidou: bool = True


RINEX_OBS_CHANNEL_TO_USE: dict[str, set[str]] = {
    "G": {"1C", "2L"},
    "R": {"1C", "2C"},
    "E": {"1C", "7Q"},
    "C": {"2I"},
}

BASE_POS_ECEF = [
    -742080.4125,
    -5462031.7412,
    3198339.6909,
]  # Base station position in ECEF coordinates (meters)

BASE_POS_LLA = pm.ecef2geodetic(
    BASE_POS_ECEF[0], BASE_POS_ECEF[1], BASE_POS_ECEF[2]
)  # Base station position in LLA (lat, lon, alt)


INCH_TO_METER = 0.0254  # 1 inch = 0.0254 meters


@dataclass
class TexCupBoschImuParams:
    # GNSS antenna to IMU translation (meters) in IMU body frame.
    t_ant_to_imu_in_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [16.1811 * INCH_TO_METER, -6.2992 * INCH_TO_METER, 4.8425 * INCH_TO_METER]
        )
    )

    gravity: float = computeGravityConst(np.deg2rad(BASE_POS_LLA[0]))
    z_up: bool = True  # IMU z-axis points upward (GTSAM assumes Z axis pointing up)


@dataclass
class TexCupLordImuParams:
    # GNSS antenna to IMU translation (meters) in IMU body frame.
    t_ant_to_imu_in_b: np.ndarray = field(
        default_factory=lambda: np.array(
            [-0.461 * INCH_TO_METER, -0.125 * INCH_TO_METER, -0.119 * INCH_TO_METER]
        )
    )

    gravity: float = computeGravityConst(np.deg2rad(BASE_POS_LLA[0]))
    z_up: bool = False
