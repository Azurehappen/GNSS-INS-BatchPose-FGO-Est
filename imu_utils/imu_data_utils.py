from dataclasses import dataclass, field
from gnss_utils.time_utils import GpsTime
from typing import Dict
import numpy as np
import pandas as pd
import pymap3d as pm


@dataclass
class GroundTruthSingleEpoch:
    """
    Represents the ground truth for a single epoch.
    """

    epoch: GpsTime
    lat_deg: float  # Latitude in degrees
    lon_deg: float  # Longitude in degrees
    ellipsoid_h_m: float  # Ellipsoid height, meter positive upward
    orthometric_h_m: (
        float  # Orthometric height (above sea level), meter positive upward
    )
    pos_ecef_m: np.ndarray = field(
        default_factory=lambda: np.zeros((3,))
    )  # Position (3 x 1) in ECEF coordinates (meters)
    vel_ned_mps: np.ndarray = field(
        default_factory=lambda: np.zeros((3,))
    )  # Velocity (3 x 1) in NED coordinates (meters per second)


@dataclass
class ImuSingleEpoch:
    """
    Represents the IMU data for a single epoch.
    """

    epoch: GpsTime
    acc_mps2: np.ndarray = field(default_factory=lambda: np.zeros((3,)))
    gyro_rps: np.ndarray = field(default_factory=lambda: np.zeros((3,)))


def parse_ground_truth_log(file_path: str) -> Dict[GpsTime, GroundTruthSingleEpoch]:
    """
    Parses the ground truth log file and returns a dictionary of GpsTime to GroundTruthSingleEpoch.
    """
    ground_truth_data = {}

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines

            parts = line.split()
            if len(parts) < 15:
                continue  # Skip malformed lines

            date = parts[0]
            time = parts[1]
            epoch = GpsTime.fromDatetime(timestamp=pd.Timestamp(f"{date} {time}"))

            lat_deg = float(parts[3])
            lon_deg = float(parts[4])
            ellipsoid_h_m = float(parts[5])
            orthometric_h_m = float(parts[6])
            x, y, z = pm.geodetic2ecef(lat_deg, lon_deg, ellipsoid_h_m)
            pos_ecef_m = np.asarray([x, y, z])
            vel_ned_mps = np.asarray(
                [float(parts[17]), float(parts[18]), float(parts[19])]
            )

            ground_truth_data[epoch] = GroundTruthSingleEpoch(
                epoch=epoch,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                ellipsoid_h_m=ellipsoid_h_m,
                orthometric_h_m=orthometric_h_m,
                pos_ecef_m=pos_ecef_m,
                vel_ned_mps=vel_ned_mps,
            )

    print(f"Loaded {len(ground_truth_data)} ground truth epochs")
    return ground_truth_data


def parse_imu_log(file_path: str, z_up: bool) -> list[ImuSingleEpoch]:
    """
    Parses the IMU log file and returns a list of ImuSingleEpoch.
    """
    imu_data = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue  # Skip comments and empty lines

            # Split by ", "
            parts = line.split(", ")
            if len(parts) < 9:
                continue  # Skip malformed lines

            week = parts[0]
            whole_sec = parts[1]
            frac_sec = parts[2]
            epoch = GpsTime.fromWeekAndTow(
                week=int(week), tow=float(whole_sec) + float(frac_sec)
            )

            if z_up:
                acc_mps2 = np.asarray(
                    [float(parts[3]), float(parts[4]), float(parts[5])]
                )
                gyro_rps = np.asarray(
                    [float(parts[6]), float(parts[7]), float(parts[8])]
                )
            else:
                acc_mps2 = np.asarray(
                    [float(parts[3]), -float(parts[4]), -float(parts[5])]
                )
                gyro_rps = np.asarray(
                    [float(parts[6]), -float(parts[7]), -float(parts[8])]
                )

            imu_data.append(
                ImuSingleEpoch(epoch=epoch, acc_mps2=acc_mps2, gyro_rps=gyro_rps)
            )

    print(f"Loaded {len(imu_data)} IMU epochs")
    return imu_data
