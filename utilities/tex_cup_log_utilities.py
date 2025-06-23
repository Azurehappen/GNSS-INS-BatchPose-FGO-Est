from dataclasses import dataclass


@dataclass
class GroundTruthSignleEpoch:
    """
    Represents the ground truth for a single epoch.
    """

    pos_ecef_m: list[float]  # Position in ECEF coordinates (meters)
    lat_deg: float  # Latitude in degrees
    lon_deg: float  # Longitude in degrees
    ellipsoid_h_m: float  # Ellipsoid height, meter positive upward
    orthometric_h_m: (
        float  # Orthometric height (above sea level), meter positive upward
    )
    vel_ecef_mps: list[float]  # Velocity in ECEF coordinates (meters per second)
    vel_ned_mps: list[float]  # Velocity in NED coordinates (meters per second)
