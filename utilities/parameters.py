from dataclasses import dataclass


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
