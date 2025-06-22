from dataclasses import dataclass


@dataclass
class GnssParameters:
    ELEVATION_MASK: float = 15.0  # Minimum elevation angle in degrees
    CNO_THRESHOLD: float = (
        20.0  # Minimum C/N0 in dB-Hz for a satellite to be considered valid
    )


RINEX_OBS_CHANNEL_TO_USE: dict[str, set[str]] = {
    "G": {"1C", "2L"},
    "R": {"1C", "2C"},
    "E": {"1C", "5Q"},
    "C": {"2I"},
}
