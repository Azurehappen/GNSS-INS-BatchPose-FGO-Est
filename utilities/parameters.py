from dataclasses import dataclass, field


@dataclass
class GnssParameters:
    elevation_mask: float = 15.0  # Minimum elevation angle in degrees
    cn0_threshold: float = (
        20.0  # Minimum C/N0 in dB-Hz for a satellite to be considered valid
    )


@dataclass
class RinexParameters:
    """Parameters controlling RINEX observation parsing."""

    obs_channel_to_use: dict[str, set[str]] = field(
        default_factory=lambda: {
            "G": {"1C", "2L"},
            "R": {"1C", "2C"},
            "E": {"1C", "5Q"},
            "C": {"2I"},
        }
    )
