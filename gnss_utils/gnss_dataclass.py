from constants.gnss_constants import Constellation
from dataclasses import dataclass


@dataclass(frozen=True)
class SignalType:
    """Represents a GNSS signal type."""

    constellation: Constellation
    obs_code: int
    channel_id: str = ""

    def __repr__(self) -> str:
        return f"{self.constellation.name} Signal Code {self.obs_code}{self.channel_id}"


@dataclass(frozen=True)
class SatelliteId:
    """Identifies a satellite by constellation and PRN."""

    constellation: Constellation
    prn: int

    def __repr__(self) -> str:
        return f"{self.constellation.name} PRN {self.prn}"


@dataclass(frozen=True)
class SignalChannelId:
    """Identifies a measurement channel by satellite and signal type."""

    satellite_id: SatelliteId
    signal_type: SignalType

    def __repr__(self) -> str:
        return f"{self.signal_type} {self.satellite_id}"
