from __future__ import annotations

from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from numpy import nan
import constants.gnss_constants as gnssConst

if TYPE_CHECKING:
    from utilities.time_utils import GpsTime


class Constellation(Enum):
    GPS = 1
    GLO = 2
    GAL = 3
    BDS = 4


class SignalType:
    # Base class for signal types
    def __init__(self, constellation: Constellation, obs_code: int):
        self.constellation = constellation
        # RINEX observation code
        self.obs_code = obs_code
        # Unique identifier for the signal channel, See RINEX 3.03 Table 5 Channel.
        self.channel_id = ""

    def __repr__(self):
        return f"{self.constellation.name} Signal Code {self.obs_code}{self.channel_id}"

    def __eq__(self, other):
        return (
            isinstance(other, SignalType)
            and self.constellation == other.constellation
            and self.obs_code == other.obs_code
            and self.channel_id == other.channel_id
        )

    def __hash__(self):
        return hash((self.constellation, self.obs_code, self.channel_id))


class GnssSignalChannel:
    """
    Class to hold observation data
    """

    def __init__(self):
        self.time = None  # GpsTime object
        self.signal_type = None  # SignalType object
        self.prn = None  # Satellite PRN number

        self.code_m = None  # Code measurement value
        self.phase_m = None  # Phase measurement value
        self.doppler_mps = None  # Doppler frequency shift in meters per second
        self.cn0_dbhz = None  # Signal Strength (C/N0) in dB-Hz

        self.sat_pos_ecef_m = None  # Satellite position in ECEF coordinates (x, y, z)
        self.sat_vel_ecef_m = (
            None  # Satellite velocity in ECEF coordinates (vx, vy, vz)
        )
        self.sat_clock_bias_m = None  # Satellite clock bias in meters
        self.sat_group_delay_m = None  # Satellite group delay in meters
        self.sat_clock_drift_mps = None  # Satellite clock drift in meters per second

    def addMeasurementFromObs(
        self,
        time: "GpsTime",
        signal_type: SignalType,
        prn: int,
        code_m: float,
        phase_m: float,
        doppler_mps: float,
        cn0_dbhz: float,
    ) -> None:
        """Add a measurement from an observation."""
        self.time = time
        self.signal_type = signal_type
        self.prn = prn
        self.code_m = code_m
        self.phase_m = phase_m
        self.doppler_mps = doppler_mps
        self.cn0_dbhz = cn0_dbhz

    def computeSatelliteInformation(
        self,
        ephemeris: Any,
    ) -> None:
        """Compute and store satellite information."""

        # TODO: Implement satellite position and velocity computation


class GnssMeasurementChannel(GnssSignalChannel):
    """
    Class to hold observation data and satellite information for a single epoch.

    """

    def __init__(self):
        super().__init__()

        self.elevation_deg = None  # Satellite elevation angle in degrees
        self.azimuth_deg = None  # Satellite azimuth angle in degrees

        self.correction_code_m = None  # Correction for code measurement in meters
        self.correction_phase_m = (
            None  # Correction phase in meters (For DGNSS/RTK, including ambiguity)
        )

        self.sigma_code_m = None  # Standard deviation of code measurement in meters
        self.sigma_phase_m = None  # Standard deviation of phase measurement in meters
        self.sigma_doppler_mps = (
            None  # Standard deviation of Doppler measurement in m/s
        )


class EphemerisData:
    """
    Class to hold ephemeris data for multiple constellations.

    After loading ephemeris data, the ephemerides are organized by
    constellation and by PRN.

    Each constellation has a dictionary mapping PRN to a sorted list of tuples containing
    GpsTime and the corresponding ephemeris object.

    The class also maintains lookup dictionaries to track the current index
    in the ephemeris time lists for each PRN.

    """

    def __init__(self):
        # key: PRN, value: a list of tuples of {GpsTime, GpsEphemeris}
        self.gps_ephemerides = {}
        # key: PRN, value: a list of tuples of {GpsTime, GloEphemeris}
        self.glo_ephemerides = {}
        # key: PRN, value: a list of tuples of {GpsTime, GalEphemeris}
        self.gal_ephemerides = {}
        # key: PRN, value: a list of tuples of {GpsTime, BdsEphemeris}
        self.bds_ephemerides = {}

        # key: PRN, value: the current index in the ephemeris time list
        self.gps_eph_index_lookup = {}
        self.glo_eph_index_lookup = {}
        self.gal_eph_index_lookup = {}
        self.bds_eph_index_lookup = {}

    def _get_dict_and_lookup(self, constellation: Constellation):
        if constellation == Constellation.GPS:
            return (
                self.gps_ephemerides,
                self.gps_eph_index_lookup,
                gnssConst.GpsConstants.MAX_DURATION_TO_EPH,
            )
        elif constellation == Constellation.GLO:
            return (
                self.glo_ephemerides,
                self.glo_eph_index_lookup,
                gnssConst.GloConstants.MAX_DURATION_TO_EPH,
            )
        elif constellation == Constellation.GAL:
            return (
                self.gal_ephemerides,
                self.gal_eph_index_lookup,
                gnssConst.GalConstants.MAX_DURATION_TO_EPH,
            )
        elif constellation == Constellation.BDS:
            return (
                self.bds_ephemerides,
                self.bds_eph_index_lookup,
                gnssConst.BdsConstants.MAX_DURATION_TO_EPH,
            )
        else:
            raise ValueError(f"Unsupported constellation {constellation}")

    def addEphemeris(
        self, constellation: Constellation, prn: int, time: "GpsTime", eph: Any
    ) -> None:
        """Add an ephemeris entry for the given PRN and constellation."""

        eph_dict, idx_lookup, _ = self._get_dict_and_lookup(constellation)
        eph_dict.setdefault(prn, []).append((time, eph))
        idx_lookup.setdefault(prn, 0)

    def getCurrentEphemeris(
        self, constellation: Constellation, prn: int, gps_time: "GpsTime"
    ) -> Optional[Any]:
        """Return the ephemeris covering ``gps_time`` if available."""

        eph_dict, index_lookup, max_dur = self._get_dict_and_lookup(constellation)
        eph_list = eph_dict.get(prn)
        if not eph_list:
            return None
        idx = index_lookup.get(prn, 0)

        # Move index forward to the first ephemeris time greater than gps_time
        while idx < len(eph_list) and gps_time > eph_list[idx][0]:
            idx += 1

        if idx >= len(eph_list):
            return None

        # eph_list[idx] is tuple of (GpsTime, Ephemeris)
        diff = eph_list[idx][0] - gps_time
        if diff < 0 or diff >= max_dur:
            return None

        index_lookup[prn] = idx
        return eph_list[idx][1]


class GpsEphemeris:
    """Class to hold GPS ephemeris parameters. See RINEX 3.03 documentation for details."""

    def __init__(self):
        self.prn = nan
        self.toc = nan  # Time of Clock (represented in GpsTime)
        self.sv_clock_bias = nan
        self.sv_clock_drift = nan
        self.sv_clock_drift_rate = nan
        self.iode = nan
        self.crs = nan
        self.delta_n = nan
        self.m0 = nan
        self.cuc = nan
        self.ecc = nan
        self.cus = nan
        self.sqrtA = nan
        self.toe = nan  # Time of Ephemeris (represented in GpsTime)
        self.cic = nan
        self.omega0 = nan
        self.cis = nan
        self.i0 = nan
        self.crc = nan
        self.omega = nan
        self.omega_dot = nan
        self.idot = nan
        self.code_on_L2 = nan
        self.gps_week = nan
        self.l2p_data_flag = nan
        self.sv_accuracy = nan
        self.sv_health = nan  # 0: healthy
        self.tgd = nan
        self.iodc = nan
        self.transmission_time = nan  # Time of Transmission (represented in GpsTime)
        self.fit_interval = nan


class GloEphemeris:
    """Class to hold GLONASS ephemeris parameters. See RINEX 3.03 documentation for details."""

    def __init__(self):
        self.prn = nan
        self.toc = nan  # Time of Clock (represented in GpsTime)
        self.sv_clock_bias = nan
        self.sv_relative_freq_bias = nan
        self.message_frame_time = nan
        self.x_pos = nan
        self.x_vel = nan
        self.x_acc = nan
        self.health = nan
        self.y_pos = nan
        self.y_vel = nan
        self.y_acc = nan
        self.freq_number = nan
        self.z_pos = nan
        self.z_vel = nan
        self.z_acc = nan
        self.age_of_oper_info = nan


class GalEphemeris:
    """Class to hold Galileo ephemeris parameters. See RINEX 3.03 documentation for details."""

    def __init__(self):
        self.prn = nan
        self.toc = nan  # Time of Clock (represented in GpsTime)
        self.sv_clock_bias = nan
        self.sv_clock_drift = nan
        self.sv_clock_drift_rate = nan
        self.iod_nav = nan
        self.crs = nan
        self.delta_n = nan
        self.m0 = nan
        self.cuc = nan
        self.ecc = nan
        self.cus = nan
        self.sqrtA = nan
        self.toe = nan  # Time of Ephemeris (represented in GpsTime)
        self.cic = nan
        self.omega0 = nan
        self.cis = nan
        self.i0 = nan
        self.crc = nan
        self.omega = nan
        self.omega_dot = nan
        self.idot = nan
        self.data_source = ""  # "F-NAV" or "I-NAV". Currently use "F-NAV" only
        self.gal_week = nan
        self.sisa = nan
        self.sv_health = nan  # 0: healthy
        self.bgd_e1e5a = nan
        self.bgd_e1e5b = nan
        self.iodnav = nan
        self.transmission_time = nan  # Time of Transmission (represented in GpsTime)

        # Health status flags based on sv_health value.
        self.e1b_is_valid = False
        self.e1b_is_health = False
        self.e5a_is_valid = False
        self.e5a_is_health = False
        self.e5b_is_valid = False
        self.e5b_is_health = False


class BdsEphemeris:
    """Class to hold BeiDou ephemeris parameters. See RINEX 3.03 documentation for details."""

    def __init__(self):
        self.prn = nan
        self.toc = nan  # Time of Clock (represented in GpsTime)
        self.sv_clock_bias = nan
        self.sv_clock_drift = nan
        self.sv_clock_drift_rate = nan
        self.aode = nan
        self.crs = nan
        self.delta_n = nan
        self.m0 = nan
        self.cuc = nan
        self.ecc = nan
        self.cus = nan
        self.sqrtA = nan
        self.toe = nan  # Time of Ephemeris (represented in GpsTime)
        self.cic = nan
        self.omega0 = nan
        self.cis = nan
        self.i0 = nan
        self.crc = nan
        self.omega = nan
        self.omega_dot = nan
        self.idot = nan
        self.bds_week = nan
        self.sv_accuracy = nan
        self.sv_health = nan  # 0: healthy
        self.tgd1 = nan  # B1-B3
        self.tgd2 = nan  # B1-B2
        self.transmission_time = nan  # Time of Transmission (represented in GpsTime)
        self.aodc = nan
