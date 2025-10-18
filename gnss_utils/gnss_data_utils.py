from __future__ import annotations

from enum import Enum
from typing import Dict, Any, Optional, TYPE_CHECKING

from numpy import nan
import constants.gnss_constants as gnssConst
import numpy as np
from constants.gnss_constants import Constellation
from constants.parameters import BASE_POS_ECEF

if TYPE_CHECKING:
    from gnss_utils.time_utils import GpsTime

from dataclasses import dataclass
from gnss_utils.gnss_dataclass import *


def apply_ephemerides_to_obs(
    obs: Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]],
    eph_data: EphemerisData,
) -> None:
    """Populate satellite information for all observation channels.

    Channels without valid ephemeris are removed from ``obs``.
    """

    for epoch in list(obs.keys()):
        channels = obs[epoch]
        to_delete = []
        for ch_id, ch in channels.items():
            eph = eph_data.getCurrentEphemeris(
                ch_id.signal_type.constellation, ch_id.satellite_id.prn, epoch
            )
            if eph is None:
                to_delete.append(ch_id)
                continue
            ch.computeSatelliteInformation(eph)
            if ch.sat_pos_ecef_m is None:
                to_delete.append(ch_id)
        for d in to_delete:
            channels.pop(d, None)
        if not channels:
            obs.pop(epoch)


def apply_base_corrections(
    rover_obs: Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]],
    base_obs: Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]],
) -> None:
    """Apply differential corrections from base station observations."""

    from gnss_utils import satellite_utils

    base_epochs = sorted(base_obs.keys())
    if not base_epochs:
        return

    base_idx = 0
    base_pos = np.asarray(BASE_POS_ECEF)

    for epoch in sorted(list(rover_obs.keys())):
        while base_idx < len(base_epochs) - 1 and base_epochs[base_idx + 1] <= epoch:
            base_idx += 1

        if base_epochs[base_idx] > epoch:
            rover_obs.pop(epoch, None)
            continue

        base_channels = base_obs[base_epochs[base_idx]]
        rover_channels = rover_obs[epoch]
        to_delete = []
        for channel_id, channel in rover_channels.items():
            base_ch = base_channels.get(channel_id)
            if base_ch is None or base_ch.sat_pos_ecef_m is None:
                to_delete.append(channel_id)
                continue
            geometric_range = np.linalg.norm(
                base_pos - base_ch.sat_pos_ecef_m
            ) + satellite_utils.sagnac_correction(base_pos, base_ch.sat_pos_ecef_m)
            channel.correction_code_m = base_ch.code_m - geometric_range
            channel.correction_phase_m = base_ch.phase_m - geometric_range
            channel.correction_doppler_mps = base_ch.doppler_mps
            channel.applyRangeCorrections()
        for d in to_delete:
            rover_channels.pop(d, None)
        if not rover_channels:
            rover_obs.pop(epoch, None)


class GnssSignalChannel:
    """
    Class to hold observation data
    """

    def __init__(self):
        self.time = None  # GpsTime object
        self.signal_id: SignalChannelId = None  # SignalChannelId object
        self.svid: str = None  # Satellite SVID (Constellation+PRN)

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
        self.cycle_slip_status = gnssConst.CycleSlipType.NOT_AVAILABLE

    def addMeasurementFromObs(
        self,
        time: "GpsTime",
        signal_id: SignalChannelId,
        svid: str,
        code_m: float,
        phase_m: float,
        doppler_mps: float,
        cn0_dbhz: float,
    ) -> None:
        """Add a measurement from an observation."""
        self.time = time
        self.signal_id = signal_id
        self.svid = svid
        self.code_m = code_m
        self.phase_m = phase_m
        self.doppler_mps = doppler_mps
        self.cn0_dbhz = cn0_dbhz

    def __eq__(self, other):
        return (
            isinstance(other, GnssSignalChannel)
            and self.time == other.time
            and self.signal_id == other.signal_id
        )

    def __hash__(self):
        return hash((self.time, self.signal_id))

    def computeSatelliteInformation(
        self,
        ephemeris: Any,
    ) -> None:
        """Compute and store satellite information."""
        from gnss_utils import satellite_utils

        const = self.signal_id.signal_type.constellation

        t_sv = self.time.minus_float_seconds(self.code_m / gnssConst.SPEED_OF_LIGHT_MS)

        if const == Constellation.GLO:
            try:
                (
                    self.sat_pos_ecef_m,
                    self.sat_vel_ecef_m,
                    self.sat_clock_bias_m,
                    self.sat_clock_drift_mps,
                ) = satellite_utils.compute_glo_sat_info(t_sv, ephemeris)
            except ValueError as e:
                raise ValueError(
                    f"Failed to compute GLONASS satellite information: {e}"
                )
            # Clock bias has been adjusted by group delay
            self.code_m += self.sat_clock_bias_m
            self.phase_m += self.sat_clock_bias_m
            self.doppler_mps += self.sat_clock_drift_mps
            return

        if const == Constellation.GPS:
            mu = gnssConst.GpsConstants.MU
            omega_e = gnssConst.GpsConstants.OMEGA_DOT_E
            F_relativistic = gnssConst.GpsConstants.F
            tgd = ephemeris.tgd
            if self.signal_id.signal_type.obs_code == 1:
                group_delay_s = tgd
            elif self.signal_id.signal_type.obs_code == 2:
                group_delay_s = (
                    tgd * gnssConst.GpsConstants.L1_FREQ_SQUARE_D_L2_FREQ_SQUARE
                )
            elif self.signal_id.signal_type.obs_code == 5:
                raise ValueError(
                    "Group delay for L5 signal is not supported in this routine"
                )
            else:
                raise ValueError(
                    f"Unsupported observation code {self.signal_id.signal_type.obs_code} for GPS"
                )
        elif const == Constellation.GAL:
            mu = gnssConst.GalConstants.MU
            omega_e = gnssConst.GalConstants.OMEGA_DOT_E
            F_relativistic = gnssConst.GalConstants.F
            if self.signal_id.signal_type.obs_code == 1:
                group_delay_s = ephemeris.bgd_e1e5b
            elif self.signal_id.signal_type.obs_code == 5:
                group_delay_s = ephemeris.bgd_e1e5a
            elif self.signal_id.signal_type.obs_code == 7:
                group_delay_s = ephemeris.bgd_e1e5b
            else:
                raise ValueError(
                    f"Unsupported observation code {self.signal_id.signal_type.obs_code} for Galileo"
                )
        elif const == Constellation.BDS:
            mu = gnssConst.BdsConstants.MU
            omega_e = gnssConst.BdsConstants.OMEGA_DOT_E
            F_relativistic = gnssConst.BdsConstants.F
            if self.signal_id.signal_type.obs_code == 2:
                group_delay_s = ephemeris.tgd1
            elif self.signal_id.signal_type.obs_code == 7:
                group_delay_s = ephemeris.tgd2
            else:
                raise ValueError(
                    f"Unsupported observation code {self.signal_id.signal_type.obs_code} for BDS"
                )
        else:
            raise ValueError("Invalid constellation for this routine")
        try:
            (
                self.sat_pos_ecef_m,
                self.sat_vel_ecef_m,
                self.sat_clock_bias_m,
                self.sat_clock_drift_mps,
            ) = satellite_utils.compute_satellite_info(
                t_sv, const, ephemeris, mu, omega_e, F_relativistic, group_delay_s
            )
        except:
            raise ValueError("Failed to compute satellite information")

        self.sat_group_delay_m = group_delay_s * gnssConst.SPEED_OF_LIGHT_MS
        # Clock bias has been adjusted by group delay
        self.code_m += self.sat_clock_bias_m
        self.phase_m += self.sat_clock_bias_m
        self.doppler_mps += self.sat_clock_drift_mps


class GnssMeasurementChannel(GnssSignalChannel):
    """
    Class to hold observation data and satellite information for a single epoch.

    """

    def __init__(self):
        super().__init__()

        self.wavelength_m = None  # Wavelength in meters
        self.elevation_deg = None  # Satellite elevation angle in degrees
        self.azimuth_deg = None  # Satellite azimuth angle in degrees

        self.correction_code_m = None  # Correction for code measurement in meters
        self.correction_phase_m = (
            None  # Correction phase in meters (For DGNSS/RTK, including ambiguity)
        )
        self.correction_doppler_mps = None  # Correction for Doppler in m/s

        self.sigma_code_m = None  # Standard deviation of code measurement in meters
        self.sigma_phase_m = None  # Standard deviation of phase measurement in meters
        self.sigma_doppler_mps = (
            None  # Standard deviation of Doppler measurement in m/s
        )

    def __hash__(self):
        return hash((self.time, self.signal_id))

    def applyRangeCorrections(self) -> None:
        """Apply corrections to measurements."""
        if self.code_m is None or self.phase_m is None:
            raise ValueError("Measurements must be set before applying corrections")

        if self.correction_code_m is None or self.correction_phase_m is None:
            raise ValueError("Corrections must be set before applying them")

        self.code_m -= self.correction_code_m
        self.phase_m -= self.correction_phase_m


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

    def resetIndexLookup(self) -> None:
        """Reset the ephemeris index lookup tables."""
        for key in self.gps_eph_index_lookup:
            self.gps_eph_index_lookup[key] = 0
        for key in self.glo_eph_index_lookup:
            self.glo_eph_index_lookup[key] = 0
        for key in self.gal_eph_index_lookup:
            self.gal_eph_index_lookup[key] = 0
        for key in self.bds_eph_index_lookup:
            self.bds_eph_index_lookup[key] = 0

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

        if constellation == Constellation.GLO:
            # For GLONASS, | gps_time - eph_epoch_time | < 15 minutes
            while idx < len(eph_list):
                if abs(gps_time - eph_list[idx][0]) <= 15 * 60:
                    # Find the target ephemeris index
                    break
                if eph_list[idx][0] - gps_time > 15 * 60:
                    # If the next ephemeris time is more than 15 minutes ahead, stop searching
                    # This also means the previous ephemeris does not cover gps_time
                    return None
                idx += 1
        else:
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


class CdmaEphemeris:
    """Class to hold CDMA ephemeris common parameters."""

    def __init__(self):
        self.prn = nan
        self.toc: GpsTime = None  # Time of Clock (represented in GpsTime)
        self.toc_str: str = ""
        self.sv_clock_bias: float = nan
        self.sv_clock_drift: float = nan
        self.sv_clock_drift_rate: float = nan
        self.crs: float = nan
        self.delta_n: float = nan
        self.m0: float = nan
        self.cuc: float = nan
        self.ecc: float = nan
        self.cus: float = nan
        self.sqrtA: float = nan
        self.toe_sec: float = nan  # Time of Ephemeris in seconds of system week
        self.toe: GpsTime = None  # Time of Ephemeris (represented in GpsTime)
        self.cic: float = nan
        self.omega0: float = nan
        self.cis: float = nan
        self.i0: float = nan
        self.crc: float = nan
        self.omega: float = nan
        self.omega_dot: float = nan
        self.idot: float = nan


class GpsEphemeris(CdmaEphemeris):
    """Class to hold GPS ephemeris parameters. See RINEX 3.03 documentation for details."""

    def __init__(self):
        super().__init__()
        self.iode: float = nan
        self.code_on_L2: float = nan
        self.gps_week: int = nan
        self.l2p_data_flag: int = nan
        self.sv_accuracy: float = nan
        self.sv_health: int = nan  # 0: healthy
        self.tgd: float = nan
        self.iodc: float = nan
        self.transmission_time: GpsTime = (
            None  # Time of Transmission (represented in GpsTime)
        )
        self.fit_interval: float = nan


class GloEphemeris:
    """Class to hold GLONASS ephemeris parameters. See RINEX 3.03 documentation for details."""

    def __init__(self):
        self.prn = nan
        self.toc: GpsTime = None  # Time of Clock (represented in GpsTime)
        self.toc_str: str = ""
        self.sv_clock_bias: float = nan
        self.sv_relative_freq_bias: float = nan
        self.message_frame_time: float = nan
        self.x_pos: float = nan
        self.x_vel: float = nan
        self.x_acc: float = nan
        self.health: int = nan
        self.y_pos: float = nan
        self.y_vel: float = nan
        self.y_acc: float = nan
        self.freq_number: int = (
            -100
        )  # Frequency number (-7<->+13, -100 if not applicable)
        self.z_pos: float = nan
        self.z_vel: float = nan
        self.z_acc: float = nan
        self.age_of_oper_info: float = nan


class GalEphemeris:
    """Class to hold Galileo ephemeris parameters. See RINEX 3.03 documentation for details."""

    def __init__(self):
        super().__init__()
        self.iod_nav: float = nan
        self.data_source: int = -1
        self.gal_week: int = nan
        self.sisa: float = nan
        self.sv_health: int = nan  # 0: healthy
        self.bgd_e1e5a: float = nan
        self.bgd_e1e5b: float = nan
        self.iodnav: float = nan
        self.transmission_time: GpsTime = (
            None  # Time of Transmission (represented in GpsTime)
        )

        # Health status flags based on sv_health value.
        self.e1b_is_valid: bool = False
        self.e1b_is_health: bool = False
        self.e5a_is_valid: bool = False
        self.e5a_is_health: bool = False
        self.e5b_is_valid: bool = False
        self.e5b_is_health: bool = False


class BdsEphemeris:
    """Class to hold BeiDou ephemeris parameters. See RINEX 3.03 documentation for details."""

    def __init__(self):
        super().__init__()
        self.aode: float = None
        self.bds_week: int = None
        self.sv_accuracy: float = None
        self.sv_health: int = None  # 0: healthy
        self.tgd1: float = None  # B1-B3
        self.tgd2: float = None  # B1-B2
        self.transmission_time: GpsTime = (
            None  # Time of Transmission (represented in GpsTime)
        )
        self.aodc: float = None
