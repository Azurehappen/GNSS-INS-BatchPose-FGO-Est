"""RINEX Observation file parser.

The parser only extracts basic observation measurements (code, carrier phase,
Doppler and C/N0).  It supports a small subset of the RINEX 3 observation
format which is sufficient for the example data bundled with this repository.

Two helper functions are exposed:

``parse_rinex_obs``
    Generic parser that returns observation channels either as
    :class:`~utilities.gnss_data_structures.GnssSignalChannel` or
    :class:`~utilities.gnss_data_structures.GnssMeasurementChannel` instances.

``parse_rinex_obs`` takes an ``interval`` argument which can be used to down
sample the file.  For example the base station RINEX file is provided at 1 Hz
but only every 30 seconds are used in the example workflow.

The implementation here is intentionally lightweight but aims to follow the
RINEX 3.04 specification closely enough for our test data.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Type

import pandas as pd

import constants.gnss_constants as gnssConst
from utilities.gnss_data_structures import (
    Constellation,
    GnssMeasurementChannel,
    GnssSignalChannel,
    SignalType,
)
from utilities.parameters import GnssParameters
from utilities.time_utils import GpsTime


_CONST_MAP = {
    "G": Constellation.GPS,
    "R": Constellation.GLO,
    "E": Constellation.GAL,
    "C": Constellation.BDS,
}


def _parse_header(f) -> Dict[Constellation, List[str]]:
    """Parse the RINEX observation header.

    Parameters
    ----------
    f : Iterable[str]
        Open file object positioned at the start of the file.

    Returns
    -------
    Dict[Constellation, List[str]]
        Mapping from constellation to the ordered list of observation type
        strings (e.g. ``"C1C"``, ``"L1C"``).
    """

    obs_types: Dict[Constellation, List[str]] = {}

    for line in f:
        if "SYS / # / OBS TYPES" in line:
            sys_char = line[0]
            num_types = int(line[3:6])
            types = line[7:60].split()
            while len(types) < num_types:
                nxt = next(f)
                types.extend(nxt[7:60].split())
            const = _CONST_MAP.get(sys_char)
            if const:
                obs_types[const] = types[:num_types]
        elif "END OF HEADER" in line:
            break

    return obs_types


def _get_wavelength(constellation: Constellation, prn: int, obs_code: int) -> float:
    """Return wavelength in metres for the given observation code."""

    if constellation == Constellation.GPS:
        return gnssConst.GpsConstants.ObsCodeToWavelength[obs_code]
    if constellation == Constellation.GAL:
        return gnssConst.GalConstants.ObsCodeToWavelength[obs_code]
    if constellation == Constellation.BDS:
        return gnssConst.BdsConstants.ObsCodeToWavelength[obs_code]
    if constellation == Constellation.GLO:
        ch = gnssConst.GloConstants.PrnToChannelNum.get(prn)
        if ch is None:
            raise ValueError(f"Unknown GLONASS channel number for PRN {prn}")
        return gnssConst.GloConstants().getObsCodeToWavelength(obs_code, ch)
    raise ValueError(f"Unsupported constellation {constellation}")


def _parse_epoch_header(line: str) -> Tuple[GpsTime, int]:
    """Parse epoch header line beginning with ``>``."""

    parts = line[1:].split()
    year, month, day, hour, minute = map(int, parts[:5])
    second = float(parts[5])
    num_sats = int(parts[7])
    ts = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
    ts += pd.Timedelta(seconds=second)
    return GpsTime.fromDatetime(ts, Constellation.GPS), num_sats


def parse_rinex_obs(
    file_path: str,
    *,
    interval: int = 1,
    measurement_channel: bool = False,
    parameters: GnssParameters | None = None,
) -> Dict[GpsTime, set[GnssSignalChannel]]:
    """Parse a RINEX observation file.

    Parameters
    ----------
    file_path : str
        Path to the RINEX observation file.
    interval : int, optional
        Only epochs with timestamps that are multiples of ``interval`` seconds
        (relative to the first epoch) are returned.  Default is ``1`` which
        keeps all epochs.
    measurement_channel : bool, optional
        When ``True`` the returned channels are instances of
        :class:`GnssMeasurementChannel` instead of :class:`GnssSignalChannel`.
    parameters : GnssParameters | None
        Parameters controlling the CN0 threshold.  If ``None`` a default
        ``GnssParameters`` instance is used.

    Returns
    -------
    Dict[GpsTime, set[GnssSignalChannel]]
        Mapping from :class:`GpsTime` to a set of observation channels.
    """

    params = parameters or GnssParameters()
    channel_cls: Type[GnssSignalChannel]
    channel_cls = GnssMeasurementChannel if measurement_channel else GnssSignalChannel

    result: Dict[GpsTime, set[GnssSignalChannel]] = defaultdict(set)

    with open(file_path, "r") as f:
        obs_type_map = _parse_header(f)

        first_epoch: GpsTime | None = None

        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(">"):
                continue

            epoch_time, num_sats = _parse_epoch_header(line)

            if first_epoch is None:
                first_epoch = epoch_time

            if int(epoch_time.gps_timestamp - first_epoch.gps_timestamp) % interval != 0:
                # Skip this epoch to achieve the desired interval
                # Still need to consume the following observation lines
                for _ in range(num_sats):
                    f.readline()
                continue

            for _ in range(num_sats):
                sat_line = f.readline()
                if not sat_line:
                    break
                prn_id = sat_line[:3]
                const = _CONST_MAP.get(prn_id[0])
                if const is None or const not in obs_type_map:
                    continue

                obs_list = obs_type_map[const]
                needed_len = len(obs_list) * 16
                data_str = sat_line[3:].rstrip("\n")
                while len(data_str) < needed_len:
                    cont = f.readline()
                    data_str += cont.rstrip("\n")

                values = [data_str[i : i + 16] for i in range(0, needed_len, 16)]
                parsed_vals = [v[:14].strip() for v in values]

                meas_map: Dict[Tuple[int, str], Dict[str, float]] = {}
                for t, val_str in zip(obs_list, parsed_vals):
                    if not val_str:
                        val = None
                    else:
                        val = float(val_str)
                    key = (int(t[1:-1]), t[-1])
                    meas_map.setdefault(key, {})[t[0]] = val

                prn = int(prn_id[1:])
                for (obs_code, chan_id), meas in meas_map.items():
                    if not {"C", "L", "D", "S"}.issubset(meas):
                        continue

                    cn0 = meas["S"]
                    if cn0 is None or cn0 < params.cn0_threshold:
                        continue

                    code = meas["C"]
                    phase_cycles = meas["L"]
                    doppler_hz = meas["D"]
                    if code is None or phase_cycles is None or doppler_hz is None:
                        continue

                    wavelength = _get_wavelength(const, prn, obs_code)
                    phase_m = phase_cycles * wavelength
                    doppler_mps = -doppler_hz * wavelength

                    signal = SignalType(const, obs_code)
                    signal.channel_id = chan_id

                    channel = channel_cls()
                    channel.addMeasurementFromObs(
                        epoch_time, signal, prn, code, phase_m, doppler_mps, cn0
                    )

                    result[epoch_time].add(channel)

    return result

