"""RINEX 3.04 Observation file parser.

The parser only basic observation measurements (code, carrier phase,
Doppler and C/N0).

Two helper functions are exposed:

``parse_rinex_obs`` takes an ``interval`` argument which can be used to down
sample the file.  For example the base station RINEX file is provided at 1 Hz
but only every 30 seconds are used in the example workflow.

The implementation here is intentionally lightweight but aims to follow the
RINEX 3.04 specifications.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Type

import pandas as pd

import constants.gnss_constants as gnssConst
from utilities.gnss_data_structures import (
    Constellation,
    GnssMeasurementChannel,
    SignalType,
)
from utilities.parameters import GnssParameters, RINEX_OBS_CHANNEL_TO_USE
from utilities.time_utils import GpsTime


_SYS_CHAR_TO_CONSTEL_MAP = {
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

    sys_char_to_obs_type: Dict[str, List[str]] = {}

    for line in f:
        if "SYS / # / OBS TYPES" in line:
            sys_char = line[0]
            num_types = int(line[3:6])
            types = line[7:60].split()
            while len(types) < num_types:
                nxt = next(f)
                types.extend(nxt[7:60].split())
            sys_char_to_obs_type[sys_char] = types[:num_types]
        elif "END OF HEADER" in line:
            break

    return sys_char_to_obs_type


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
) -> Dict[GpsTime, set[GnssMeasurementChannel]]:
    """Parse a RINEX observation file.

    Parameters
    ----------
    file_path : str
        Path to the RINEX observation file.
    interval : int, optional
        Only epochs with timestamps that are multiples of ``interval`` seconds
        (relative to the first epoch) are returned.  Default is ``1`` which
        keeps all epochs.

    Returns
    -------
    Dict[GpsTime, set[GnssSignalChannel]]
        Mapping from :class:`GpsTime` to a set of observation channels.
    """

    result: Dict[GpsTime, set[GnssMeasurementChannel]] = defaultdict(set)

    with open(file_path, "r") as f:
        sys_char_to_obs_type = _parse_header(f)

        first_epoch: GpsTime | None = None
        prev_epoch: GpsTime | None = None

        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith(">"):
                continue

            epoch_time, num_sats = _parse_epoch_header(line)

            if first_epoch is None:
                first_epoch = epoch_time
                prev_epoch = epoch_time
            else:
                if (
                    interval > 1
                    and prev_epoch is not None
                    and (epoch_time.gps_timestamp - prev_epoch.gps_timestamp)
                    < interval - 1e-6
                ):
                    for _ in range(num_sats):
                        f.readline()
                    continue
                prev_epoch = epoch_time

            for _ in range(num_sats):
                sat_line = f.readline()
                if not sat_line:
                    break
                sys_char = sat_line[0]
                if (
                    sys_char not in RINEX_OBS_CHANNEL_TO_USE
                    or sys_char not in _SYS_CHAR_TO_CONSTEL_MAP
                ):
                    continue
                prn_id = sat_line[:3]
                obs_list = sys_char_to_obs_type[sys_char]
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
                allowed = RINEX_OBS_CHANNEL_TO_USE.get(sys_char, set())
                for (obs_code, chan_id), meas in meas_map.items():
                    if f"{obs_code}{chan_id}" not in allowed:
                        continue
                    if not {"C", "L", "D", "S"}.issubset(meas):
                        continue

                    cn0 = meas["S"]
                    if cn0 is None or cn0 < GnssParameters.CNO_THRESHOLD:
                        continue

                    code = meas["C"]
                    phase_cycles = meas["L"]
                    doppler_hz = meas["D"]
                    if code is None or phase_cycles is None or doppler_hz is None:
                        continue

                    wavelength = _get_wavelength(
                        _SYS_CHAR_TO_CONSTEL_MAP[sys_char], prn, obs_code
                    )
                    phase_m = phase_cycles * wavelength
                    doppler_mps = -doppler_hz * wavelength

                    signal = SignalType(_SYS_CHAR_TO_CONSTEL_MAP[sys_char], obs_code)
                    signal.channel_id = chan_id

                    channel = GnssMeasurementChannel()
                    channel.addMeasurementFromObs(
                        epoch_time, signal, prn, code, phase_m, doppler_mps, cn0
                    )

                    result[epoch_time].add(channel)

    return result
