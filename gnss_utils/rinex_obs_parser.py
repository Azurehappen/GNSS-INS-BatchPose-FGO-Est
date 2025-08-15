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
from gnss_utils.gnss_data_utils import (
    Constellation,
    GnssMeasurementChannel,
    SignalChannelId,
    SignalType,
)
from constants.parameters import GnssParameters, RINEX_OBS_CHANNEL_TO_USE
from gnss_utils.time_utils import GpsTime


_SYS_CHAR_TO_CONSTEL_MAP: Dict[str, Tuple[Constellation, bool]] = {
    "G": (Constellation.GPS, GnssParameters.enable_gps),
    "R": (Constellation.GLO, GnssParameters.enable_glonass),
    "E": (Constellation.GAL, GnssParameters.enable_galileo),
    "C": (Constellation.BDS, GnssParameters.enable_beidou),
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


def _get_wavelength_m(constellation: Constellation, prn: int, obs_code: int) -> float:
    """Return wavelength in metres for the given observation code."""

    if constellation == Constellation.GPS:
        return gnssConst.GpsConstants.ObsCodeToWavelengthM[obs_code]
    if constellation == Constellation.GAL:
        return gnssConst.GalConstants.ObsCodeToWavelengthM[obs_code]
    if constellation == Constellation.BDS:
        return gnssConst.BdsConstants.ObsCodeToWavelengthM[obs_code]
    if constellation == Constellation.GLO:
        ch = gnssConst.GloConstants.PrnToChannelNum.get(prn)
        if ch is None:
            raise ValueError(f"Unknown GLONASS channel number for PRN {prn}")
        return gnssConst.GloConstants().getObsCodeToWavelengthM(obs_code, ch)
    raise ValueError(f"Unsupported constellation {constellation}")


def _parse_epoch_header(line: str) -> Tuple[GpsTime, int]:
    """Parse epoch header line beginning with ``>``."""

    # Some RINEX files use a double ``>>`` prefix.  Handle both cases
    # by stripping all leading ``>`` characters before splitting.
    parts = line.lstrip(">").split()
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
    obs_channel_to_use: Dict[str, set[str]] = RINEX_OBS_CHANNEL_TO_USE,
) -> Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]]:
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
    Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]]
        Mapping from :class:`GpsTime` to measurement channels keyed by
        :class:`~utilities.gnss_data_structures.SignalChannelId`.
    """

    result: Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]] = defaultdict(
        dict
    )

    with open(file_path, "r") as f:
        sys_char_to_obs_type = _parse_header(f)

        first_epoch: GpsTime | None = None
        prev_epoch: GpsTime | None = None

        next_line: str | None = None
        while True:
            line = next_line or f.readline()
            next_line = None
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
                sat_line = next_line or f.readline()
                next_line = None
                if not sat_line:
                    break
                sys_char = sat_line[0]
                if (
                    sys_char not in obs_channel_to_use
                    or sys_char not in _SYS_CHAR_TO_CONSTEL_MAP
                    # Check if constellation is enabled
                    or not _SYS_CHAR_TO_CONSTEL_MAP[sys_char][1]
                ):
                    continue
                prn_id = sat_line[:3]
                obs_list = sys_char_to_obs_type[sys_char]

                # Calculate the total number of characters needed to represent all observation types
                # Each observation type is represented by a 16-character field in the RINEX file
                needed_len = len(obs_list) * 16

                # Extract the observation data from the current line, skipping the first 3 characters
                # (which represent the satellite identifier) and removing any trailing newline characters
                data_str = sat_line[3:].rstrip("\n")
                while len(data_str) < needed_len:
                    cont = f.readline()
                    if not cont or not cont.startswith(" "):
                        # Check if it's a continuation line
                        next_line = cont
                        break
                    data_str += cont[3:].rstrip("\n")

                values = [data_str[i : i + 16] for i in range(0, len(data_str), 16)]
                parsed_vals = [v[:14].strip() for v in values]

                # Initialize a dictionary to store parsed measurement data for each observation type.
                # The key is a tuple `(obs_code, chan_id)`:
                #   - `obs_code` is the observation code.
                #   - `chan_id` is the signal channel identifier (e.g., 'C', 'W').
                # The value is another dictionary where:
                #   - The key is the measurement type ('C' for code, 'L' for phase, 'D' for Doppler, 'S' for signal strength).
                #   - The value is the parsed measurement value (float) or `None` if the value is missing.
                meas_map: Dict[Tuple[int, str], Dict[str, float]] = {}
                for t, val_str in zip(obs_list, parsed_vals):
                    if not val_str:
                        val = None
                    else:
                        val = float(val_str)
                    key = (int(t[1:-1]), t[-1])
                    meas_map.setdefault(key, {})[t[0]] = val

                prn = int(prn_id[1:])
                allowed = obs_channel_to_use.get(sys_char, set())
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
                    if code < 100.0 or phase_cycles < 100.0:
                        # Code and phase must be valid measurements
                        continue

                    # TODO: Set wavelength_m in Satellite position computation, for GLO, the channel
                    # number should be obtained from the Ephemeris data directly.
                    wavelength_m = _get_wavelength_m(
                        _SYS_CHAR_TO_CONSTEL_MAP[sys_char][0], prn, obs_code
                    )
                    phase_m = phase_cycles * wavelength_m
                    doppler_mps = -doppler_hz * wavelength_m

                    signal = SignalType(
                        _SYS_CHAR_TO_CONSTEL_MAP[sys_char][0], obs_code, chan_id
                    )
                    signal_id = SignalChannelId(prn, signal)

                    channel = GnssMeasurementChannel()
                    channel.wavelength_m = wavelength_m
                    channel.addMeasurementFromObs(
                        epoch_time,
                        signal_id,
                        f"{sys_char}{prn:02d}",
                        code,
                        phase_m,
                        doppler_mps,
                        cn0,
                    )

                    result[epoch_time][signal_id] = channel

    print(f"Parsed {len(result)} epochs observations from {file_path}")
    return result
