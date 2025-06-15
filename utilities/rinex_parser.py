"""Simple RINEX navigation file parser."""

from datetime import datetime, timezone
import re

from utilities.gnss_data_structures import (
    Constellation,
    EphemerisData,
    GpsEphemeris,
    GloEphemeris,
    GalEphemeris,
    BdsEphemeris,
)
from utilities.time_utils import GpsTime


_FLOAT_RE = re.compile(r"[-+]?\d*\.\d+(?:[Ee][+-]?\d+)?")


def _parse_float_fields(line: str, count: int = 4):
    """Extract floating point numbers from a RINEX navigation line.

    RINEX navigation records use fixed-width formatting without explicit
    delimiters.  Using simple slicing is brittle so we instead search for
    floating point patterns in the line.  The first ``count`` values are
    returned and any missing entries are filled with ``NaN``.
    """

    line = line.replace("D", "E")
    matches = _FLOAT_RE.findall(line)
    values = [float(v) for v in matches[:count]]
    while len(values) < count:
        values.append(float("nan"))
    return values


def _parse_gps_block(lines):
    eph = GpsEphemeris()
    first = lines[0]
    eph.prn = int(first[1:3])

    year = int(first[4:8])
    month = int(first[9:11])
    day = int(first[12:14])
    hour = int(first[15:17])
    minute = int(first[18:20])
    second = int(first[21:23])  # TODO: Use seconds and nanoseconds with pd.Timestamp
    eph.toc = GpsTime.fromDatetime(
        datetime(year, month, day, hour, minute, second),
        Constellation.GPS,
    )
    eph.sv_clock_bias = float(first[23:42])
    eph.sv_clock_drift = float(first[42:61])
    eph.sv_clock_drift_rate = float(first[61:80])

    eph.iode, eph.crs, eph.delta_n, eph.m0 = _parse_float_fields(lines[1])
    eph.cuc, eph.ecc, eph.cus, eph.sqrtA = _parse_float_fields(lines[2])
    toe, eph.cic, eph.omega0, eph.cis = _parse_float_fields(lines[3])
    eph.i0, eph.crc, eph.omega, eph.omega_dot = _parse_float_fields(lines[4])
    (
        eph.idot,
        eph.code_on_L2,
        gps_week,
        eph.l2p_data_flag,
    ) = _parse_float_fields(lines[5])
    (
        eph.sv_accuracy,
        sv_health,
        eph.tgd,
        eph.iodc,
    ) = _parse_float_fields(lines[6])
    trans_time, eph.fit_interval, _, _ = _parse_float_fields(lines[7])

    eph.gps_week = int(gps_week)
    eph.toe = GpsTime.fromWeekAndTow(eph.gps_week, toe, Constellation.GPS)
    eph.transmission_time = GpsTime.fromWeekAndTow(
        eph.gps_week, trans_time, Constellation.GPS
    )

    eph.sv_health = int(sv_health)
    if eph.sv_health != 0:
        return None

    return eph


def _parse_glo_block(lines):
    eph = GloEphemeris()
    first = lines[0]
    eph.prn = int(first[1:3])

    year = int(first[4:8])
    month = int(first[9:11])
    day = int(first[12:14])
    hour = int(first[15:17])
    minute = int(first[18:20])
    second = int(first[21:23])  # TODO: Use seconds and nanoseconds with pd.Timestamp
    eph.toc = GpsTime.fromDatetime(
        datetime(year, month, day, hour, minute, second),
        Constellation.GLO,
    )
    eph.sv_clock_bias = float(first[23:42])
    eph.sv_relative_freq_bias = float(first[42:61])
    eph.message_frame_time = float(first[61:80])

    eph.x_pos, eph.x_vel, eph.x_acc, health = _parse_float_fields(lines[1])
    eph.y_pos, eph.y_vel, eph.y_acc, eph.freq_number = _parse_float_fields(lines[2])
    eph.z_pos, eph.z_vel, eph.z_acc, eph.age_of_oper_info = _parse_float_fields(
        lines[3]
    )

    eph.health = int(health)
    if eph.health != 0:
        return None

    return eph


def _parse_gal_block(lines):
    eph = GalEphemeris()
    first = lines[0]
    eph.prn = int(first[1:3])

    year = int(first[4:8])
    month = int(first[9:11])
    day = int(first[12:14])
    hour = int(first[15:17])
    minute = int(first[18:20])
    second = int(first[21:23])  # TODO: Use seconds and nanoseconds with pd.Timestamp
    eph.toc = GpsTime.fromDatetime(
        datetime(year, month, day, hour, minute, second),
        Constellation.GAL,
    )
    eph.sv_clock_bias = float(first[23:42])
    eph.sv_clock_drift = float(first[42:61])
    eph.sv_clock_drift_rate = float(first[61:80])

    eph.iod_nav, eph.crs, eph.delta_n, eph.m0 = _parse_float_fields(lines[1])
    eph.cuc, eph.ecc, eph.cus, eph.sqrtA = _parse_float_fields(lines[2])
    toe, eph.cic, eph.omega0, eph.cis = _parse_float_fields(lines[3])
    eph.i0, eph.crc, eph.omega, eph.omega_dot = _parse_float_fields(lines[4])
    (
        eph.idot,
        data_source,
        gal_week,
        _,
    ) = _parse_float_fields(lines[5])
    (
        eph.sisa,
        sv_health,
        eph.bgd_e1e5a,
        eph.bgd_e1e5b,
    ) = _parse_float_fields(lines[6])
    trans_time, _, _, _ = _parse_float_fields(lines[7])

    eph.data_source = int(data_source)
    eph.gal_week = int(gal_week)
    eph.toe = GpsTime.fromWeekAndTow(eph.gal_week, toe, Constellation.GAL)
    eph.transmission_time = GpsTime.fromWeekAndTow(
        eph.gal_week, trans_time, Constellation.GAL
    )

    eph.sv_health = int(sv_health)
    # Determine health validity flags according to Galileo ICD
    eph.e1b_is_valid = (eph.sv_health & 0x1) == 0
    eph.e1b_is_health = ((eph.sv_health >> 1) & 0x3) == 0
    eph.e5a_is_valid = ((eph.sv_health >> 3) & 0x1) == 0
    eph.e5a_is_health = ((eph.sv_health >> 4) & 0x3) == 0
    eph.e5b_is_valid = ((eph.sv_health >> 6) & 0x1) == 0
    eph.e5b_is_health = ((eph.sv_health >> 7) & 0x3) == 0

    if eph.data_source != 258 or eph.sv_health != 0:
        return None

    return eph


def _parse_bds_block(lines):
    eph = BdsEphemeris()
    first = lines[0]
    eph.prn = int(first[1:3])

    year = int(first[4:8])
    month = int(first[9:11])
    day = int(first[12:14])
    hour = int(first[15:17])
    minute = int(first[18:20])
    second = int(first[21:23])  # TODO: Use seconds and nanoseconds with pd.Timestamp
    eph.toc = GpsTime.fromDatetime(
        datetime(year, month, day, hour, minute, second),
        Constellation.BDS,
    )
    eph.sv_clock_bias = float(first[23:42])
    eph.sv_clock_drift = float(first[42:61])
    eph.sv_clock_drift_rate = float(first[61:80])

    eph.aode, eph.crs, eph.delta_n, eph.m0 = _parse_float_fields(lines[1])
    eph.cuc, eph.ecc, eph.cus, eph.sqrtA = _parse_float_fields(lines[2])
    toe, eph.cic, eph.omega0, eph.cis = _parse_float_fields(lines[3])
    eph.i0, eph.crc, eph.omega, eph.omega_dot = _parse_float_fields(lines[4])
    (
        eph.idot,
        _,
        bds_week,
        _,
    ) = _parse_float_fields(lines[5])
    (
        eph.sv_accuracy,
        sv_health,
        eph.tgd1,
        eph.tgd2,
    ) = _parse_float_fields(lines[6])
    trans_time, eph.aodc, _, _ = _parse_float_fields(lines[7])

    eph.bds_week = int(bds_week)
    eph.sv_health = int(sv_health)
    eph.toe = GpsTime.fromWeekAndTow(eph.bds_week, toe, Constellation.BDS)
    eph.transmission_time = GpsTime.fromWeekAndTow(
        eph.bds_week, trans_time, Constellation.BDS
    )

    if eph.sv_health != 0:
        return None

    return eph


def parse_rinex_nav(file_path: str) -> EphemerisData:
    """Parse a RINEX navigation file and return EphemerisData."""
    eph_data = EphemerisData()

    with open(file_path, "r") as f:
        # Skip header
        for line in f:
            if "END OF HEADER" in line:
                break

        while True:
            line = f.readline()
            if not line:
                break
            if len(line.strip()) == 0:
                continue

            const = line[0]
            if const == "G":
                body = [line.rstrip("\n")] + [
                    f.readline().rstrip("\n") for _ in range(7)
                ]
                eph = _parse_gps_block(body)
                if eph:
                    eph_data.add_ephemeris(Constellation.GPS, eph.prn, eph.toc, eph)
            elif const == "R":
                body = [line.rstrip("\n")] + [
                    f.readline().rstrip("\n") for _ in range(3)
                ]
                eph = _parse_glo_block(body)
                if eph:
                    eph_data.add_ephemeris(Constellation.GLO, eph.prn, eph.toc, eph)
            elif const == "E":
                body = [line.rstrip("\n")] + [
                    f.readline().rstrip("\n") for _ in range(7)
                ]
                eph = _parse_gal_block(body)
                if eph:
                    eph_data.add_ephemeris(Constellation.GAL, eph.prn, eph.toc, eph)
            elif const == "C":
                body = [line.rstrip("\n")] + [
                    f.readline().rstrip("\n") for _ in range(7)
                ]
                eph = _parse_bds_block(body)
                if eph:
                    eph_data.add_ephemeris(Constellation.BDS, eph.prn, eph.toc, eph)
            else:
                # Unsupported constellation
                pass

    return eph_data
