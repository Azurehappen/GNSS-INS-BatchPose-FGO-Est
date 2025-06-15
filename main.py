from utilities.rinex_parser import parse_rinex_nav
from utilities.gnss_data_structures import Constellation
from utilities.time_utils import GpsTime


if __name__ == "__main__":
    nav_file = "data/tex_cup/brdm1290.19p"
    eph_data = parse_rinex_nav(nav_file)

    # Example usage: query GPS PRN 1 ephemeris at first epoch
    query_time = eph_data.gps_ephemerides[1][0][0]
    eph = eph_data.get_current_ephemeris(Constellation.GPS, 1, query_time)
    print("Example ephemeris for GPS PRN 1 at", query_time, ":")
    print(eph.__dict__)
