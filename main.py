from gnss_utils.rinex_nav_parser import parse_rinex_nav
from gnss_utils.rinex_obs_parser import parse_rinex_obs
from gnss_utils.gnss_data_utils import apply_ephemerides_to_obs, apply_base_corrections
from gnss_utils.cycle_slip_detection import detect_cycle_slips
from constants.gnss_constants import Constellation
from imu_utils.imu_data_utils import parse_ground_truth_log, parse_imu_log
import constants.parameters as params

if __name__ == "__main__":
    nav_file = "data/tex_cup/brdm1290.19p"
    rover_file = "data/tex_cup/asterx4_rover.obs"
    base_file = "data/tex_cup/asterx4_base_1hz.obs"
    imu_file = "data/tex_cup/bosch_imu.log"
    ground_truth_file = "data/tex_cup/ground_truth.log"
    imu_params = params.TexCupBoschImuParams()

    ground_truth_data = None
    if ground_truth_file is not None:
        ground_truth_data = parse_ground_truth_log(ground_truth_file)

    eph_data = parse_rinex_nav(nav_file)
    rover_obs = parse_rinex_obs(rover_file)
    base_obs = parse_rinex_obs(base_file, interval=30)
    imu_data_list = parse_imu_log(imu_file, imu_params.z_up)

    apply_ephemerides_to_obs(rover_obs, eph_data)
    eph_data.resetIndexLookup()
    apply_ephemerides_to_obs(base_obs, eph_data)
    apply_base_corrections(rover_obs, base_obs)
    detect_cycle_slips(rover_obs)

    # Example usage: query GPS PRN 1 ephemeris at first epoch
    query_time = eph_data.gps_ephemerides[1][0][0]
    eph = eph_data.getCurrentEphemeris(Constellation.GPS, 1, query_time)
    print("Example ephemeris for GPS PRN 1 at", query_time, ":")
    print(eph.__dict__)

    print(
        f"Loaded {len(rover_obs)} rover epochs and {len(base_obs)} base epochs after applying ephemerides"
    )
