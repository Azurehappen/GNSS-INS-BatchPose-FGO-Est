import constants.parameters as params
from fgo_solver.gnss_ins_fgo import RtkInsFgo
from gnss_utils.cycle_slip_detection import detect_cycle_slips
from gnss_utils.gnss_data_utils import apply_base_corrections, apply_ephemerides_to_obs
from gnss_utils.rinex_nav_parser import parse_rinex_nav
from gnss_utils.rinex_obs_parser import parse_rinex_obs
from imu_utils.imu_data_utils import parse_ground_truth_log, parse_imu_log
from plotting import analyze_position_results


def _filter_epochs_with_gt(rover_obs, ground_truth_data):
    if not ground_truth_data:
        return rover_obs
    first_gt_epoch = min(ground_truth_data.keys())
    return {epoch: obs for epoch, obs in rover_obs.items() if epoch >= first_gt_epoch}


def main():
    nav_file = "data/tex_cup/brdm1290.19p"
    rover_file = "data/tex_cup/asterx4_rover.obs"
    base_file = "data/tex_cup/asterx4_base_1hz.obs"
    imu_file = "data/tex_cup/bosch_imu.log"
    ground_truth_file = "data/tex_cup/ground_truth.log"
    imu_params = params.TexCupBoschImuParams()

    ground_truth_data = {}
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

    rover_obs = _filter_epochs_with_gt(rover_obs, ground_truth_data)
    epochs_sorted = sorted(rover_obs.keys())
    if not epochs_sorted:
        print("No rover epochs available after ground truth alignment.")
        return

    start_idx = 200
    end_idx = min(900, len(epochs_sorted) - 1)
    if start_idx >= len(epochs_sorted):
        start_idx = 0

    solver = RtkInsFgo(
        imu_params,
        imu_data_list,
        window_size_s=5.0,
        show_progress=True,
    )

    results = solver.run(
        rover_obs,
        # start_idx=start_idx,
        # end_idx=end_idx,
    )
    if not results:
        print("Solver produced no results.")
        return

    summary = analyze_position_results(results, ground_truth_data, imu_params)

    if summary:
        print(f"Processed {summary.processed_epochs} epochs in solver log.")
        print(f"Evaluated {summary.evaluated_epochs} epochs with ground truth.")
        print(
            f"Horizontal error < {summary.horizontal_threshold_m:.1f} m for "
            f"{summary.horizontal_within_threshold_pct:.1f}% of evaluated epochs"
        )
        print(f"Horizontal error RMS: {summary.horizontal_rms:.3f} m")
        print(f"Max horizontal error: {summary.horizontal_max:.3f} m")
        print(f"Vertical error RMS: {summary.vertical_rms:.3f} m")
        print(
            "Mean pose covariance trace: "
            f"{summary.cov_trace_mean:.3e}, max: {summary.cov_trace_max:.3e}"
        )
        print(f"Saved position error plots to {summary.plot_path}")
        print(f"Saved trajectory comparison plot to {summary.trajectory_plot_path}")
    else:
        print("Ground truth unavailable for evaluated epochs; skipping error metrics.")


if __name__ == "__main__":
    main()
