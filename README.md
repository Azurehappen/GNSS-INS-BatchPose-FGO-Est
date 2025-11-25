# Batch-RTK-INS

GNSS/INS factor-graph optimizer for RTK-level positioning in challenging (urban) environments. The solver blends IMU preintegration, robust double-differenced GNSS code/phase factors, and optional ISAM2 incremental updates, with a final full-trajectory Levenberg–Marquardt batch solve if desired.

## What it does
- Parses rover/base RINEX observations and broadcast ephemerides, applies base corrections, and detects cycle slips.
- Builds a factor graph with IMU preintegration and DD GNSS code/phase factors (pivot selection, elevation-based noise, ambiguity tracking).
- Supports two workflows: (1) incremental ISAM2 for fast per-epoch estimates; (2) full-graph batch LM for a final refined trajectory (configurable via `RtkInsFgo` flags).
- Generates ENU error/trajectory plots when ground truth is available.

## Quick start
1. Install deps (example): `pip install -r requirements.txt`
2. Place data in `data/tex_cup/` or adjust paths in `main.py`:
   - `nav_file`: broadcast nav (RINEX `.19p` in example)
   - `rover_file`, `base_file`: rover/base RINEX obs
   - `imu_file`: IMU log; pick the matching `TexCup*ImuParams` in `constants/parameters.py`
   - `ground_truth_file` (optional) for metrics/plots
3. Run: `python main.py`
4. Outputs:
   - Console summary of horizontal/vertical errors (if GT provided)
   - Plots in `plotting/position_errors.html` and `plotting/trajectory_comparison.html`
   - Debug log at `logs/solver_debug.log`

## Configuration highlights
- Solver flags (in `main.py` when constructing `RtkInsFgo`):
  - `use_isam`: enable incremental ISAM2 (good for online/initialization)
  - `final_batch_opt`: run full-trajectory LM after all epochs
  - `isam_relinearize_skip`: ISAM2 relinearization cadence (lower for aggressive urban motion)
- GNSS/IMU noise and initialization parameters live in `constants/parameters.py`.
- Debugging: set `debug_times` when creating `RtkInsFgo` to log per-epoch residuals/factors.

## Notes
- Double-differencing removes receiver/clock biases; pivot selection is elevation/CN0-aware.
- WLS provides position seeds when enough DD signals are available; IMU propagation seeds attitude/velocity.
