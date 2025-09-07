import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gnss_utils.gnss_data_utils import (
    apply_base_corrections,
    GnssMeasurementChannel,
    SignalChannelId,
    SignalType,
    Constellation,
    SatelliteId,
)
from gnss_utils.time_utils import GpsTime
from constants.parameters import BASE_POS_ECEF
from gnss_utils import satellite_utils


class TestBaseCorrections(unittest.TestCase):
    def test_basic_correction_and_removal(self):
        base_epoch = GpsTime.fromWeekAndTow(0, 0)
        ch_id = SignalChannelId(
            SatelliteId(Constellation.GPS, 1),
            SignalType(Constellation.GPS, 1, "C"),
        )
        base_ch = GnssMeasurementChannel()
        base_ch.addMeasurementFromObs(
            time=base_epoch,
            signal_id=ch_id,
            svid="G01",
            code_m=100.0,
            phase_m=50.0,
            doppler_mps=-2.0,
            cn0_dbhz=40.0,
        )
        base_ch.sat_pos_ecef_m = np.array([1000.0, 2000.0, 3000.0])
        base_obs = {base_epoch: {ch_id: base_ch}}

        rover_epoch = GpsTime.fromWeekAndTow(0, 10)
        rover_ch = GnssMeasurementChannel()
        rover_ch.addMeasurementFromObs(
            time=rover_epoch,
            signal_id=ch_id,
            svid="G01",
            code_m=120.0,
            phase_m=60.0,
            doppler_mps=-3.0,
            cn0_dbhz=30.0,
        )
        rover_obs = {rover_epoch: {ch_id: rover_ch}}

        other_id = SignalChannelId(
            SatelliteId(Constellation.GPS, 2),
            SignalType(Constellation.GPS, 1, "C"),
        )
        other_ch = GnssMeasurementChannel()
        other_ch.addMeasurementFromObs(
            time=rover_epoch,
            signal_id=other_id,
            svid="G02",
            code_m=110.0,
            phase_m=55.0,
            doppler_mps=-1.0,
            cn0_dbhz=30.0,
        )
        rover_obs[rover_epoch][other_id] = other_ch

        original_code = rover_ch.code_m
        original_phase = rover_ch.phase_m

        apply_base_corrections(rover_obs, base_obs)

        base_pos = np.asarray(BASE_POS_ECEF)
        geometric_range = np.linalg.norm(
            base_pos - base_ch.sat_pos_ecef_m
        ) + satellite_utils.sagnac_correction(base_pos, base_ch.sat_pos_ecef_m)

        expected_correction = base_ch.code_m - geometric_range
        self.assertAlmostEqual(rover_ch.correction_code_m, expected_correction)
        self.assertAlmostEqual(
            rover_ch.correction_phase_m, base_ch.phase_m - geometric_range
        )
        self.assertAlmostEqual(rover_ch.correction_doppler_mps, base_ch.doppler_mps)

        self.assertAlmostEqual(rover_ch.code_m, original_code - expected_correction)
        self.assertAlmostEqual(
            rover_ch.phase_m, original_phase - (base_ch.phase_m - geometric_range)
        )

        self.assertNotIn(other_id, rover_obs[rover_epoch])


if __name__ == "__main__":
    unittest.main()
