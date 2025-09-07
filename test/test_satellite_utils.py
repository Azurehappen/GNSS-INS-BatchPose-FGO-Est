import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from gnss_utils.gnss_data_utils import (
    Constellation,
    EphemerisData,
    GpsEphemeris,
    GnssMeasurementChannel,
    SignalChannelId,
    SignalType,
    SatelliteId,
)
from gnss_utils.time_utils import GpsTime
from gnss_utils.gnss_data_utils import apply_ephemerides_to_obs


class TestSatelliteUtils(unittest.TestCase):
    def test_channel_removed_when_no_eph(self):
        epoch = GpsTime.fromWeekAndTow(0, 0)
        ch_id = SignalChannelId(
            SatelliteId(Constellation.GPS, 1),
            SignalType(Constellation.GPS, 1, "C"),
        )
        ch = GnssMeasurementChannel()
        ch.addMeasurementFromObs(
            time=epoch,
            signal_id=ch_id,
            svid="G01",
            code_m=100.0,
            phase_m=50.0,
            doppler_mps=0.0,
            cn0_dbhz=45.0,
        )
        obs = {epoch: {ch_id: ch}}
        eph = EphemerisData()
        apply_ephemerides_to_obs(obs, eph)
        self.assertFalse(obs)

    def test_basic_clock_correction(self):
        eph = GpsEphemeris()
        eph.prn = 1
        eph.toc = GpsTime.fromWeekAndTow(0, 0)
        eph.toe = GpsTime.fromWeekAndTow(0, 0)
        eph.sqrtA = np.sqrt(26560e3)
        eph.ecc = 0.01
        eph.i0 = 0.94
        eph.omega0 = 0.0
        eph.omega = 0.0
        eph.omega_dot = 0.0
        eph.delta_n = 0.0
        eph.m0 = 0.0
        eph.cuc = eph.cus = eph.cic = eph.cis = eph.crc = eph.crs = 0.0
        eph.idot = 0.0
        eph.sv_clock_bias = 0.0
        eph.sv_clock_drift = 0.0
        eph.sv_clock_drift_rate = 0.0
        eph.tgd = 1e-8

        epoch = GpsTime.fromWeekAndTow(0, 60)
        ch_id = SignalChannelId(
            SatelliteId(Constellation.GPS, 1),
            SignalType(Constellation.GPS, 1, "C"),
        )
        ch = GnssMeasurementChannel()
        ch.addMeasurementFromObs(
            time=epoch,
            signal_id=ch_id,
            svid="G01",
            code_m=2e7,
            phase_m=1000.0,
            doppler_mps=-10.0,
            cn0_dbhz=45.0,
        )

        ch.computeSatelliteInformation(eph)

        self.assertIsNotNone(ch.sat_pos_ecef_m)
        # code measurement should be corrected by clock_bias (which includes group delay)
        expected = 2e7 + ch.sat_clock_bias_m
        self.assertAlmostEqual(ch.code_m, expected, places=2)
        self.assertAlmostEqual(ch.phase_m, 1000.0 + ch.sat_clock_bias_m, places=2)


if __name__ == "__main__":
    unittest.main()
