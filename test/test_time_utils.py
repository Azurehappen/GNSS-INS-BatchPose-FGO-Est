import os
import sys
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.time_utils import GpsTime
from utilities.gnss_data_structures import Constellation


class TestGpsTime(unittest.TestCase):
    def test_time_equivalence(self):
        gps_epoch = GpsTime.fromWeekAndTow(2370, 592752, Constellation.GPS)

        gal_epoch = GpsTime.fromWeekAndTow(1346, 592752, Constellation.GAL)

        bds_epoch = GpsTime.fromWeekAndTow(1014, 592738, Constellation.BDS)

        # GLONASS time: UTC time 2025-06-14 20:38:54.000
        glo_datetime = pd.Timestamp("2025-06-14 20:38:54.000")
        glo_epoch = GpsTime.fromDatetime(glo_datetime, Constellation.GLO)

        # Assert all times are equivalent
        self.assertEqual(gps_epoch, gal_epoch)
        self.assertEqual(gps_epoch, bds_epoch)
        self.assertEqual(gps_epoch, glo_epoch)

    def test_fractional_seconds_roundtrip(self):
        ts = pd.Timestamp("2025-06-14 20:38:54") + pd.Timedelta(microseconds=123456)
        gps_time = GpsTime.fromDatetime(ts, Constellation.GLO)
        result = gps_time.toDatetimeInUtc()
        # GPS time is 18 seconds ahead of UTC. ``fromDatetime`` adds this offset
        # and ``toDatetimeInUtc`` removes it so the original timestamp is recovered.
        self.assertAlmostEqual((result - ts).total_seconds(), 0.0, places=5)
        self.assertAlmostEqual(
            result.value % 1_000_000_000, ts.value % 1_000_000_000, delta=1000
        )

        gps_time = GpsTime.fromDatetime(ts, Constellation.BDS)
        result = gps_time.toDatetimeInUtc()
        self.assertAlmostEqual((ts - result).total_seconds(), 4.0, places=5)

        gps_time = GpsTime.fromDatetime(ts, Constellation.GAL)
        result = gps_time.toDatetimeInUtc()
        self.assertAlmostEqual((ts - result).total_seconds(), 18.0, places=5)


if __name__ == "__main__":
    unittest.main()
