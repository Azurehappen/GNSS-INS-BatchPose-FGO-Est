import unittest
from datetime import datetime, timezone
from utilities.time_utils import GpsTime
from utilities.gnss_data_structures import Constellation


class TestGpsTime(unittest.TestCase):
    def test_time_equivalence(self):
        gps_epoch = GpsTime.fromWeekAndTow(2370, 592752, Constellation.GPS)

        gal_epoch = GpsTime.fromWeekAndTow(1346, 592752, Constellation.GAL)

        bds_epoch = GpsTime.fromWeekAndTow(1014, 592738, Constellation.BDS)

        # GLONASS time: UTC time 2025-06-14 20:38:54.000
        glo_datetime = datetime(2025, 6, 14, 20, 38, 54)
        glo_epoch = GpsTime.fromDatetime(glo_datetime, Constellation.GLO)

        # Assert all times are equivalent
        self.assertEqual(gps_epoch, gal_epoch)
        self.assertEqual(gps_epoch, bds_epoch)
        self.assertEqual(gps_epoch, glo_epoch)


if __name__ == "__main__":
    unittest.main()
