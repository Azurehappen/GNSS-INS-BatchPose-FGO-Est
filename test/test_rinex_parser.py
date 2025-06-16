import os
import sys
import tempfile
import unittest
import pandas as pd
from textwrap import dedent

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.rinex_parser import parse_rinex_nav
from utilities.gnss_data_structures import Constellation, EphemerisData, GpsEphemeris
from utilities.time_utils import GpsTime
from datetime import datetime


class TestRinexParser(unittest.TestCase):
    def _create_sample_file(self):
        data = dedent(
            """\
            DUMMY HEADER
            END OF HEADER
            G22 2019 05 09 18 00 00-6.980421021581e-04-6.480149750132e-12 0.000000000000e+00
                 1.800000000000e+01-3.368750000000e+01 5.209145553250e-09 8.788389075109e-01
                -1.873821020126e-06 7.250079303049e-03 9.683892130852e-06 5.153810482025e+03
                 4.104000000000e+05-3.166496753693e-08-1.635098977281e+00-8.381903171539e-08
                 9.269336384381e-01 1.783125000000e+02-1.351920716307e+00-8.349276352302e-09
                 3.621579424748e-10 1.000000000000e+00 2.052000000000e+03 0.000000000000e+00
                 2.000000000000e+00 0.000000000000e+00-1.816079020500e-08 1.800000000000e+01
                 4.032180000000e+05 4.000000000000e+00
            R01 2019 05 09 21 15 00 4.353187978268e-05 0.000000000000e+00 4.212000000000e+05
                 2.123729931641e+04-1.742321968079e+00-9.313225746155e-10 0.000000000000e+00
                -3.013212890625e+03 7.353792190552e-01-1.862645149231e-09 1.000000000000e+00
                 1.379858251953e+04 2.845211982727e+00-9.313225746155e-10 0.000000000000e+00
            E01 2019 05 09 00 30 00-6.006166804582e-04-8.157030606526e-12 0.000000000000e+00
                 6.700000000000e+01 1.878437500000e+02 2.445459006041e-09-3.277637191431e-01
                 8.631497621536e-06 1.712775556371e-04 9.058043360710e-06 5.440619096756e+03
                 3.474000000000e+05 2.421438694000e-08 1.231036872507e+00-6.705522537231e-08
                 9.880324324995e-01 1.562187500000e+02 1.966250675719e+00-5.353080120131e-09
                 7.357449324439e-11 5.170000000000e+02 2.052000000000e+03
                 3.120000000000e+00 0.000000000000e+00-2.095475792885e-09-2.328306436539e-09
                 3.480640000000e+05
            E01 2019 05 09 00 20 00-6.006091134623e-04-8.043343768804e-12 0.000000000000e+00
                 6.600000000000e+01 1.843125000000e+02 2.430101223470e-09-4.027602495868e-01
                 8.473172783852e-06 1.717308769003e-04 9.112060070038e-06 5.440620021820e+03
                 3.468000000000e+05 1.303851604462e-08 1.231040013392e+00-7.078051567078e-08
                 9.880323754457e-01 1.550312500000e+02 1.966862909861e+00-5.347722754118e-09
                 7.964617472573e-11 2.580000000000e+02 2.052000000000e+03      
                -1.000000000000e+00 0.000000000000e+00-2.095475792885e-09 0.000000000000e+00
                 3.479500000000e+05
            C01 2019 05 09 01 00 00 5.602736491710e-04 4.826539168334e-11 0.000000000000e+00
                 1.000000000000e+00-6.211875000000e+02-3.925877814283e-09-2.167503995504e+00
                -2.036709338427e-05 2.164692850783e-04 2.449378371239e-07 6.493401098251e+03
                 3.492000000000e+05-2.691522240639e-07-3.043904224429e+00-1.937150955200e-07
                 9.950466975175e-02-9.062500000000e-01 1.703501283060e+00 4.950206195958e-09
                 6.410981328820e-10 0.000000000000e+00 6.960000000000e+02 0.000000000000e+00
                 2.000000000000e+00 0.000000000000e+00 1.420000000000e-08-1.040000000000e-08
                 3.492000000000e+05 0.000000000000e+00
            """
        ).strip()
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w+")
        tmp.write(data)
        tmp.flush()
        return tmp

    def test_parse_rinex_nav(self):
        tmp = self._create_sample_file()
        eph_data = parse_rinex_nav(tmp.name)
        tmp.close()

        self.assertIn(22, eph_data.gps_ephemerides)
        self.assertIn(1, eph_data.glo_ephemerides)
        self.assertIn(1, eph_data.bds_ephemerides)
        self.assertIn(1, eph_data.gal_ephemerides)
        # Only one Galileo block should be stored (F-NAV)
        self.assertEqual(len(eph_data.gal_ephemerides[1]), 1)

        # GPS Ephemeris (PRN 22)
        epoch_time = eph_data.gps_ephemerides[22][0][0]
        gps_eph = eph_data.gps_ephemerides[22][0][1]
        self.assertEqual(
            epoch_time,
            GpsTime.fromDatetime(
                pd.Timestamp("2019-05-09 18:00:00"), Constellation.GPS
            ),
        )
        self.assertEqual(gps_eph.prn, 22)
        self.assertEqual(gps_eph.toc.gps_week, 2052)
        self.assertEqual(gps_eph.toc.gps_tow, 410400.0)
        self.assertAlmostEqual(gps_eph.sv_clock_bias, -6.980421021581e-04)
        self.assertAlmostEqual(gps_eph.sv_clock_drift, -6.480149750132e-12)
        self.assertAlmostEqual(gps_eph.sv_clock_drift_rate, 0.0)
        self.assertAlmostEqual(gps_eph.crs, -3.368750000000e01)
        self.assertAlmostEqual(gps_eph.delta_n, 5.209145553250e-09)
        self.assertAlmostEqual(gps_eph.m0, 8.788389075109e-01)

        # GLONASS Ephemeris (PRN 1)
        epoch_time = eph_data.glo_ephemerides[1][0][0]
        glo_eph = eph_data.glo_ephemerides[1][0][1]
        self.assertEqual(
            epoch_time,
            GpsTime.fromDatetime(
                pd.Timestamp(2019, 5, 9, 21, 15, 0), Constellation.GLO
            ),
        )
        self.assertEqual(glo_eph.prn, 1)
        self.assertEqual(glo_eph.toc.gps_week, 2052)
        self.assertAlmostEqual(glo_eph.sv_clock_bias, 4.353187978268e-05)
        self.assertAlmostEqual(glo_eph.sv_relative_freq_bias, 0.0)

        # Galileo Ephemeris (PRN 1)
        epoch_time = eph_data.gal_ephemerides[1][0][0]
        gal_eph = eph_data.gal_ephemerides[1][0][1]
        self.assertEqual(
            epoch_time,
            GpsTime.fromDatetime(pd.Timestamp(2019, 5, 9, 0, 20, 0), Constellation.GAL),
        )
        self.assertEqual(gal_eph.prn, 1)
        self.assertEqual(gal_eph.toc.gps_week, 2052)
        self.assertAlmostEqual(gal_eph.sv_clock_bias, -6.006166804582e-04)
        self.assertAlmostEqual(gal_eph.sv_clock_drift, -8.157030606526e-12)
        self.assertAlmostEqual(gal_eph.sv_clock_drift_rate, 0.0)

        # BeiDou Ephemeris (PRN 1)
        epoch_time = eph_data.bds_ephemerides[1][0][0]
        bds_eph = eph_data.bds_ephemerides[1][0][1]
        self.assertEqual(
            epoch_time,
            GpsTime.fromDatetime(pd.Timestamp(2019, 5, 9, 1, 0, 0), Constellation.BDS),
        )
        self.assertEqual(bds_eph.prn, 1)
        self.assertEqual(bds_eph.toc.gps_week, 2052)
        self.assertAlmostEqual(bds_eph.sv_clock_bias, 5.602736491710e-04)
        self.assertAlmostEqual(bds_eph.sv_clock_drift, 4.826539168334e-11)
        self.assertAlmostEqual(bds_eph.sv_clock_drift_rate, 0.0)

    def test_get_current_ephemeris(self):
        data = EphemerisData()
        t1 = GpsTime.fromWeekAndTow(1, 1000)
        t2 = GpsTime.fromWeekAndTow(1, 2000)
        eph1 = GpsEphemeris()
        eph2 = GpsEphemeris()
        eph1.prn = eph2.prn = 1
        data.add_ephemeris(Constellation.GPS, 1, t1, eph1)
        data.add_ephemeris(Constellation.GPS, 1, t2, eph2)

        q = GpsTime.fromWeekAndTow(1, 1500)
        result = data.get_current_ephemeris(Constellation.GPS, 1, q)
        self.assertIs(result, eph2)


if __name__ == "__main__":
    unittest.main()
