import os
import sys
import tempfile
from textwrap import dedent
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.rinex_obs_parser import parse_rinex_obs


class TestRinexObsParser(unittest.TestCase):
    def _create_sample(self):
        data = dedent(
            """\
                 3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE
            G   12 C1C L1C D1C S1C C2W L2W D2W S2W C2L L2L D2L S2L      SYS / # / OBS TYPES 
            E   12 C1C L1C D1C S1C C5Q L5Q D5Q S5Q C7Q L7Q D7Q S7Q      SYS / # / OBS TYPES 
            R    8 C1C L1C D1C S1C C2C L2C D2C S2C                      SYS / # / OBS TYPES 
            C    4 C2I L2I D2I S2I                                      SYS / # / OBS TYPES
                                                                        END OF HEADER
            >> 2019 05 09 18 02 18.0000000  0 33
            C23  24823818.007 7 129264135.51407       105.162 7        42.219
            C27  21912100.435 7 114102054.46507      3839.397 7        46.344
            E03  27979941.231 5 147035580.18905      2700.667 5        34.625    27979941.684 5                      2016.891 5        30.625    27979941.969 5 112663635.87105      2069.309 5        31.750
            E09  25164121.539 6 132238353.65806      -300.355 6        38.688    25164124.606 6  98749429.63306      -224.491 6        39.250    25164123.429 6 101325495.96806      -230.037 6        40.063
            G17  22119480.918 8 116238657.90408      4731.156 8        48.438    22119479.778 5  90575571.58405      3686.610 5        31.875    22119480.056 6  90575556.58506      3686.594 6        40.063
            G18  22200304.783 7 116663387.87707      2306.389 7        43.563    22200303.635 4  90906527.21604      1797.190 4        24.969
            R14  21718988.211 7 115774426.76107      4857.673 7        44.000    21718992.259 7  90046793.93707      3778.193 7        42.594
            R17  23740050.974 4 127037816.84004      6755.738 4        29.844
            R23                                                                  19397871.832 6  80706526.53706      1213.339 6        40.406
            """
        ).strip()
        tmp = tempfile.NamedTemporaryFile(delete=False, mode="w+")
        tmp.write(data)
        tmp.flush()
        return tmp

    def test_simple_parse(self):
        tmp = self._create_sample()
        try:
            result = parse_rinex_obs(tmp.name)
        finally:
            tmp.close()
            os.unlink(tmp.name)

        self.assertEqual(len(result), 1)
        epoch = next(iter(result))
        channels = result[epoch]
        self.assertEqual(len(channels), 8 + 4 + 0)

        G18_1C_CHANNEL = next(
            c
            for c in channels
            if c.prn == 18
            and c.signal_type.obs_code == 1
            and c.signal_type.channel_id == "C"
        )
        self.assertEqual(G18_1C_CHANNEL.prn, 18)
        self.assertEqual(G18_1C_CHANNEL.signal_type.obs_code, 1)
        self.assertEqual(G18_1C_CHANNEL.signal_type.channel_id, "C")
        self.assertAlmostEqual(G18_1C_CHANNEL.code_m, 22200304.783)
        self.assertAlmostEqual(G18_1C_CHANNEL.cn0_dbhz, 43.563)


if __name__ == "__main__":
    unittest.main()
