import os
import sys
import tempfile
from textwrap import dedent
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.rinex_obs_parser import parse_rinex_obs


class TestRinexObsParser(unittest.TestCase):
    def _create_sample(self):
        def slot(val):
            return f"{val:14.3f} 0"

        header = dedent(
            """
             3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE
G    8 C1C L1C D1C S1C C2L L2L D2L S2L                       SYS / # / OBS TYPES
                                                            END OF HEADER
> 2019 05 09 18 02 18.0000000  0 2
"""
        )
        line1 = (
            "G01"
            + slot(12345678.0)
            + slot(1.0)
            + slot(-0.1)
            + slot(45.0)
            + slot(12345678.5)
            + slot(1.1)
            + slot(-0.2)
            + slot(40.0)
            + "\n"
        )
        line2 = (
            "G02"
            + slot(22345678.0)
            + slot(2.0)
            + slot(-0.3)
            + slot(46.0)
            + slot(22345678.5)
            + slot(2.1)
            + slot(-0.4)
            + slot(41.0)
            + "\n"
        )
        data = header + line1 + line2
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
        self.assertEqual(len(channels), 4)
        combos = {(c.prn, c.signal_type.obs_code, c.signal_type.channel_id) for c in channels}
        self.assertIn((1, 1, "C"), combos)
        self.assertIn((1, 2, "L"), combos)
        self.assertIn((2, 1, "C"), combos)
        self.assertIn((2, 2, "L"), combos)


if __name__ == "__main__":
    unittest.main()
