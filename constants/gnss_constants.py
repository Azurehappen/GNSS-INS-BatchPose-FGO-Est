from enum import Enum
import numpy as np
import pandas as pd

SPEED_OF_LIGHT_MS = 299792458.0  # Speed of light in meters per second


class EarthConstants:
    # Earth radius in meters
    RADIUS_M = 6378137.0

    # Height of ionosphere in meters
    HEIGHT_OF_IONO_M = 350000.0

    # Standard gravitational acceleration in m/s^2
    G_M_PER_S2 = 9.80665

    # Earth rotation rate in rad/s
    OMEGA_E_RAD_PER_SEC = 7.2921151467e-5

    # Earth rotation rate vector in rad/s
    OMEGA_IEE_VEC = np.array([0.0, 0.0, OMEGA_E_RAD_PER_SEC])

    # Semi-major axis of Earth in meters
    A_M = 6378137.0

    # Semi-minor axis of Earth in meters
    B_M = 6356752.31424518

    # Flatness of ellipsoid
    F = 1.0 / 298.257223563

    # First eccentricity of ellipsoid
    E_FIRST = np.sqrt((A_M**2 - B_M**2) / (A_M**2))

    # Second eccentricity of ellipsoid
    E_SECOND = np.sqrt((A_M**2 - B_M**2) / (B_M**2))


class GpsConstants:
    # Maximum PRN number for GPS (Index 0 is not used)
    MAX_PRN = 32 + 1

    # RINEX 3.04 Obs Code Number: 1
    L1_FREQ_HZ = 1575.42e6

    # Wavelength of L1 frequency in meters
    L1_WAVELENGTH_M = SPEED_OF_LIGHT_MS / L1_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 2
    L2_FREQ_HZ = 1227.60e6

    # Wavelength of L2 frequency in meters
    L2_WAVELENGTH_M = SPEED_OF_LIGHT_MS / L2_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 5
    L5_FREQ_HZ = 1176.45e6

    # Wavelength of L5 frequency in meters
    L5_WAVELENGTH_M = SPEED_OF_LIGHT_MS / L5_FREQ_HZ

    # Mapping of observation code number to frequency
    ObsCodeToFreq = {1: L1_FREQ_HZ, 2: L2_FREQ_HZ, 5: L5_FREQ_HZ}

    # Mapping of observation code number to wavelength
    ObsCodeToWavelengthM = {
        1: L1_WAVELENGTH_M,
        2: L2_WAVELENGTH_M,
        5: L5_WAVELENGTH_M,
    }

    # GPS start time in UTC with nanosecond precision support
    START_TIME_IN_UTC = pd.Timestamp("1980-01-06T00:00:00")

    # Maximum duration to use ephemeris in seconds
    MAX_DURATION_TO_EPH = 7200.0

    # Geocentric gravitational constant in m^3/s^2
    MU = 3.986005e14

    # Earth rotation rate in rad/s
    OMEGA_DOT_E = 7.2921151467e-5

    # Relativistic correction term in s/s
    F = -4.442807633e-10


class GloConstants:
    # Maximum PRN number for GLONASS (Index 0 is not used)
    MAX_PRN = 24 + 1

    # RINEX 3.04 Obs Code Number: 1
    L1_FREQ_HZ = 1602.00e6

    # Wavelength of L1 frequency in meters
    L1_WAVELENGTH_M = SPEED_OF_LIGHT_MS / L1_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 2
    L2_FREQ_HZ = 1246.00e6

    # Wavelength of L2 frequency in meters
    L2_WAVELENGTH_M = SPEED_OF_LIGHT_MS / L2_FREQ_HZ

    # Mapping of observation code number to frequency
    ObsCodeToFreq = {1: L1_FREQ_HZ, 2: L2_FREQ_HZ}

    # Mapping of observation code number to wavelength
    ObsCodeToWavelength = {1: L1_WAVELENGTH_M, 2: L2_WAVELENGTH_M}

    # Maximum duration to use ephemeris in seconds
    MAX_DURATION_TO_EPH = 1800.0

    # Geocentric gravitational constant in m^3/s^2
    MU = 3.986004418e14

    # Earth rotation rate in rad/s
    OMEGA_DOT_E = 7.2921151467e-5

    # Relativistic correction term in s/s: -2*sqrt(p.glo.mu)/(p.c^2)
    F = -2 * np.sqrt(MU) / SPEED_OF_LIGHT_MS

    # Semi-major (equatorial) axis of the PZ-90 Earth's ellipsoid
    A_E = 6378136.0

    # Second zonal coefficient of spherical harmonic expansion
    C_20 = -108263e-9

    PrnToChannelNum = {
        1: 1,
        2: -4,
        3: 5,
        4: 6,
        5: 1,
        6: -4,
        7: 5,
        8: 6,
        9: -2,
        10: -7,
        11: 0,
        12: -1,
        13: -2,
        14: -7,
        15: 0,
        16: -1,
        17: 4,
        18: -3,
        19: 3,
        20: 2,
        21: 4,
        22: -3,
        23: 3,
        24: 2,
        25: -5,
    }

    def getObsCodeToFreq(self, obs_code: int, channel_num: int) -> float:
        if channel_num < -7 or channel_num > 12:
            raise ValueError(f"Invalid GLONASS channel number: {channel_num}")
        match obs_code:
            case 1:
                return 1602.0e6 + channel_num * 9.0 / 16.0
            case 4:
                return 1600.995e6
            case 2:
                return 1246.0e6 + channel_num * 7.0 / 16.0
            case 6:
                return 1248.06e6
            case 3:
                return 1202.025e6
        raise ValueError(
            f"Invalid GLONASS observation code: {obs_code}. Valid codes are 1, 2, 3, 4, or 6."
        )

    def getObsCodeToWavelengthM(self, obs_code: int, channel_num: int) -> float:
        freq_hz = self.getObsCodeToFreq(obs_code, channel_num)
        return SPEED_OF_LIGHT_MS / freq_hz


class GalConstants:
    # Maximum PRN number for Galileo (Index 0 is not used)
    MAX_PRN = 36 + 1

    # RINEX 3.04 Obs Code Number: 1
    E1_FREQ_HZ = 1575.42e6

    # Wavelength of E1 frequency in meters
    E1_WAVELENGTH_M = SPEED_OF_LIGHT_MS / E1_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 5
    E5A_FREQ_HZ = 1176.45e6

    # Wavelength of E5A frequency in meters
    E5A_WAVELENGTH_M = SPEED_OF_LIGHT_MS / E5A_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 7
    E5B_FREQ_HZ = 1207.14e6

    # Wavelength of E5B frequency in meters
    E5B_WAVELENGTH_M = SPEED_OF_LIGHT_MS / E5B_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 8
    E5_FREQ_HZ = 1191.795e6

    # Wavelength of E5 frequency in meters
    E5_WAVELENGTH_M = SPEED_OF_LIGHT_MS / E5_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 6
    E6_FREQ_HZ = 1278.75e6

    # Wavelength of E6 frequency in meters
    E6_WAVELENGTH_M = SPEED_OF_LIGHT_MS / E6_FREQ_HZ

    ObsCodeToFreq = {
        1: E1_FREQ_HZ,
        5: E5A_FREQ_HZ,
        7: E5B_FREQ_HZ,
        8: E5_FREQ_HZ,
        6: E6_FREQ_HZ,
    }

    ObsCodeToWavelengthM = {
        1: E1_WAVELENGTH_M,
        5: E5A_WAVELENGTH_M,
        7: E5B_WAVELENGTH_M,
        8: E5_WAVELENGTH_M,
        6: E6_WAVELENGTH_M,
    }

    # Galileo start time in UTC
    START_TIME_IN_UTC = pd.Timestamp("1999-08-21T23:59:47")

    # Maximum duration to use ephemeris in seconds
    MAX_DURATION_TO_EPH = 3600.0

    # Geocentric gravitational constant in m^3/s^2
    MU = 3.986004418e14

    # Earth rotation rate in rad/s
    OMEGA_DOT_E = 7.2921151467e-5

    # Relativistic correction term in s/s
    F = -4.442807309e-10


class BdsConstants:
    # Maximum PRN number for Beidou (Index 0 is not used)
    MAX_PRN = 65 + 1

    # RINEX 3.04 Obs Code Number: 2
    B12_FREQ_HZ = 1561.098e6

    # Wavelength of B1-2 frequency in meters
    B12_WAVELENGTH_M = SPEED_OF_LIGHT_MS / B12_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 1
    B1_FREQ_HZ = 1575.42e6

    # Wavelength of B1 frequency in meters
    B1_WAVELENGTH_M = SPEED_OF_LIGHT_MS / B1_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 5
    B2A_FREQ_HZ = 1176.45e6

    # Wavelength of B2a frequency in meters
    B2A_WAVELENGTH_M = SPEED_OF_LIGHT_MS / B2A_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 7
    B2B_FREQ_HZ = 1207.14e6

    # Wavelength of B2b frequency in meters
    B2B_WAVELENGTH_M = SPEED_OF_LIGHT_MS / B2B_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 8
    B2_FREQ_HZ = 1191.795e6

    # Wavelength of B2 frequency in meters
    B2_WAVELENGTH_M = SPEED_OF_LIGHT_MS / B2_FREQ_HZ

    # RINEX 3.04 Obs Code Number: 6
    B3_FREQ_HZ = 1268.52e6

    # Wavelength of B3 frequency in meters
    B3_WAVELENGTH_M = SPEED_OF_LIGHT_MS / B3_FREQ_HZ

    ObsCodeToFreq = {
        1: B1_FREQ_HZ,
        2: B12_FREQ_HZ,
        5: B2A_FREQ_HZ,
        7: B2B_FREQ_HZ,
        8: B2_FREQ_HZ,
        6: B3_FREQ_HZ,
    }
    ObsCodeToWavelengthM = {
        1: B1_WAVELENGTH_M,
        2: B12_WAVELENGTH_M,
        5: B2A_WAVELENGTH_M,
        7: B2B_WAVELENGTH_M,
        8: B2_WAVELENGTH_M,
        6: B3_WAVELENGTH_M,
    }

    # Beidou start time in UTC
    START_TIME_IN_UTC = pd.Timestamp("2006-01-01T00:00:00")

    # Maximum duration to use ephemeris in seconds
    MAX_DURATION_TO_EPH = 3600.0
