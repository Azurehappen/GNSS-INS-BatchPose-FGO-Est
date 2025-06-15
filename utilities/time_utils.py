from datetime import datetime, timedelta
import constants.gnss_constants as gnssConst
from utilities.gnss_data_structures import Constellation


GAL_START_TIME_OFFSET_TO_GPS = (
    gnssConst.GalConstants.START_TIME_IN_UTC - gnssConst.GpsConstants.START_TIME_IN_UTC
)
BDS_START_TIME_OFFSET_TO_GPS = (
    gnssConst.BdsConstants.START_TIME_IN_UTC - gnssConst.GpsConstants.START_TIME_IN_UTC
)


class GpsTime:

    def __init__(self, gps_timestamp: float):
        """Initialize with absolute GPS timestamp (seconds since GPS epoch)."""
        self.gps_timestamp = (
            gps_timestamp  # seconds since GPS epoch (1980-01-06 00:00:00 UTC)
        )
        self.gps_week = int(gps_timestamp // 604800)
        self.gps_tow = gps_timestamp % 604800  # Time of week in seconds

    @classmethod
    def fromWeekAndTow(
        cls, week: int, tow: float, constellation: Constellation = Constellation.GPS
    ):
        """Initialize with week and time-of-week (seconds) for the given constellation."""
        # Convert to absolute datetime (GPS epoch + offset)
        if constellation == Constellation.GPS:
            return cls(week * 604800 + tow)  # 604800 seconds in a week
        elif constellation == Constellation.GAL:
            # Leap seconds between 1980 and 1999 is 13s.
            dt = GAL_START_TIME_OFFSET_TO_GPS + timedelta(weeks=week, seconds=tow)
            return cls(dt.total_seconds() + 13.0)  # Adjust for leap seconds
        elif constellation == Constellation.BDS:
            # Leap seconds between 1980 and 2060 is 14s.
            dt = BDS_START_TIME_OFFSET_TO_GPS + timedelta(weeks=week, seconds=tow)
            return cls(dt.total_seconds() + 14.0)
        elif constellation == Constellation.GLO:
            # GLONASS uses a different epoch and week system
            raise NotImplementedError(
                "GLONASS time handling not implemented for week and TOW."
            )

    @classmethod
    def fromDatetime(
        cls, datetime: datetime, constellation: Constellation = Constellation.GPS
    ):
        """Initialize from constellation system datatime."""
        # Ensure both are offset-naive
        datetime = datetime.replace(tzinfo=None)

        if constellation == Constellation.GPS:
            dt = datetime - gnssConst.GpsConstants.START_TIME_IN_UTC
            return cls(dt.total_seconds())
        elif constellation == Constellation.GLO:
            # GLONASS is synced with UTC where UTS is 18 seconds behind GPS.
            dt = (
                datetime
                + timedelta(seconds=18)
                - gnssConst.GpsConstants.START_TIME_IN_UTC
            )
            return cls(dt.total_seconds())
        elif constellation == Constellation.GAL:
            # Galileo epoch time since 1999-08-22 00:00:00 which is aligned with GPS epoch.
            dt = datetime - gnssConst.GpsConstants.START_TIME_IN_UTC
            return cls(dt.total_seconds())
        elif constellation == Constellation.BDS:
            # BDS epoch time since 2006-01-01 00:00:00 which is aligned with UTC epoch.
            dt = (
                datetime
                + timedelta(seconds=14)
                - gnssConst.GpsConstants.START_TIME_IN_UTC
            )
            return cls(dt.total_seconds())

    def toDatetimeInUtc(self):
        """Return a Python datetime (UTC) for this GPS time."""
        return (
            gnssConst.GpsConstants.START_TIME_IN_UTC
            + timedelta(seconds=self.gps_timestamp)
            + timedelta(seconds=18)  # Leap seconds adjustment for GPS to UTC
        )

    def __eq__(self, other):
        # Allow for floating point precision
        return (
            isinstance(other, GpsTime)
            and abs(self.gps_timestamp - other.gps_timestamp) <= 1e-6
        )

    def __lt__(self, other):
        if isinstance(other, GpsTime):
            return self.gps_timestamp < other.gps_timestamp
        else:
            raise TypeError(
                f"Unsupported type for comparison: {type(other)}. Must be GpsTime."
            )

    def __le__(self, other):
        if isinstance(other, GpsTime):
            return self.gps_timestamp <= other.gps_timestamp
        else:
            raise TypeError(
                f"Unsupported type for comparison: {type(other)}. Must be GpsTime."
            )

    def __gt__(self, other):
        if isinstance(other, GpsTime):
            return self.gps_timestamp > other.gps_timestamp
        else:
            raise TypeError(
                f"Unsupported type for comparison: {type(other)}. Must be GpsTime."
            )

    def __ge__(self, other):
        if isinstance(other, GpsTime):
            return self.gps_timestamp >= other.gps_timestamp
        else:
            raise TypeError(
                f"Unsupported type for comparison: {type(other)}. Must be GpsTime."
            )

    def __hash__(self):
        # Use a unique combination for hashing
        return hash(round(self.gps_timestamp, 6))
        # rounding gps_timestamp to avoid floating issues

    def __repr__(self):
        return f"GpsTime(week={self.gps_week}, tow={self.gps_tow:.3f})"

    def __sub__(self, other):
        if isinstance(other, GpsTime):
            # Return difference in seconds between two GpsTime
            return self.gps_timestamp - other.gps_timestamp
        else:
            raise TypeError(
                f"Unsupported type for subtraction: {type(other)}. Must be GpsTime."
            )

    def __add__(self, other):
        raise NotImplementedError("Addition of gps_timestamp is not supported.")
