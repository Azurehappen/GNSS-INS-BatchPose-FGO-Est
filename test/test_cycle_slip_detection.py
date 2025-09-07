import constants.gnss_constants as gnssConst
from gnss_utils.cycle_slip_detection import detect_cycle_slips
from gnss_utils.gnss_data_utils import (
    GnssMeasurementChannel,
    SignalChannelId,
    SignalType,
    SatelliteId,
)
from gnss_utils.time_utils import GpsTime


def _make_channel(time, prn, signal_type, phase_m, wavelength_m):
    ch = GnssMeasurementChannel()
    ch.wavelength_m = wavelength_m
    ch.addMeasurementFromObs(
        time,
        SignalChannelId(SatelliteId(signal_type.constellation, prn), signal_type),
        f"G{prn:02d}",
        20000.0,
        phase_m,
        0.0,
        45.0,
    )
    ch.cycle_slip_status = gnssConst.CycleSlipType.NOT_AVAILABLE
    return ch


def test_geometry_free_no_slip():
    t1 = GpsTime(0.0)
    t2 = GpsTime(1.0)
    st1 = SignalType(gnssConst.Constellation.GPS, 1)
    st2 = SignalType(gnssConst.Constellation.GPS, 2)
    w1 = gnssConst.GpsConstants.ObsCodeToWavelengthM[1]
    w2 = gnssConst.GpsConstants.ObsCodeToWavelengthM[2]

    ch1_t1 = _make_channel(t1, 1, st1, 1000.0, w1)
    ch2_t1 = _make_channel(t1, 1, st2, 1000.0 - 0.5, w2)
    ch1_t2 = _make_channel(t2, 1, st1, 1000.1, w1)
    ch2_t2 = _make_channel(t2, 1, st2, 1000.1 - 0.5, w2)

    obs = {
        t1: {ch1_t1.signal_id: ch1_t1, ch2_t1.signal_id: ch2_t1},
        t2: {ch1_t2.signal_id: ch1_t2, ch2_t2.signal_id: ch2_t2},
    }

    detect_cycle_slips(obs)

    assert (
        ch1_t2.cycle_slip_status == gnssConst.CycleSlipType.NOT_DETECTED
        and ch2_t2.cycle_slip_status == gnssConst.CycleSlipType.NOT_DETECTED
    )


def test_geometry_free_with_slip():
    t1 = GpsTime(0.0)
    t2 = GpsTime(1.0)
    st1 = SignalType(gnssConst.Constellation.GPS, 1)
    st2 = SignalType(gnssConst.Constellation.GPS, 2)
    w1 = gnssConst.GpsConstants.ObsCodeToWavelengthM[1]
    w2 = gnssConst.GpsConstants.ObsCodeToWavelengthM[2]

    ch1_t1 = _make_channel(t1, 1, st1, 1000.0, w1)
    ch2_t1 = _make_channel(t1, 1, st2, 1000.0 - 0.5, w2)
    ch1_t2 = _make_channel(t2, 1, st1, 1000.1, w1)
    ch2_t2 = _make_channel(t2, 1, st2, 1000.1 - 0.5 + w2, w2)  # slip in L2

    obs = {
        t1: {ch1_t1.signal_id: ch1_t1, ch2_t1.signal_id: ch2_t1},
        t2: {ch1_t2.signal_id: ch1_t2, ch2_t2.signal_id: ch2_t2},
    }

    detect_cycle_slips(obs)

    assert (
        ch1_t2.cycle_slip_status == gnssConst.CycleSlipType.DETECTED
        and ch2_t2.cycle_slip_status == gnssConst.CycleSlipType.DETECTED
    )


def test_not_available_single_frequency():
    t1 = GpsTime(0.0)
    t2 = GpsTime(1.0)
    st1 = SignalType(gnssConst.Constellation.GPS, 1)
    st2 = SignalType(gnssConst.Constellation.GPS, 2)
    w1 = gnssConst.GpsConstants.ObsCodeToWavelengthM[1]
    w2 = gnssConst.GpsConstants.ObsCodeToWavelengthM[2]

    ch1_t1 = _make_channel(t1, 1, st1, 1000.0, w1)
    ch2_t1 = _make_channel(t1, 1, st2, 1000.0 - 0.5, w2)
    ch1_t2 = _make_channel(t2, 1, st1, 1000.1, w1)

    obs = {
        t1: {ch1_t1.signal_id: ch1_t1, ch2_t1.signal_id: ch2_t1},
        t2: {ch1_t2.signal_id: ch1_t2},
    }

    detect_cycle_slips(obs)

    assert ch1_t2.cycle_slip_status == gnssConst.CycleSlipType.NOT_AVAILABLE
