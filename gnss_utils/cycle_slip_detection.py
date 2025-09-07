"""Cycle slip detection using geometry-free combination for dual-frequency measurements."""

from __future__ import annotations

from typing import Dict

import constants.gnss_constants as gnssConst
from constants.parameters import GnssParameters
from gnss_utils.gnss_data_utils import (
    GnssMeasurementChannel,
    SignalChannelId,
    SignalType,
    SatelliteId,
)
from gnss_utils.time_utils import GpsTime


def _group_by_sat(
    channels: Dict[SignalChannelId, GnssMeasurementChannel]
) -> Dict[SatelliteId, Dict[SignalType, GnssMeasurementChannel]]:
    """Group observation channels by satellite."""
    sat_map: Dict[SatelliteId, Dict[SignalType, GnssMeasurementChannel]] = {}
    for scid, ch in channels.items():
        sat_map.setdefault(scid.satellite_id, {})[scid.signal_type] = ch
    return sat_map


def detect_cycle_slips(
    rover_obs: Dict[GpsTime, Dict[SignalChannelId, GnssMeasurementChannel]]
) -> None:
    """Detect cycle slips in-place for rover observations.

    Parameters
    ----------
    rover_obs: Dict
        Rover observations organised by epoch and signal channel.

    Notes
    -----
    Cycle slip detection is performed between consecutive epochs for each
    satellite. Only satellites with at least two common frequency
    measurements between epochs are processed. For other cases, the cycle
    slip status remains ``CycleSlipType.NOT_AVAILABLE``.
    """

    epochs = sorted(rover_obs.keys())
    if len(epochs) < 2:
        return

    prev_epoch = epochs[0]
    for epoch in epochs[1:]:
        prev_map = _group_by_sat(rover_obs[prev_epoch])
        curr_map = _group_by_sat(rover_obs[epoch])

        for sat_key, curr_signals in curr_map.items():
            prev_signals = prev_map.get(sat_key)
            if prev_signals is None:
                for ch in curr_signals.values():
                    ch.cycle_slip_status = gnssConst.CycleSlipType.NOT_AVAILABLE
                continue

            common_types = set(prev_signals.keys()) & set(curr_signals.keys())
            if len(common_types) < 2:
                for ch in curr_signals.values():
                    ch.cycle_slip_status = gnssConst.CycleSlipType.NOT_AVAILABLE
                continue

            # Mark non-common current signals as not available
            for st in set(curr_signals.keys()) - common_types:
                curr_signals[st].cycle_slip_status = gnssConst.CycleSlipType.NOT_AVAILABLE

            # Prepare common signals for detection
            for st in common_types:
                curr_signals[st].cycle_slip_status = gnssConst.CycleSlipType.NOT_DETECTED

            sorted_types = sorted(common_types, key=lambda s: s.obs_code)
            ref_type = sorted_types[0]
            ref_prev = prev_signals[ref_type]
            ref_curr = curr_signals[ref_type]

            for other_type in sorted_types[1:]:
                prev_ch = prev_signals[other_type]
                curr_ch = curr_signals[other_type]

                delta_prev = ref_prev.phase_m - prev_ch.phase_m
                delta_curr = ref_curr.phase_m - curr_ch.phase_m
                diff = abs(delta_prev - delta_curr)

                threshold = GnssParameters.CYCLE_SLIP_THRESHOLD_M
                if diff >= threshold:
                    ref_curr.cycle_slip_status = gnssConst.CycleSlipType.DETECTED
                    curr_ch.cycle_slip_status = gnssConst.CycleSlipType.DETECTED
                else:
                    if curr_ch.cycle_slip_status != gnssConst.CycleSlipType.DETECTED:
                        curr_ch.cycle_slip_status = gnssConst.CycleSlipType.NOT_DETECTED
                    if ref_curr.cycle_slip_status != gnssConst.CycleSlipType.DETECTED:
                        ref_curr.cycle_slip_status = gnssConst.CycleSlipType.NOT_DETECTED

        prev_epoch = epoch
