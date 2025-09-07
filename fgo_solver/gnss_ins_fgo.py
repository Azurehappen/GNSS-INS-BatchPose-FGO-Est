import gtsam
import numpy as np
from typing import List, Optional
from constants.parameters import BASE_POS_ECEF, BASE_ECEF_TO_NED_ROT_MAT


# Define the error function for the GNSS code measurement factor
@staticmethod
def error_code_meas(
    measurement: float,
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    """
    Nonlinear error function.
    Computes the residual.

    Args:
        measurement (float): GNSS double difference measurement.
        this (CustomFactor): GTSAM custom factor instance.
        values (Values): Current state estimates.
        jacobians (Optional[List[np.ndarray]]): Optional list to store Jacobian matrices.

    Returns:
        np.ndarray: Residual error array of shape (1,).
    """

    key = this.keys()[0]
    pos_ned = values.atPose3(key).translation()

    # Convert position from NED frame (relative to base station) to ECEF frame
    pos_ecef = BASE_ECEF_TO_NED_ROT_MAT().T @ pos_ned + BASE_POS_ECEF
