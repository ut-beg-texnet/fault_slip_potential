from dataclasses import dataclass
import numpy as np


@dataclass
class StressState:
    """Holds the stress state at a reference depth.

    principal_stresses: [Svert, Shmin, SHmax] in psi
    sH_azimuth: azimuth of SHmax in degrees clockwise from North
    """
    principal_stresses: np.ndarray  # shape (3,): [Svert, Shmin, SHmax]
    sH_azimuth: float

    def __post_init__(self):
        self.principal_stresses = np.asarray(self.principal_stresses, dtype=float)
