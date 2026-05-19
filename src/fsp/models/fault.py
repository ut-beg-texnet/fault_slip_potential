from dataclasses import dataclass
from typing import Optional


@dataclass
class FaultData:
    fault_id: str
    latitude: float
    longitude: float
    strike: float         # degrees, 0-360
    dip: float            # degrees, 0-90
    length_km: float
    friction_coefficient: Optional[float] = None

    def validate(self):
        if not (0 <= self.strike <= 360):
            raise ValueError(f"Strike must be 0-360 deg, got {self.strike}")
        if not (0 <= self.dip <= 90):
            raise ValueError(f"Dip must be 0-90 deg, got {self.dip}")
        if self.friction_coefficient is not None and not (0 < self.friction_coefficient <= 1):
            raise ValueError(f"Friction coefficient must be 0-1, got {self.friction_coefficient}")
