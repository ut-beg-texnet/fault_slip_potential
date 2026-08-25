from dataclasses import dataclass


@dataclass
class FaultData:
    fault_id: str
    latitude: float
    longitude: float
    strike: float         # degrees, 0-360
    dip: float            # degrees, 0-90
    length_km: float

    def validate(self):
        if not (0 <= self.strike <= 360):
            raise ValueError(f"Strike must be 0-360 deg, got {self.strike}")
        if not (0 <= self.dip <= 90):
            raise ValueError(f"Dip must be 0-90 deg, got {self.dip}")
