from dataclasses import dataclass
import numpy as np


@dataclass
class WellData:
    well_id: str
    latitude: float
    longitude: float
    days: np.ndarray    # days from injection start (float64)
    rates: np.ndarray   # injection rates in bbl/day (float64)

    def __post_init__(self):
        self.days = np.asarray(self.days, dtype=float)
        self.rates = np.asarray(self.rates, dtype=float)
