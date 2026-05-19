from dataclasses import dataclass, field
from typing import Dict


@dataclass
class HydrologyParams:
    aquifer_thickness: float       # feet
    porosity: float                # fraction
    permeability: float            # millidarcies
    fluid_density: float           # kg/m³
    dynamic_viscosity: float       # Pa·s
    fluid_compressibility: float   # 1/Pa
    rock_compressibility: float    # 1/Pa
    plus_minus: Dict[str, float] = field(default_factory=dict)
    n_iterations: int = 750

    def validate(self):
        if self.aquifer_thickness <= 0:
            raise ValueError("Aquifer thickness must be positive")
        if not (0 < self.porosity < 1):
            raise ValueError("Porosity must be between 0 and 1")
        if self.permeability <= 0:
            raise ValueError("Permeability must be positive")
