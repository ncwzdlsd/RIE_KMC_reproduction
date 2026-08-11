from dataclasses import dataclass
from enum import Enum
import math
from paper_parameters import (
    PAPER_DISSOLUTION_PROBABILITY,
    PAPER_SONICATION_RADIUS_NM,
)


class SonicationEventType(str, Enum):
    CORROSION = "sonication_corrosion"


@dataclass(frozen=True)
class SonicationParameters:
    """Paper-inspired stochastic corrosion at the particle-solution interface.

    The paper specifies a 1 nm sphere and a 10% dissolution probability, but
    does not report the event frequency.  ``event_rate`` belongs to an
    independent Poisson condition clock per eligible interface center
    (s^-1 per site); it is not a chemical KMC reaction propensity and must be
    fitted before interpreting the acoustic exposure quantitatively.
    """

    event_rate: float
    radius_nm: float = PAPER_SONICATION_RADIUS_NM
    dissolution_probability: float = PAPER_DISSOLUTION_PROBABILITY

    def __post_init__(self):
        if self.event_rate < 0.0 or not math.isfinite(self.event_rate):
            raise ValueError("event_rate must be finite and non-negative")
        if self.radius_nm <= 0.0 or not math.isfinite(self.radius_nm):
            raise ValueError("radius_nm must be finite and positive")
        if not 0.0 <= self.dissolution_probability <= 1.0:
            raise ValueError("dissolution_probability must be between 0 and 1")
