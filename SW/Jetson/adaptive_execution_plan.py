"""Final execution contract. Values are SI in phase-entry pre-posture frame."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class LegPlan:
    landing: tuple[float, float, float]
    clearance: float
    apex: float
    transfer: float

@dataclass(frozen=True)
class AdaptiveExecutionPlan:
    session_id: int
    observation_sequence: int
    sequence: int
    plan_id: int
    swing_mask: int
    requested_gait: int
    duration: float
    roll: float
    pitch: float
    body_height: float
    twist: tuple[float, float, float, float]
    legs: tuple[LegPlan, ...]
    execute: bool = True

    def validate(self):
        if not 0 < self.session_id < 2**32:
            raise ValueError('A live nonzero STM32 session is required')
        if any(not 0 <= v < 65536 for v in (self.observation_sequence, self.sequence, self.plan_id)):
            raise ValueError('sequence/plan IDs must be uint16')
        if not 0 < self.swing_mask < 64 or self.requested_gait not in (0, 1) or len(self.legs) != 6:
            raise ValueError('invalid gait/leg contract')
        values = (self.duration, self.roll, self.pitch, self.body_height, *self.twist,
                  *(v for leg in self.legs for v in (*leg.landing, leg.clearance, leg.apex, leg.transfer)))
        if not all(math.isfinite(x) for x in values):
            raise ValueError('execution values must be finite')
        if not (.6 if self.requested_gait else .5) <= self.duration <= 1.4:
            raise ValueError('phase duration outside v4 limits')
        if abs(self.roll) > math.radians(15) or abs(self.pitch) > math.radians(15) or abs(self.body_height) > .03:
            raise ValueError('absolute body reference outside STM32 limits')
        if abs(self.twist[0]) > .1 or abs(self.twist[1]) > .07 or self.twist[2] != 0 or abs(self.twist[3]) > math.radians(18):
            raise ValueError('applied twist outside STM32 limits')
        for leg in self.legs:
            if any(abs(x) > .75 for x in leg.landing) or not .04 <= leg.clearance <= .18 or not .3 <= leg.apex <= .7 or not .35 <= leg.transfer <= .65:
                raise ValueError('invalid final leg target')
        return self
