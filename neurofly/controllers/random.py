"""Random baseline controller (Brownian random walk)."""

from typing import Any, Dict
import numpy as np

from neurofly.controllers.base import BaseController
from neurofly.motor.decoder import MotorAction


class RandomController(BaseController):
    """Uniform / Brownian random walk agent."""

    def __init__(self, max_speed: float = 6.0, max_turn: float = 0.3):
        self.max_speed = max_speed
        self.max_turn = max_turn
        self.curr_turn = 0.0

    @property
    def name(self) -> str:
        return "Random Walk"

    def reset(self):
        self.curr_turn = 0.0

    def act(self, obs: Dict[str, Any]) -> MotorAction:
        # Correlated Brownian motion
        turn_delta = np.random.uniform(-0.1, 0.1)
        self.curr_turn = np.clip(self.curr_turn * 0.7 + turn_delta, -self.max_turn, self.max_turn)
        fwd = np.random.uniform(1.0, self.max_speed)

        return MotorAction(
            forward_velocity=float(fwd),
            angular_velocity=float(self.curr_turn),
            feed_proboscis=False,
            raw_dn_activity=0.0,
        )
