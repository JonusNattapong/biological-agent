"""Heuristic / Rule-based optimal reactive controller."""

from typing import Any, Dict
import numpy as np

from neurofly.controllers.base import BaseController
from neurofly.motor.decoder import MotorAction


class HeuristicController(BaseController):
    """Rule-based baseline combining tropotaxis (odor tracking) and looming threat evasion."""

    def __init__(self, max_speed: float = 6.5, max_turn: float = 0.35):
        self.max_speed = max_speed
        self.max_turn = max_turn
        self.evasion_timer = 0
        self.evasion_turn = 0.0

    @property
    def name(self) -> str:
        return "Heuristic (Rule-based)"

    def reset(self):
        self.evasion_timer = 0
        self.evasion_turn = 0.0

    def act(self, obs: Dict[str, Any]) -> MotorAction:
        v = obs["visual"]
        o = obs["olfactory"]
        t = obs["tactile"]

        # 1. Threat evasion takes top priority
        if v.looming_threat_left > 0.2 or v.looming_threat_right > 0.2:
            self.evasion_timer = 15
            # Turn away from threat
            if v.looming_threat_left > v.looming_threat_right:
                self.evasion_turn = self.max_turn  # Turn right
            else:
                self.evasion_turn = -self.max_turn  # Turn left

        if self.evasion_timer > 0:
            self.evasion_timer -= 1
            return MotorAction(
                forward_velocity=self.max_speed,
                angular_velocity=self.evasion_turn,
                feed_proboscis=False,
                raw_dn_activity=1.0,
            )

        # 2. Obstacle / collision avoidance
        if t.head_collision > 0.5 or t.body_contact > 0.5:
            return MotorAction(
                forward_velocity=self.max_speed * 0.4,
                angular_velocity=self.max_turn * 0.8,
                feed_proboscis=False,
                raw_dn_activity=0.5,
            )

        # 3. Odor tropotaxis: turn towards higher odor concentration
        if o.total_concentration > 0.02:
            # gradient = left - right
            # If left > right, turn left (-); if right > left, turn right (+)
            turn = float(np.clip(-o.gradient * 2.0, -self.max_turn, self.max_turn))
            speed = float(np.clip(self.max_speed * 0.7 + o.total_concentration * 2.0, 2.0, self.max_speed))
            return MotorAction(
                forward_velocity=speed,
                angular_velocity=turn,
                feed_proboscis=o.total_concentration > 0.6,
                raw_dn_activity=0.8,
            )

        # 4. Default exploration
        return MotorAction(
            forward_velocity=self.max_speed * 0.6,
            angular_velocity=float(np.random.uniform(-0.08, 0.08)),
            feed_proboscis=False,
            raw_dn_activity=0.3,
        )
