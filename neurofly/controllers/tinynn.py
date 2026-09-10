"""Tiny Neural Network (MLP) baseline controller."""

from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn

from neurofly.controllers.base import BaseController
from neurofly.motor.decoder import MotorAction


class TinyNNController(BaseController, nn.Module):
    """A lightweight 2-layer reactive MLP baseline mapping sensory inputs directly to motor actions."""

    def __init__(self, seed: int = 42):
        BaseController.__init__(self)
        nn.Module.__init__(self)
        torch.manual_seed(seed)

        # Inputs:
        # 1. Left visual intensity (1)
        # 2. Right visual intensity (1)
        # 3. Left looming threat (1)
        # 4. Right looming threat (1)
        # 5. Left odor conc (1)
        # 6. Right odor conc (1)
        # 7. Odor gradient (1)
        # 8. Tactile collision (1)
        # Total = 8 inputs
        # Outputs: forward_velocity, angular_velocity
        self.net = nn.Sequential(
            nn.Linear(8, 24),
            nn.Tanh(),
            nn.Linear(24, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
            nn.Tanh(),
        )

    @property
    def name(self) -> str:
        return "Tiny Neural Network (MLP)"

    def reset(self):
        pass

    def act(self, obs: Dict[str, Any]) -> MotorAction:
        v = obs["visual"]
        o = obs["olfactory"]
        t = obs["tactile"]

        inp = torch.tensor(
            [
                v.left_intensity,
                v.right_intensity,
                v.looming_threat_left,
                v.looming_threat_right,
                o.left_concentration,
                o.right_concentration,
                o.gradient,
                t.head_collision,
            ],
            dtype=torch.float32,
        )

        with torch.no_grad():
            out = self.net(inp).numpy()

        # Map tanh [-1, 1] to physical velocities
        # forward: [0.5, 6.0]
        fwd = float((out[0] + 1.0) * 0.5 * 5.5 + 0.5)
        # angular: [-0.3, 0.3]
        turn = float(out[1] * 0.3)

        return MotorAction(
            forward_velocity=fwd,
            angular_velocity=turn,
            feed_proboscis=False,
            raw_dn_activity=0.0,
        )
