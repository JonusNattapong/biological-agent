"""MotorDecoder: Biologically grounded decoding of Descending Neurons into continuous locomotion."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch

from neurofly.brain.connectome import Connectome, Neuropil


@dataclass
class MotorAction:
    """Decoded physical motor action."""
    forward_velocity: float  # mm/s
    angular_velocity: float  # rad/s
    feed_proboscis: bool
    raw_dn_activity: float


class MotorDecoder:
    """Decodes population activity of Descending Neurons (DNs) into smooth insect flight kinematics.

    Features:
    - Population-normalized spike rate integration (eliminates saturation artifacts)
    - Bilateral asymmetry decoding: (Right DN - Left DN) governs steering torque
    - Critically damped muscle filter: models thoracic biomechanics and wing inertia
    - Realistic cruising speeds (2.5 to 5.5 mm/s)
    """

    def __init__(
        self,
        connectome: Connectome,
        base_speed: float = 3.2,
        max_speed: float = 6.2,
        max_turn_rate: float = 0.28,  # radians per physical step
        muscle_inertia: float = 0.82,  # biomechanical low-pass filter
    ):
        self.connectome = connectome
        self.base_speed = base_speed
        self.max_speed = max_speed
        self.max_turn_rate = max_turn_rate
        self.inertia = muscle_inertia

        # Descending neuron indices
        self.dn_indices = connectome.get_neuropil_indices(Neuropil.DESCENDING_MOTOR)
        num_dns = len(self.dn_indices)

        # Bilateral partition of descending motor neurons
        quarter_dn = max(1, num_dns // 4)
        self.left_dn = self.dn_indices[:quarter_dn]
        self.right_dn = self.dn_indices[quarter_dn : 2 * quarter_dn]
        self.forward_dn = self.dn_indices[2 * quarter_dn :]

        # Muscle kinematics state
        self.smooth_forward: float = base_speed
        self.smooth_turn: float = 0.0

    def reset(self):
        self.smooth_forward = self.base_speed
        self.smooth_turn = 0.0

    def decode(self, spikes: torch.Tensor) -> MotorAction:
        """Decode descending spikes into physical velocity and steering torque."""
        if self.dn_indices.numel() == 0:
            return MotorAction(
                forward_velocity=self.base_speed,
                angular_velocity=0.0,
                feed_proboscis=False,
                raw_dn_activity=0.0,
            )

        # Population-normalized firing rates (fraction of population spiking in this window)
        rate_left = float(spikes[self.left_dn].mean().item()) if self.left_dn.numel() > 0 else 0.0
        rate_right = float(spikes[self.right_dn].mean().item()) if self.right_dn.numel() > 0 else 0.0
        rate_fwd = float(spikes[self.forward_dn].mean().item()) if self.forward_dn.numel() > 0 else 0.0
        total_active_dns = float(spikes[self.dn_indices].sum().item())

        # Forward velocity: baseline cruising + forward drive
        target_fwd = np.clip(self.base_speed + (rate_fwd * 3.5) + (rate_left + rate_right) * 1.0, 1.2, self.max_speed)

        # Steering: Bilateral rate difference (Right DN - Left DN)
        # In Drosophila, turning is driven by asymmetric wing-beat amplitude (WBA) via unilateral DN activation
        dn_asymmetry = rate_right - rate_left
        target_turn = np.clip(dn_asymmetry * self.max_turn_rate * 2.5, -self.max_turn_rate, self.max_turn_rate)

        # Biomechanical low-pass filter (muscle & inertia damping)
        self.smooth_forward = self.inertia * self.smooth_forward + (1.0 - self.inertia) * target_fwd
        self.smooth_turn = self.inertia * self.smooth_turn + (1.0 - self.inertia) * target_turn

        feed = bool(rate_fwd > 0.4)

        return MotorAction(
            forward_velocity=float(self.smooth_forward),
            angular_velocity=float(self.smooth_turn),
            feed_proboscis=feed,
            raw_dn_activity=total_active_dns,
        )
