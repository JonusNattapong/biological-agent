"""Antennae olfaction model for food odor concentration and chemical gradients."""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class OlfactoryStimulus:
    """Odor concentrations sensed by left and right antennae."""
    left_concentration: float  # [0.0, 1.0]
    right_concentration: float  # [0.0, 1.0]
    gradient: float  # left - right (> 0 means left is stronger)
    total_concentration: float


class AntennaeOlfaction:
    """Bilateral antennae model simulating olfactory receptor neurons (ORNs).

    Features:
    - Left and right antennae separated by antenna span (approx 2 mm equivalent in model units).
    - Odor plume diffusion: Exponential decay from food sources.
    - Senses spatial gradient for tropotaxis / chemotaxis.
    """

    def __init__(self, antenna_span: float = 4.0, odor_decay_sigma: float = 120.0):
        self.antenna_span = antenna_span
        self.sigma_sq = 2.0 * (odor_decay_sigma ** 2)

    def perceive(
        self,
        fly_pos: np.ndarray,
        fly_heading: float,
        food_positions: List[np.ndarray],
    ) -> OlfactoryStimulus:
        """Calculate odor concentration at left and right antennae positions."""
        if not food_positions:
            return OlfactoryStimulus(0.0, 0.0, 0.0, 0.0)

        # Antenna positions offset perpendicularly to heading
        # Normal vector to heading: [-sin(heading), cos(heading)]
        half_span = self.antenna_span / 2.0
        normal = np.array([-np.sin(fly_heading), np.cos(fly_heading)], dtype=np.float32)

        left_antenna_pos = fly_pos + normal * half_span
        right_antenna_pos = fly_pos - normal * half_span

        c_left = 0.0
        c_right = 0.0

        for f_pos in food_positions:
            d_l_sq = float(np.sum((left_antenna_pos - f_pos) ** 2))
            d_r_sq = float(np.sum((right_antenna_pos - f_pos) ** 2))

            # Gaussian plume intensity
            c_left += float(np.exp(-d_l_sq / self.sigma_sq))
            c_right += float(np.exp(-d_r_sq / self.sigma_sq))

        c_left = float(np.clip(c_left, 0.0, 1.0))
        c_right = float(np.clip(c_right, 0.0, 1.0))
        gradient = float(c_left - c_right)
        total = float((c_left + c_right) / 2.0)

        return OlfactoryStimulus(
            left_concentration=c_left,
            right_concentration=c_right,
            gradient=gradient,
            total_concentration=total,
        )
