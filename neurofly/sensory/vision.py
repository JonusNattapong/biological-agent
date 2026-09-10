"""Compound eye vision model and looming stimulus detector."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class VisualStimulus:
    """Retinal activations from compound eyes."""
    left_intensity: float  # [0.0, 1.0] left eye average luminance / food cue
    right_intensity: float  # [0.0, 1.0] right eye average luminance / food cue
    left_ommatidia: np.ndarray  # (num_ommatidia,) array of sector intensities
    right_ommatidia: np.ndarray  # (num_ommatidia,) array of sector intensities
    looming_threat_left: float  # [0.0, 1.0] optical expansion rate on left eye
    looming_threat_right: float  # [0.0, 1.0] optical expansion rate on right eye


class CompoundEyeVision:
    """Drosophila compound eye model.

    Features:
    - 2 lateral compound eyes spanning -135° to +135° azimuth.
    - Ommatidia array per eye detecting light & food cues.
    - Looming detector: Computes angular expansion rate of approaching threats.
    """

    def __init__(self, num_ommatidia_per_eye: int = 16, fov_deg: float = 135.0):
        self.num_ommatidia = num_ommatidia_per_eye
        self.fov_rad = np.radians(fov_deg)

        # Angular centers for left eye (-fov to 0) and right eye (0 to +fov)
        self.left_angles = np.linspace(-self.fov_rad, -0.05, num_ommatidia_per_eye)
        self.right_angles = np.linspace(0.05, self.fov_rad, num_ommatidia_per_eye)

        # Threat memory for optical expansion tracking
        self.prev_threat_distance: float = 9999.0
        self.prev_threat_angle: float = 0.0

    def reset(self):
        self.prev_threat_distance = 9999.0
        self.prev_threat_angle = 0.0

    def perceive(
        self,
        fly_pos: np.ndarray,
        fly_heading: float,
        food_positions: List[np.ndarray],
        threat_pos: Optional[np.ndarray] = None,
        threat_radius: float = 15.0,
    ) -> VisualStimulus:
        """Calculate visual stimulus on left and right compound eyes.

        Args:
            fly_pos: [x, y] coordinates of the fly
            fly_heading: heading angle in radians (-pi to pi)
            food_positions: list of [x, y] coordinates of visible food patches
            threat_pos: [x, y] coordinates of predator, if active
            threat_radius: radius of predator
        """
        left_ommatidia = np.zeros(self.num_ommatidia, dtype=np.float32)
        right_ommatidia = np.zeros(self.num_ommatidia, dtype=np.float32)

        # 1. Food visual cues (Greenish / attractant wavelength)
        for f_pos in food_positions:
            rel_vec = f_pos - fly_pos
            dist = np.linalg.norm(rel_vec)
            if dist < 1e-4 or dist > 250.0:
                continue

            # Angle relative to fly heading
            angle_to_food = np.arctan2(rel_vec[1], rel_vec[0]) - fly_heading
            # Normalize to [-pi, pi]
            angle_to_food = (angle_to_food + np.pi) % (2 * np.pi) - np.pi

            intensity = float(np.clip(1.0 - (dist / 250.0), 0.0, 1.0))

            if angle_to_food < 0:  # Left eye
                # Find closest ommatidium
                diffs = np.abs(self.left_angles - angle_to_food)
                best_idx = int(np.argmin(diffs))
                left_ommatidia[best_idx] = max(left_ommatidia[best_idx], intensity)
            else:  # Right eye
                diffs = np.abs(self.right_angles - angle_to_food)
                best_idx = int(np.argmin(diffs))
                right_ommatidia[best_idx] = max(right_ommatidia[best_idx], intensity)

        # 2. Looming threat detector (Predator optical expansion)
        looming_left = 0.0
        looming_right = 0.0

        if threat_pos is not None:
            rel_vec = threat_pos - fly_pos
            dist = float(np.linalg.norm(rel_vec))
            angle_to_threat = np.arctan2(rel_vec[1], rel_vec[0]) - fly_heading
            angle_to_threat = (angle_to_threat + np.pi) % (2 * np.pi) - np.pi

            if dist < 200.0 and dist > 0.0:
                # Looming expansion speed: -d(dist)/dt
                approach_speed = max(0.0, self.prev_threat_distance - dist)
                # Angular size ~ 2 * radius / dist
                angular_size = (2.0 * threat_radius) / max(dist, 10.0)
                looming_intensity = float(np.clip((approach_speed * 0.2 + angular_size * 0.8), 0.0, 1.0))

                if angle_to_threat < 0:
                    looming_left = looming_intensity
                else:
                    looming_right = looming_intensity

            self.prev_threat_distance = dist
            self.prev_threat_angle = angle_to_threat
        else:
            self.prev_threat_distance = 9999.0

        left_intensity = float(np.mean(left_ommatidia))
        right_intensity = float(np.mean(right_ommatidia))

        return VisualStimulus(
            left_intensity=left_intensity,
            right_intensity=right_intensity,
            left_ommatidia=left_ommatidia,
            right_ommatidia=right_ommatidia,
            looming_threat_left=looming_left,
            looming_threat_right=looming_right,
        )
