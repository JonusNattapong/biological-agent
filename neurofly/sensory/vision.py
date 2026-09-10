"""Compound eye vision model with retinotopic ommatidia grids and looming detection."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class VisualStimulus:
    """Retinal activations and ommatidia visual feeds from compound eyes."""
    left_intensity: float
    right_intensity: float
    left_ommatidia: np.ndarray  # 1D array of horizontal sectors
    right_ommatidia: np.ndarray
    looming_threat_left: float
    looming_threat_right: float
    # 2D Ommatidia Pixel Grids (8x8 for Left Eye, 8x8 for Right Eye)
    left_eye_grid: List[float]  # 64 normalized values [0.0, 1.0]
    right_eye_grid: List[float]


class CompoundEyeVision:
    """Drosophila compound eye model simulating lateral ommatidia lenses and retinotopy.

    Features:
    - 2 lateral compound eyes spanning -135° to +135° azimuth, -45° to +45° elevation.
    - Generates 8x8 ommatidia pixel grids for Left and Right eyes (Inside Fly vision).
    - Detects light sources (phototaxis), food contrast, and looming optical expansion.
    """

    def __init__(self, num_ommatidia_per_eye: int = 16, fov_deg: float = 135.0):
        self.num_ommatidia = num_ommatidia_per_eye
        self.fov_rad = np.radians(fov_deg)

        self.left_angles = np.linspace(-self.fov_rad, -0.05, num_ommatidia_per_eye)
        self.right_angles = np.linspace(0.05, self.fov_rad, num_ommatidia_per_eye)

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
        light_pos: Optional[np.ndarray] = None,
    ) -> VisualStimulus:
        """Calculate visual stimulus and 8x8 ommatidia pixel grids for left and right eyes."""
        left_ommatidia = np.zeros(self.num_ommatidia, dtype=np.float32)
        right_ommatidia = np.zeros(self.num_ommatidia, dtype=np.float32)

        # 8x8 2D grids (elevation rows 0..7, azimuth cols 0..7)
        grid_left = np.full((8, 8), 0.12, dtype=np.float32)   # ambient room baseline
        grid_right = np.full((8, 8), 0.12, dtype=np.float32)

        # 1. Window Light Source (Sunlight / Lamp beam)
        if light_pos is not None:
            rel_l = light_pos[:2] - fly_pos[:2]
            d_l = float(np.linalg.norm(rel_l))
            if d_l > 1.0:
                angle_l = (np.arctan2(rel_l[1], rel_l[0]) - fly_heading + np.pi) % (2 * np.pi) - np.pi
                l_intensity = float(np.clip(1.2 - (d_l / 280.0), 0.1, 1.0))

                if angle_l < 0:  # Left eye
                    col = int(np.clip((angle_l / -self.fov_rad) * 7.0, 0, 7))
                    grid_left[2:6, col] = np.clip(grid_left[2:6, col] + l_intensity * 0.7, 0.0, 1.0)
                else:  # Right eye
                    col = int(np.clip((angle_l / self.fov_rad) * 7.0, 0, 7))
                    grid_right[2:6, col] = np.clip(grid_right[2:6, col] + l_intensity * 0.7, 0.0, 1.0)

        # 2. Food visual cues
        for f_pos in food_positions:
            rel_vec = f_pos[:2] - fly_pos[:2]
            dist = float(np.linalg.norm(rel_vec))
            if dist < 1e-4 or dist > 220.0:
                continue

            angle_to_food = (np.arctan2(rel_vec[1], rel_vec[0]) - fly_heading + np.pi) % (2 * np.pi) - np.pi
            intensity = float(np.clip(1.0 - (dist / 220.0), 0.0, 1.0))

            if angle_to_food < 0:
                diffs = np.abs(self.left_angles - angle_to_food)
                best_idx = int(np.argmin(diffs))
                left_ommatidia[best_idx] = max(left_ommatidia[best_idx], intensity)

                col = int(np.clip(((-angle_to_food) / self.fov_rad) * 7.0, 0, 7))
                grid_left[3:6, col] = np.clip(grid_left[3:6, col] + intensity * 0.8, 0.0, 1.0)
            else:
                diffs = np.abs(self.right_angles - angle_to_food)
                best_idx = int(np.argmin(diffs))
                right_ommatidia[best_idx] = max(right_ommatidia[best_idx], intensity)

                col = int(np.clip((angle_to_food / self.fov_rad) * 7.0, 0, 7))
                grid_right[3:6, col] = np.clip(grid_right[3:6, col] + intensity * 0.8, 0.0, 1.0)

        # 3. Looming Threat (if active)
        looming_left = 0.0
        looming_right = 0.0
        if threat_pos is not None:
            rel_t = threat_pos[:2] - fly_pos[:2]
            d_t = float(np.linalg.norm(rel_t))
            if 0 < d_t < 180.0:
                angle_t = (np.arctan2(rel_t[1], rel_t[0]) - fly_heading + np.pi) % (2 * np.pi) - np.pi
                speed_approach = max(0.0, self.prev_threat_distance - d_t)
                angular_size = (2.0 * threat_radius) / max(d_t, 8.0)
                loom_val = float(np.clip((speed_approach * 0.25 + angular_size * 0.75), 0.0, 1.0))

                if angle_t < 0:
                    looming_left = loom_val
                    col = int(np.clip((-angle_t / self.fov_rad) * 7.0, 0, 7))
                    grid_left[:, col] = np.clip(grid_left[:, col] + loom_val, 0.0, 1.0)
                else:
                    looming_right = loom_val
                    col = int(np.clip((angle_t / self.fov_rad) * 7.0, 0, 7))
                    grid_right[:, col] = np.clip(grid_right[:, col] + loom_val, 0.0, 1.0)

            self.prev_threat_distance = d_t
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
            left_eye_grid=grid_left.flatten().tolist(),
            right_eye_grid=grid_right.flatten().tolist(),
        )
