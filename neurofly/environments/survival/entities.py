"""Entities inhabiting the Survival Arena (Food, Predator, Obstacles)."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class FoodItem:
    """A nutrient source emitting visual and chemical odor cues."""
    id: int
    pos: np.ndarray  # [x, y]
    radius: float = 6.0
    energy_value: float = 35.0
    consumed: bool = False
    respawn_timer: int = 0


class Predator:
    """A looming threat that stalks the fly or patrols the arena."""

    def __init__(
        self,
        pos: np.ndarray,
        radius: float = 14.0,
        speed: float = 2.2,
        hunt_radius: float = 180.0,
    ):
        self.pos = pos.astype(np.float32)
        self.radius = radius
        self.speed = speed
        self.hunt_radius = hunt_radius
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.active = True

    def step(self, fly_pos: np.ndarray, arena_radius: float):
        """Move predator toward fly if in range, or patrol."""
        rel = fly_pos - self.pos
        dist = float(np.linalg.norm(rel))

        if dist < self.hunt_radius and dist > 1.0:
            # Hunt fly directly
            direction = rel / dist
            self.velocity = direction * self.speed
        else:
            # Random patrol or center-directed drift
            if np.random.rand() < 0.08:
                rand_angle = np.random.uniform(0, 2 * np.pi)
                self.velocity = np.array([np.cos(rand_angle), np.sin(rand_angle)], dtype=np.float32) * (self.speed * 0.6)

        self.pos += self.velocity

        # Constrain within arena bounds
        p_dist = np.linalg.norm(self.pos)
        if p_dist > arena_radius - self.radius:
            self.pos = (self.pos / p_dist) * (arena_radius - self.radius)
            self.velocity *= -0.5


@dataclass
class Obstacle:
    """A circular or bounding obstacle in the arena."""
    pos: np.ndarray
    radius: float
