"""Survival Arena: Continuous 2D/3D physics arena with realistic collisions, food, and predators."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from neurofly.environments.base import BaseEnvironment
from neurofly.environments.survival.entities import FoodItem, Predator, Obstacle
from neurofly.sensory.vision import CompoundEyeVision, VisualStimulus
from neurofly.sensory.olfaction import AntennaeOlfaction, OlfactoryStimulus
from neurofly.sensory.mechanosensory import Mechanosensation, TactileStimulus
from neurofly.motor.decoder import MotorAction


class SurvivalArena(BaseEnvironment):
    """Survival Arena simulation with smooth elastic collision physics and realistic kinematics."""

    def __init__(
        self,
        arena_radius: float = 220.0,
        num_food: int = 6,
        include_predator: bool = True,
        max_steps: int = 2000,
        seed: Optional[int] = 42,
    ):
        self.arena_radius = arena_radius
        self.num_food = num_food
        self.include_predator = include_predator
        self.max_steps = max_steps
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)

        # Sensory models
        self.vision = CompoundEyeVision(num_ommatidia_per_eye=16)
        self.olfaction = AntennaeOlfaction(antenna_span=5.0)
        self.mechanosensation = Mechanosensation()

        # Kinematic state
        self.fly_pos = np.zeros(2, dtype=np.float32)
        self.fly_heading: float = 0.0
        self.fly_speed: float = 3.0
        self.fly_radius: float = 4.0
        self.fly_energy: float = 100.0

        # Stats
        self.steps_survived: int = 0
        self.total_distance: float = 0.0
        self.food_eaten_count: int = 0
        self.collision_count: int = 0

        # Entities
        self.foods: List[FoodItem] = []
        self.predator: Optional[Predator] = None
        self.obstacles: List[Obstacle] = []

        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset arena and entities."""
        self.fly_pos = np.array([0.0, -80.0], dtype=np.float32)
        self.fly_heading = float(np.random.uniform(-np.pi, np.pi))
        self.fly_speed = 3.0
        self.fly_energy = 100.0

        self.steps_survived = 0
        self.total_distance = 0.0
        self.food_eaten_count = 0
        self.collision_count = 0

        # Circular obstacles (pillars)
        self.obstacles = [
            Obstacle(pos=np.array([75.0, 30.0], dtype=np.float32), radius=18.0),
            Obstacle(pos=np.array([-75.0, 30.0], dtype=np.float32), radius=18.0),
        ]

        # Food sources
        self.foods = []
        for i in range(self.num_food):
            r = np.random.uniform(40.0, self.arena_radius - 30.0)
            theta = np.random.uniform(0, 2 * np.pi)
            pos = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
            self.foods.append(FoodItem(id=i, pos=pos))

        # Predator
        if self.include_predator:
            pred_pos = np.array([0.0, 110.0], dtype=np.float32)
            self.predator = Predator(pos=pred_pos, radius=14.0, speed=1.9)
        else:
            self.predator = None

        self.vision.reset()

        return self._get_observation()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance simulation by one tick with smooth physics and collision reflections."""
        self.steps_survived += 1

        if isinstance(action, MotorAction):
            fwd_vel = action.forward_velocity
            turn_vel = action.angular_velocity
        elif isinstance(action, (tuple, list, np.ndarray)):
            fwd_vel = float(action[0])
            turn_vel = float(action[1])
        else:
            fwd_vel = 3.0
            turn_vel = 0.0

        self.fly_speed = fwd_vel

        # 1. Update heading with angular velocity
        self.fly_heading = (self.fly_heading + turn_vel + np.pi) % (2 * np.pi) - np.pi
        heading_dir = np.array([np.cos(self.fly_heading), np.sin(self.fly_heading)], dtype=np.float32)

        # 2. Advance position
        step_delta = heading_dir * fwd_vel
        self.fly_pos += step_delta

        step_dist = float(np.linalg.norm(step_delta))
        self.total_distance += step_dist

        # 3. Metabolic energy expenditure
        energy_burn = 0.03 + (fwd_vel * 0.012) + (abs(turn_vel) * 0.008)
        self.fly_energy -= energy_burn

        # 4. Arena Boundary Collision with Smooth Elastic Reflection
        dist_from_center = float(np.linalg.norm(self.fly_pos))
        wall_dist = self.arena_radius - dist_from_center
        collision_detected = False

        if dist_from_center > self.arena_radius - self.fly_radius:
            collision_detected = True
            self.collision_count += 1

            # Clamp inside boundary
            radial_dir = self.fly_pos / dist_from_center
            self.fly_pos = radial_dir * (self.arena_radius - self.fly_radius)

            # Inward normal vector: -radial_dir
            inward_normal = -radial_dir
            # Current velocity vector
            vel_vec = heading_dir
            # Reflect velocity off wall: v' = v - 2(v . n)n
            dot = float(np.dot(vel_vec, inward_normal))
            if dot < 0:  # Heading towards wall
                reflected = vel_vec - 1.8 * dot * inward_normal
                new_heading = float(np.arctan2(reflected[1], reflected[0]))
                # Smoothly blend heading to reflected angle
                self.fly_heading = new_heading

        # 5. Obstacle Collisions with Smooth Surface Gliding
        for obs in self.obstacles:
            rel = self.fly_pos - obs.pos
            dist = float(np.linalg.norm(rel))
            min_dist = obs.radius + self.fly_radius

            if dist < min_dist:
                collision_detected = True
                self.collision_count += 1

                # Surface normal pointing outward from obstacle
                outward_normal = rel / max(dist, 1e-4)
                self.fly_pos = obs.pos + outward_normal * min_dist

                dot = float(np.dot(heading_dir, -outward_normal))
                if dot > 0:  # Moving into obstacle
                    reflected = heading_dir + 1.8 * dot * outward_normal
                    self.fly_heading = float(np.arctan2(reflected[1], reflected[0]))

        # 6. Food Interaction & Odor Consumption
        reward = 0.1
        for f in self.foods:
            if not f.consumed:
                f_dist = float(np.linalg.norm(self.fly_pos - f.pos))
                if f_dist < self.fly_radius + f.radius:
                    f.consumed = True
                    f.respawn_timer = 90
                    self.food_eaten_count += 1
                    self.fly_energy = min(100.0, self.fly_energy + f.energy_value)
                    reward += 20.0
            else:
                f.respawn_timer -= 1
                if f.respawn_timer <= 0:
                    f.consumed = False
                    r = np.random.uniform(40.0, self.arena_radius - 30.0)
                    theta = np.random.uniform(0, 2 * np.pi)
                    f.pos = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)

        # 7. Predator Tracking & Capture
        predator_hit = False
        pred_dist = 9999.0
        if self.predator is not None:
            self.predator.step(self.fly_pos, self.arena_radius)
            pred_dist = float(np.linalg.norm(self.fly_pos - self.predator.pos))
            if pred_dist < self.fly_radius + self.predator.radius:
                predator_hit = True

        # 8. Terminal Conditions
        done = False
        death_reason = None
        if self.fly_energy <= 0.0:
            done = True
            death_reason = "starvation"
            reward -= 20.0
        elif predator_hit:
            done = True
            death_reason = "eaten_by_predator"
            reward -= 30.0
        elif self.steps_survived >= self.max_steps:
            done = True
            death_reason = "time_limit_reached"

        obs = self._get_observation(collision_detected, wall_dist, pred_dist)

        info = {
            "steps_survived": self.steps_survived,
            "total_distance": round(self.total_distance, 1),
            "food_eaten": self.food_eaten_count,
            "collisions": self.collision_count,
            "energy": round(self.fly_energy, 1),
            "death_reason": death_reason,
        }

        return obs, reward, done, info

    def _get_observation(
        self,
        collision_detected: bool = False,
        wall_dist: float = 100.0,
        pred_dist: float = 9999.0,
    ) -> Dict[str, Any]:
        """Compute the biological sensory stimuli."""
        active_food_positions = [f.pos for f in self.foods if not f.consumed]
        threat_pos = self.predator.pos if self.predator else None
        threat_radius = self.predator.radius if self.predator else 0.0

        visual_stim = self.vision.perceive(
            fly_pos=self.fly_pos,
            fly_heading=self.fly_heading,
            food_positions=active_food_positions,
            threat_pos=threat_pos,
            threat_radius=threat_radius,
        )

        olfactory_stim = self.olfaction.perceive(
            fly_pos=self.fly_pos,
            fly_heading=self.fly_heading,
            food_positions=active_food_positions,
        )

        tactile_stim = self.mechanosensation.perceive(
            wall_distance=wall_dist,
            collision_detected=collision_detected,
            predator_distance=pred_dist,
        )

        return {
            "visual": visual_stim,
            "olfactory": olfactory_stim,
            "tactile": tactile_stim,
            "fly_pos": self.fly_pos.copy(),
            "fly_heading": self.fly_heading,
            "energy": self.fly_energy,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return serializable world state for WebSocket stream and web frontend."""
        return {
            "fly": {
                "x": round(float(self.fly_pos[0]), 2),
                "y": round(float(self.fly_pos[1]), 2),
                "heading": round(float(self.fly_heading), 3),
                "energy": round(float(self.fly_energy), 1),
                "speed": round(float(self.fly_speed), 2),
                "radius": self.fly_radius,
            },
            "foods": [
                {
                    "id": f.id,
                    "x": round(float(f.pos[0]), 2),
                    "y": round(float(f.pos[1]), 2),
                    "consumed": f.consumed,
                }
                for f in self.foods
            ],
            "predator": {
                "x": round(float(self.predator.pos[0]), 2),
                "y": round(float(self.predator.pos[1]), 2),
                "radius": self.predator.radius,
            }
            if self.predator
            else None,
            "obstacles": [
                {
                    "x": round(float(o.pos[0]), 2),
                    "y": round(float(o.pos[1]), 2),
                    "radius": o.radius,
                }
                for o in self.obstacles
            ],
            "arena_radius": self.arena_radius,
            "step": self.steps_survived,
            "stats": {
                "food_eaten": self.food_eaten_count,
                "collisions": self.collision_count,
                "distance": round(self.total_distance, 1),
            },
        }
