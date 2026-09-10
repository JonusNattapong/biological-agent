"""3D House Room Environment: A realistic interior living space with furniture, food, and lights."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from neurofly.environments.base import BaseEnvironment
from neurofly.sensory.vision import CompoundEyeVision, VisualStimulus
from neurofly.sensory.olfaction import AntennaeOlfaction, OlfactoryStimulus
from neurofly.sensory.mechanosensory import Mechanosensation, TactileStimulus
from neurofly.motor.decoder import MotorAction


@dataclass
class HouseFood:
    id: int
    pos: np.ndarray  # [x, y, z]
    name: str
    energy_value: float = 35.0
    consumed: bool = False
    respawn_timer: int = 0


class HouseRoomEnvironment(BaseEnvironment):
    """3D House Room environment matching third-person interior perspective.

    Features:
    - 3D Room: Width [-140, 140], Depth [-90, 90], Height [0, 100]
    - Furniture: Dining/work table on left, door with knob on right wall, window on back
    - Light Source: Window sunlight / ceiling lamp (phototaxis)
    - Food: Fruit/cake on table and sugar on floor
    - 3D Flight Kinematics: Fly banks, pitches, cruises, and lands
    """

    def __init__(
        self,
        include_predator: bool = True,
        max_steps: int = 2500,
        seed: Optional[int] = 42,
    ):
        self.max_steps = max_steps
        self.include_predator = include_predator
        if seed is not None:
            np.random.seed(seed)

        # Room bounds (cm / model units)
        self.room_x = [-130.0, 130.0]
        self.room_y = [-85.0, 85.0]
        self.room_z = [0.0, 95.0]

        # Table dimensions (left side matching user sketch)
        self.table_x = [-105.0, -45.0]
        self.table_y = [-28.0, 28.0]
        self.table_height = 30.0

        # Light source (window/lamp)
        self.light_pos = np.array([80.0, 75.0, 70.0], dtype=np.float32)

        # Sensory systems
        self.vision = CompoundEyeVision(num_ommatidia_per_eye=16)
        self.olfaction = AntennaeOlfaction(antenna_span=5.0)
        self.mechanosensation = Mechanosensation()

        # 3D Fly State
        self.fly_pos = np.zeros(3, dtype=np.float32)
        self.fly_heading: float = 0.0  # yaw angle in radians
        self.fly_pitch: float = 0.0    # pitch angle in radians
        self.fly_speed: float = 3.2
        self.fly_energy: float = 100.0
        self.is_airborne: bool = True

        # Threat / Swatter
        self.threat_pos = np.array([0.0, 0.0, 70.0], dtype=np.float32)
        self.threat_active: bool = False
        self.threat_timer: int = 0

        # Food items (fruit on table, sugar crumb on floor)
        self.foods: List[HouseFood] = []

        # Stats
        self.steps_survived: int = 0
        self.total_distance: float = 0.0
        self.food_eaten_count: int = 0
        self.collision_count: int = 0

        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset fly, food, and room state."""
        # Spawn fly near center of room facing the table/door
        self.fly_pos = np.array([-10.0, -20.0, 42.0], dtype=np.float32)
        self.fly_heading = float(np.random.uniform(-np.pi, np.pi))
        self.fly_pitch = 0.0
        self.fly_speed = 3.2
        self.fly_energy = 100.0
        self.is_airborne = True

        self.steps_survived = 0
        self.total_distance = 0.0
        self.food_eaten_count = 0
        self.collision_count = 0
        self.threat_active = False

        # Spawn food: Item 1 on Table (fruit), Item 2 on Floor (crumb)
        self.foods = [
            HouseFood(
                id=0,
                pos=np.array([-70.0, 0.0, self.table_height + 2.0], dtype=np.float32),
                name="Fruit Slice on Table",
            ),
            HouseFood(
                id=1,
                pos=np.array([45.0, -30.0, 2.0], dtype=np.float32),
                name="Sugar Crumb on Floor",
            ),
            HouseFood(
                id=2,
                pos=np.array([60.0, 50.0, 2.0], dtype=np.float32),
                name="Syrup Drop near Door",
            ),
        ]

        self.vision.reset()
        return self._get_observation()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance simulation in 3D house room."""
        self.steps_survived += 1

        if isinstance(action, MotorAction):
            fwd_vel = action.forward_velocity
            turn_vel = action.angular_velocity
        elif isinstance(action, (tuple, list, np.ndarray)):
            fwd_vel = float(action[0])
            turn_vel = float(action[1])
        else:
            fwd_vel = 3.2
            turn_vel = 0.0

        self.fly_speed = fwd_vel

        # 1. Update Yaw Heading (Steering)
        self.fly_heading = (self.fly_heading + turn_vel + np.pi) % (2 * np.pi) - np.pi

        # 2. 3D Kinematics: forward thrust + vertical oscillation/altitude flight
        # Altitude target: floats gently around 30 to 55 cm unless diving for food
        target_z = 45.0 + float(np.sin(self.steps_survived * 0.08) * 12.0)
        # If nearest food is lower, bias flight downwards
        active_foods = [f for f in self.foods if not f.consumed]
        if active_foods:
            closest_f = min(active_foods, key=lambda f: float(np.linalg.norm(f.pos[:2] - self.fly_pos[:2])))
            if np.linalg.norm(closest_f.pos[:2] - self.fly_pos[:2]) < 40.0:
                target_z = closest_f.pos[2] + 4.0

        z_err = target_z - self.fly_pos[2]
        vz = np.clip(z_err * 0.1, -1.8, 1.8)

        # Forward velocity vector in 3D
        vx = np.cos(self.fly_heading) * fwd_vel
        vy = np.sin(self.fly_heading) * fwd_vel

        step_delta = np.array([vx, vy, vz], dtype=np.float32)
        self.fly_pos += step_delta

        step_dist = float(np.linalg.norm(step_delta))
        self.total_distance += step_dist

        # Pitch based on vertical movement
        self.fly_pitch = float(np.clip(-vz * 0.15, -0.4, 0.4))

        # 3. Energy Expenditure
        self.fly_energy -= (0.025 + fwd_vel * 0.01)

        # 4. Room Wall Collisions & Soft Deflections
        collision_detected = False
        wall_margin = 4.0

        # X walls (left/right)
        if self.fly_pos[0] < self.room_x[0] + wall_margin:
            self.fly_pos[0] = self.room_x[0] + wall_margin
            self.fly_heading = np.pi - self.fly_heading
            collision_detected = True
        elif self.fly_pos[0] > self.room_x[1] - wall_margin:
            self.fly_pos[0] = self.room_x[1] - wall_margin
            self.fly_heading = np.pi - self.fly_heading
            collision_detected = True

        # Y walls (front/back)
        if self.fly_pos[1] < self.room_y[0] + wall_margin:
            self.fly_pos[1] = self.room_y[0] + wall_margin
            self.fly_heading = -self.fly_heading
            collision_detected = True
        elif self.fly_pos[1] > self.room_y[1] - wall_margin:
            self.fly_pos[1] = self.room_y[1] - wall_margin
            self.fly_heading = -self.fly_heading
            collision_detected = True

        # Floor and Ceiling
        if self.fly_pos[2] < self.room_z[0] + 1.0:
            self.fly_pos[2] = self.room_z[0] + 1.0
        elif self.fly_pos[2] > self.room_z[1] - 3.0:
            self.fly_pos[2] = self.room_z[1] - 3.0

        # Table Surface & Leg Collisions
        if (
            self.table_x[0] <= self.fly_pos[0] <= self.table_x[1]
            and self.table_y[0] <= self.fly_pos[1] <= self.table_y[1]
        ):
            if self.fly_pos[2] < self.table_height + 1.0:
                self.fly_pos[2] = self.table_height + 1.0
                collision_detected = True

        if collision_detected:
            self.collision_count += 1

        # 5. Food Consumption
        reward = 0.1
        for f in self.foods:
            if not f.consumed:
                d_3d = float(np.linalg.norm(self.fly_pos - f.pos))
                if d_3d < 9.0:
                    f.consumed = True
                    f.respawn_timer = 120
                    self.food_eaten_count += 1
                    self.fly_energy = min(100.0, self.fly_energy + f.energy_value)
                    reward += 25.0
            else:
                f.respawn_timer -= 1
                if f.respawn_timer <= 0:
                    f.consumed = False

        # 6. Looming Threat (Swatter / Cat Paw / Hazard)
        if self.threat_active:
            self.threat_timer -= 1
            if self.threat_timer <= 0:
                self.threat_active = False

        # 7. Done Check
        done = False
        death_reason = None
        if self.fly_energy <= 0.0:
            done = True
            death_reason = "energy_depleted"
        elif self.steps_survived >= self.max_steps:
            done = True
            death_reason = "time_limit_reached"

        obs = self._get_observation(collision_detected)

        info = {
            "steps_survived": self.steps_survived,
            "total_distance": round(self.total_distance, 1),
            "food_eaten": self.food_eaten_count,
            "collisions": self.collision_count,
            "energy": round(self.fly_energy, 1),
            "death_reason": death_reason,
        }

        return obs, reward, done, info

    def _get_observation(self, collision_detected: bool = False) -> Dict[str, Any]:
        """Compute biological visual, olfactory, and tactile stimuli in the 3D room."""
        active_food_pos_2d = [f.pos[:2] for f in self.foods if not f.consumed]
        threat_pos_2d = self.threat_pos[:2] if self.threat_active else None

        visual_stim = self.vision.perceive(
            fly_pos=self.fly_pos[:2],
            fly_heading=self.fly_heading,
            food_positions=active_food_pos_2d,
            threat_pos=threat_pos_2d,
            threat_radius=18.0,
        )

        olfactory_stim = self.olfaction.perceive(
            fly_pos=self.fly_pos[:2],
            fly_heading=self.fly_heading,
            food_positions=active_food_pos_2d,
        )

        tactile_stim = self.mechanosensation.perceive(
            wall_distance=20.0,
            collision_detected=collision_detected,
            predator_distance=999.0 if not self.threat_active else float(np.linalg.norm(self.fly_pos - self.threat_pos)),
        )

        return {
            "visual": visual_stim,
            "olfactory": olfactory_stim,
            "tactile": tactile_stim,
            "fly_pos": self.fly_pos.copy(),
            "fly_heading": self.fly_heading,
            "fly_pitch": self.fly_pitch,
            "energy": self.fly_energy,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return serializable 3D world state for Three.js Room View."""
        return {
            "fly": {
                "x": round(float(self.fly_pos[0]), 2),
                "y": round(float(self.fly_pos[1]), 2),
                "z": round(float(self.fly_pos[2]), 2),
                "heading": round(float(self.fly_heading), 3),
                "pitch": round(float(self.fly_pitch), 3),
                "energy": round(float(self.fly_energy), 1),
                "speed": round(float(self.fly_speed), 2),
            },
            "room": {
                "bounds_x": self.room_x,
                "bounds_y": self.room_y,
                "bounds_z": self.room_z,
                "table": {
                    "x": self.table_x,
                    "y": self.table_y,
                    "height": self.table_height,
                },
                "light_pos": self.light_pos.tolist(),
            },
            "foods": [
                {
                    "id": f.id,
                    "name": f.name,
                    "x": round(float(f.pos[0]), 2),
                    "y": round(float(f.pos[1]), 2),
                    "z": round(float(f.pos[2]), 2),
                    "consumed": f.consumed,
                }
                for f in self.foods
            ],
            "threat": {
                "active": self.threat_active,
                "x": round(float(self.threat_pos[0]), 2),
                "y": round(float(self.threat_pos[1]), 2),
                "z": round(float(self.threat_pos[2]), 2),
            },
            "step": self.steps_survived,
            "stats": {
                "food_eaten": self.food_eaten_count,
                "collisions": self.collision_count,
                "distance": round(self.total_distance, 1),
            },
        }
