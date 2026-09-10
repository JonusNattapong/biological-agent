"""3D House Room Environment: Natural biological flight and Inside-Fly compound eye perception."""

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
    """3D House Room environment designed for peaceful biological observation.

    Features:
    - Natural aerodynamic flight: smooth cruising, banking, and gentle wall avoidance
    - "Inside Fly 05" perception: 8x8 compound eye retinas for Left and Right eyes
    - Realistic room furniture: Table on the left, Door on the right, Window sunlight
    - No stress/death: Peaceful observation of biological connectome behavior
    """

    def __init__(
        self,
        include_predator: bool = False,  # Off by default for pure observation
        max_steps: int = 10000,
        seed: Optional[int] = 42,
    ):
        self.max_steps = max_steps
        self.include_predator = include_predator
        if seed is not None:
            np.random.seed(seed)

        # Room bounds (cm)
        self.room_x = [-120.0, 120.0]
        self.room_y = [-75.0, 75.0]
        self.room_z = [0.0, 90.0]

        # Table dimensions (left side)
        self.table_x = [-105.0, -45.0]
        self.table_y = [-28.0, 28.0]
        self.table_height = 30.0

        # Window light source
        self.light_pos = np.array([75.0, 70.0, 65.0], dtype=np.float32)

        # Sensory systems
        self.vision = CompoundEyeVision(num_ommatidia_per_eye=16)
        self.olfaction = AntennaeOlfaction(antenna_span=5.0)
        self.mechanosensation = Mechanosensation()

        # 3D Fly State
        self.fly_pos = np.zeros(3, dtype=np.float32)
        self.fly_heading: float = 0.0
        self.fly_pitch: float = 0.0
        self.fly_speed: float = 3.5
        self.fly_energy: float = 100.0

        # Threat (inactive in observation mode)
        self.threat_pos = np.array([0.0, 0.0, 70.0], dtype=np.float32)
        self.threat_active: bool = False
        self.threat_timer: int = 0

        # Foods (Fruit on table, sugar on floor)
        self.foods: List[HouseFood] = []

        # Stats
        self.steps_survived: int = 0
        self.total_distance: float = 0.0
        self.food_eaten_count: int = 0

        # Cached stimuli
        self.last_visual: Optional[VisualStimulus] = None
        self.last_olfactory: Optional[OlfactoryStimulus] = None

        self.reset()

    def reset(self) -> Dict[str, Any]:
        """Reset fly to center of room with natural initial altitude."""
        self.fly_pos = np.array([0.0, -10.0, 38.0], dtype=np.float32)
        self.fly_heading = float(np.random.uniform(-np.pi, np.pi))
        self.fly_pitch = 0.0
        self.fly_speed = 3.5
        self.fly_energy = 100.0

        self.steps_survived = 0
        self.total_distance = 0.0
        self.food_eaten_count = 0
        self.threat_active = False

        self.foods = [
            HouseFood(
                id=0,
                pos=np.array([-75.0, 0.0, self.table_height + 2.0], dtype=np.float32),
                name="Fruit Slice on Table",
            ),
            HouseFood(
                id=1,
                pos=np.array([35.0, -25.0, 2.0], dtype=np.float32),
                name="Sugar Crumb on Floor",
            ),
            HouseFood(
                id=2,
                pos=np.array([55.0, 45.0, 2.0], dtype=np.float32),
                name="Syrup near Door",
            ),
        ]

        self.vision.reset()
        return self._get_observation()

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Advance simulation with smooth biological flight and boundary avoidance."""
        self.steps_survived += 1

        if isinstance(action, MotorAction):
            fwd_vel = action.forward_velocity
            turn_vel = action.angular_velocity
        elif isinstance(action, (tuple, list, np.ndarray)):
            fwd_vel = float(action[0])
            turn_vel = float(action[1])
        else:
            fwd_vel = 3.5
            turn_vel = 0.0

        # 1. Aerodynamic Boundary Cushion (Smooth wall avoidance)
        margin = 22.0
        steer_bias = 0.0

        if self.fly_pos[0] < self.room_x[0] + margin:
            steer_bias += 0.08  # turn right away from left wall
        elif self.fly_pos[0] > self.room_x[1] - margin:
            steer_bias -= 0.08  # turn left away from right wall

        if self.fly_pos[1] < self.room_y[0] + margin:
            steer_bias += 0.08  # turn away from front wall
        elif self.fly_pos[1] > self.room_y[1] - margin:
            steer_bias -= 0.08  # turn away from back wall

        # 2. Update Yaw Heading
        total_turn = turn_vel + steer_bias
        self.fly_heading = (self.fly_heading + total_turn + np.pi) % (2 * np.pi) - np.pi

        # 3. 3D Flight Kinematics
        # Natural vertical undulating flight between 28 and 55 cm
        target_z = 38.0 + float(np.sin(self.steps_survived * 0.06) * 12.0)
        # If hovering over table, maintain clearance above table
        if (
            self.table_x[0] - 10 <= self.fly_pos[0] <= self.table_x[1] + 10
            and self.table_y[0] - 10 <= self.fly_pos[1] <= self.table_y[1] + 10
        ):
            target_z = max(target_z, self.table_height + 8.0)

        z_err = target_z - self.fly_pos[2]
        vz = float(np.clip(z_err * 0.08, -1.5, 1.5))

        vx = np.cos(self.fly_heading) * fwd_vel
        vy = np.sin(self.fly_heading) * fwd_vel

        step_delta = np.array([vx, vy, vz], dtype=np.float32)
        self.fly_pos += step_delta

        step_dist = float(np.linalg.norm(step_delta))
        self.total_distance += step_dist
        self.fly_speed = float(np.linalg.norm([vx, vy]))
        self.fly_pitch = float(np.clip(-vz * 0.12, -0.35, 0.35))

        # Clamp inside room bounds gently
        self.fly_pos[0] = np.clip(self.fly_pos[0], self.room_x[0] + 5.0, self.room_x[1] - 5.0)
        self.fly_pos[1] = np.clip(self.fly_pos[1], self.room_y[0] + 5.0, self.room_y[1] - 5.0)
        self.fly_pos[2] = np.clip(self.fly_pos[2], self.room_z[0] + 4.0, self.room_z[1] - 5.0)

        # 4. Food Interaction (Nibbling / Visiting food)
        for f in self.foods:
            if not f.consumed:
                d_3d = float(np.linalg.norm(self.fly_pos - f.pos))
                if d_3d < 12.0:
                    f.consumed = True
                    f.respawn_timer = 160
                    self.food_eaten_count += 1
            else:
                f.respawn_timer -= 1
                if f.respawn_timer <= 0:
                    f.consumed = False

        obs = self._get_observation()

        info = {
            "steps_survived": self.steps_survived,
            "total_distance": round(self.total_distance, 1),
            "food_eaten": self.food_eaten_count,
            "collisions": 0,
            "energy": 100.0,
            "death_reason": None,
        }

        return obs, 0.1, False, info

    def _get_observation(self) -> Dict[str, Any]:
        """Compute biological compound eye vision, olfaction, and tactile signals."""
        active_food_pos = [f.pos[:2] for f in self.foods if not f.consumed]
        threat_pos_2d = self.threat_pos[:2] if self.threat_active else None

        visual_stim = self.vision.perceive(
            fly_pos=self.fly_pos[:2],
            fly_heading=self.fly_heading,
            food_positions=active_food_pos,
            threat_pos=threat_pos_2d,
            threat_radius=18.0,
            light_pos=self.light_pos,
        )

        olfactory_stim = self.olfaction.perceive(
            fly_pos=self.fly_pos[:2],
            fly_heading=self.fly_heading,
            food_positions=active_food_pos,
        )

        tactile_stim = self.mechanosensation.perceive(
            wall_distance=25.0,
            collision_detected=False,
            predator_distance=999.0,
        )

        self.last_visual = visual_stim
        self.last_olfactory = olfactory_stim

        return {
            "visual": visual_stim,
            "olfactory": olfactory_stim,
            "tactile": tactile_stim,
            "fly_pos": self.fly_pos.copy(),
            "fly_heading": self.fly_heading,
            "fly_pitch": self.fly_pitch,
            "energy": 100.0,
        }

    def get_state(self) -> Dict[str, Any]:
        """Return full 3D state including Inside-Fly retinal pixel grids and olfaction levels."""
        vis = self.last_visual
        olf = self.last_olfactory

        return {
            "fly": {
                "x": round(float(self.fly_pos[0]), 2),
                "y": round(float(self.fly_pos[1]), 2),
                "z": round(float(self.fly_pos[2]), 2),
                "heading": round(float(self.fly_heading), 3),
                "pitch": round(float(self.fly_pitch), 3),
                "energy": 100.0,
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
            "inside_fly": {
                "left_eye_pixels": vis.left_eye_grid if vis else [0.1] * 64,
                "right_eye_pixels": vis.right_eye_grid if vis else [0.1] * 64,
                "smell_left": round(float(olf.left_concentration), 3) if olf else 0.0,
                "smell_right": round(float(olf.right_concentration), 3) if olf else 0.0,
                "odor_gradient": round(float(olf.gradient), 3) if olf else 0.0,
            },
            "step": self.steps_survived,
            "stats": {
                "food_eaten": self.food_eaten_count,
                "distance": round(self.total_distance, 1),
            },
        }
