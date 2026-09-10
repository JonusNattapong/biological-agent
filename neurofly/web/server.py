"""FastAPI and WebSocket streaming server for NeuroFly Web Playground."""

import asyncio
import json
import os
from collections import deque
from typing import Dict, Any, Optional, List
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from neurofly.environments.house.room import HouseRoomEnvironment
from neurofly.environments.survival.arena import SurvivalArena
from neurofly.controllers.malecns import MaleCNSController
from neurofly.controllers.heuristic import HeuristicController
from neurofly.controllers.tinynn import TinyNNController
from neurofly.controllers.random import RandomController
from neurofly.brain.connectome import Neuropil

app = FastAPI(title="NeuroFly Lab", version="0.1.0")

MAX_RENDER_NEURONS = 5000
MAX_RENDER_SPIKES = 2000


def _build_malecns_controller() -> MaleCNSController:
    """Build the configured biological controller.

    The web playground uses the lightweight structured baseline by default.
    Set ``NEUROFLY_MALECNS_WEIGHTS`` (plus optional annotation/NT paths) to
    activate the official MaleCNS v1.0 graph.
    """
    weights = os.getenv("NEUROFLY_MALECNS_WEIGHTS")
    if not weights:
        return MaleCNSController(scale=os.getenv("NEUROFLY_SYNTHETIC_SCALE", "standard"))

    return MaleCNSController.from_bulk_files(
        weights_path=weights,
        annotations_path=os.getenv("NEUROFLY_MALECNS_ANNOTATIONS") or None,
        neurotransmitters_path=os.getenv("NEUROFLY_MALECNS_NEUROTRANSMITTERS") or None,
        min_synapses=int(os.getenv("NEUROFLY_MALECNS_MIN_SYNAPSES", "1")),
        weight_transform=os.getenv("NEUROFLY_MALECNS_WEIGHT_TRANSFORM", "log1p"),
        weight_scale=float(os.getenv("NEUROFLY_MALECNS_WEIGHT_SCALE", "1.0")),
    )


STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)


class SimulationManager:
    """Manages continuous running simulation with 3D House Room and biological brain."""

    def __init__(self):
        self.environments = {
            "house": HouseRoomEnvironment(),
            "survival": SurvivalArena(),
        }
        self.active_env_key = "house"

        self.controllers = {
            "malecns": _build_malecns_controller(),
            "heuristic": HeuristicController(),
            "tinynn": TinyNNController(),
            "random": RandomController(),
        }
        self.active_controller_key = "malecns"
        self.is_running = True
        self.speed_multiplier = 1  # 1x, 5x, 10x, 20x
        self.step_delay = 0.038  # ~26 FPS base

        # Trajectory buffer (3D positions)
        self.trajectory = deque(maxlen=80)
        self._brain_render_lookup: Dict[int, int] = {}

        self.obs = self.active_env.reset()
        self.active_controller.reset()
        self._record_trajectory()

    @property
    def active_env(self):
        return self.environments[self.active_env_key]

    @property
    def active_controller(self):
        return self.controllers[self.active_controller_key]

    def set_environment(self, key: str):
        if key in self.environments:
            self.active_env_key = key
            self.reset_sim()

    def set_controller(self, key: str):
        if key in self.controllers:
            self.active_controller_key = key
            self.active_controller.reset()

    def reset_sim(self):
        self.obs = self.active_env.reset()
        self.active_controller.reset()
        self.trajectory.clear()
        self._record_trajectory()

    def _record_trajectory(self):
        pos = self.active_env.fly_pos
        if len(pos) >= 3:
            self.trajectory.append([round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(pos[2]), 2)])
        else:
            self.trajectory.append([round(float(pos[0]), 2), round(float(pos[1]), 2), 35.0])

    def get_static_brain_data(self) -> Dict[str, Any]:
        """Return a bounded 3D rendering view of the active connectome."""
        if not isinstance(self.active_controller, MaleCNSController):
            self._brain_render_lookup = {}
            return {"num_neurons": 0, "num_synapses": 0, "coordinates": [], "neuropils": [], "synaptic_tracts": []}

        conn = self.active_controller.connectome
        if conn.num_neurons <= MAX_RENDER_NEURONS:
            render_indices = np.arange(conn.num_neurons, dtype=np.int64)
        else:
            # Deterministic even sampling keeps browser payloads bounded while
            # preserving coverage across the indexed graph.
            render_indices = np.linspace(0, conn.num_neurons - 1, MAX_RENDER_NEURONS, dtype=np.int64)

        self._brain_render_lookup = {int(src): i for i, src in enumerate(render_indices)}
        coords = conn.coordinates[render_indices].astype(np.float32, copy=True)

        # MaleCNS somaLocation coordinates are voxel-space values (~10^4-10^5),
        # whereas the browser scene expects a compact centered cloud. This is
        # a display-only normalization; the Connectome retains source coords.
        if coords.size:
            nonzero = np.any(coords != 0.0, axis=1)
            if nonzero.any():
                center = np.median(coords[nonzero], axis=0)
                coords[nonzero] -= center
                radii = np.linalg.norm(coords[nonzero], axis=1)
                scale = float(np.percentile(radii, 95)) if radii.size else 1.0
                if scale > 0:
                    coords[nonzero] *= 180.0 / scale

        # Tracts are sampled only for small/medium graphs where local indices
        # map directly enough to provide useful context. Large graphs would
        # otherwise require scanning millions of edges for a decorative view.
        sampled_edges = [[], []]
        indices = conn.weight_matrix.indices().cpu().numpy()
        total_edges = indices.shape[1]
        if conn.num_neurons <= MAX_RENDER_NEURONS:
            sample_step = max(1, total_edges // 600)
            sampled_edges = indices[:, ::sample_step][:, :600].tolist()

        return {
            "num_neurons": conn.num_neurons,
            "rendered_neurons": int(len(render_indices)),
            "num_synapses": conn.num_synapses,
            "coordinates": coords.tolist(),
            "neuropils": [conn.neuron_neuropils[int(i)].value for i in render_indices],
            "synaptic_tracts": sampled_edges,
            "dataset_id": conn.metadata.get("dataset_id", "synthetic-structured"),
            "official_malecns": self.active_controller.is_official_malecns,
        }

    def _determine_behavior_state(self, obs: Dict[str, Any], action: Any) -> str:
        """Classify biological behavior mode."""
        v = obs["visual"]
        o = obs["olfactory"]
        t = obs["tactile"]

        if v.looming_threat_left > 0.15 or v.looming_threat_right > 0.15:
            return "ESCAPE_REFLEX"
        elif t.head_collision > 0.5 or t.body_contact > 0.5:
            return "WALL_AVOIDANCE"
        elif o.total_concentration > 0.25:
            return "NUTRIENT_FORAGING"
        elif o.total_concentration > 0.03:
            return "TROPOTAXIS_PURSUIT"
        else:
            return "INTERIOR_PATROL"

    def step(self) -> Dict[str, Any]:
        """Advance one tick and return combined frame payload."""
        # 1. Decide action
        action = self.active_controller.act(self.obs)

        # 2. Advance environment
        self.obs, reward, done, info = self.active_env.step(action)

        # 3. Append trajectory
        self._record_trajectory()

        # 4. Behavioral classification
        state_tag = self._determine_behavior_state(self.obs, action)

        if done:
            self.reset_sim()

        # 5. Compile telemetry
        env_state = self.active_env.get_state()
        env_state["trajectory"] = list(self.trajectory)
        env_state["behavior_state"] = state_tag
        env_state["env_type"] = self.active_env_key

        brain_data = {}
        if isinstance(self.active_controller, MaleCNSController):
            sim = self.active_controller.simulator
            raw_active_spikes = sim.get_active_spikes().tolist()
            if self._brain_render_lookup:
                active_spikes = [
                    self._brain_render_lookup[idx]
                    for idx in raw_active_spikes
                    if idx in self._brain_render_lookup
                ][:MAX_RENDER_SPIKES]
            else:
                active_spikes = raw_active_spikes[:MAX_RENDER_SPIKES]
            neuropils = sim.get_neuropil_activity()
            num_neurons = sim.num_neurons
            active_count = len(raw_active_spikes)
            sparsity = round((1.0 - (active_count / max(1, num_neurons))) * 100.0, 1)
            avg_v = round(float(sim.neurons.v.mean().item()), 1)
            fraction_firing = round((active_count / max(1, num_neurons)) * 100.0, 2)
            mean_rate = round(float(sim.smoothed_firing_rate.mean().item()), 1)

            brain_data = {
                "active_spikes": active_spikes,
                "neuropils": neuropils,
                "active_count": active_count,
                "sparsity_pct": sparsity,
                "mean_rate_hz": mean_rate,
                "avg_electrical_state_mv": avg_v,
                "fraction_firing_pct": fraction_firing,
                "dataset_id": self.active_controller.connectome.metadata.get("dataset_id", "synthetic-structured"),
                "official_malecns": self.active_controller.is_official_malecns,
            }

        # Action data
        action_data = {
            "forward_velocity": round(getattr(action, "forward_velocity", 0.0), 2),
            "angular_velocity": round(getattr(action, "angular_velocity", 0.0), 3),
            "raw_dn_activity": round(getattr(action, "raw_dn_activity", 0.0), 2),
        }

        return {
            "type": "tick",
            "arena": env_state,
            "brain": brain_data,
            "action": action_data,
            "controller": self.active_controller_key,
            "env_type": self.active_env_key,
            "info": info,
        }

    def inject_stimulus(self, cmd: Dict[str, Any]):
        """Inject user interaction stimulus into house room or brain."""
        action_type = cmd.get("type")
        if action_type == "drop_food_table":
            if isinstance(self.active_env, HouseRoomEnvironment):
                # Put fruit on the table
                self.active_env.foods[0].consumed = False
                self.active_env.foods[0].pos = np.array([-70.0, 0.0, self.active_env.table_height + 2.0], dtype=np.float32)
        elif action_type == "drop_food_floor":
            if isinstance(self.active_env, HouseRoomEnvironment):
                self.active_env.foods[1].consumed = False
                self.active_env.foods[1].pos = np.array([20.0, -10.0, 2.0], dtype=np.float32)
        elif action_type == "drop_food":
            x = float(cmd.get("x", 0.0))
            y = float(cmd.get("y", 0.0))
            if isinstance(self.active_env, HouseRoomEnvironment):
                self.active_env.foods[0].consumed = False
                self.active_env.foods[0].pos = np.array([x, y, 2.0], dtype=np.float32)
            else:
                self.active_env.foods[0].pos = np.array([x, y], dtype=np.float32)
        elif action_type == "spawn_threat":
            if isinstance(self.active_env, HouseRoomEnvironment):
                self.active_env.threat_active = True
                self.active_env.threat_timer = 40
                self.active_env.threat_pos = self.active_env.fly_pos + np.array([12.0, 10.0, 15.0], dtype=np.float32)
            elif hasattr(self.active_env, "predator") and self.active_env.predator:
                self.active_env.predator.pos = np.array(
                    [float(cmd.get("x", 0.0)), float(cmd.get("y", 80.0))],
                    dtype=np.float32,
                )
        elif action_type == "flash_light":
            if isinstance(self.active_controller, MaleCNSController):
                sim = self.active_controller.simulator
                conn = self.active_controller.connectome
                side = cmd.get("side", "left")
                np_target = (
                    Neuropil.OPTIC_LOBE_LEFT if side == "left" else Neuropil.OPTIC_LOBE_RIGHT
                )
                idxs = conn.get_neuropil_indices(np_target)
                import torch
                sim.inject_current(idxs, torch.full_like(idxs, 8.0, dtype=torch.float32))
        elif action_type == "zap_cx":
            if isinstance(self.active_controller, MaleCNSController):
                sim = self.active_controller.simulator
                conn = self.active_controller.connectome
                idxs = conn.get_neuropil_indices(Neuropil.CENTRAL_COMPLEX)
                import torch
                sim.inject_current(idxs, torch.full_like(idxs, 12.0, dtype=torch.float32))


manager = SimulationManager()


@app.get("/")
async def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "environment": manager.active_env_key,
        "controller": manager.active_controller_key,
        "step": manager.active_env.steps_survived,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    init_data = {
        "type": "init",
        "brain_topology": manager.get_static_brain_data(),
        "controllers": ["malecns", "heuristic", "tinynn", "random"],
        "active_controller": manager.active_controller_key,
        "active_env": manager.active_env_key,
    }
    await websocket.send_text(json.dumps(init_data))

    async def incoming_listener():
        try:
            while True:
                data = await websocket.receive_text()
                cmd = json.loads(data)
                action = cmd.get("action")
                if action == "set_controller":
                    manager.set_controller(cmd.get("controller"))
                    await websocket.send_text(
                        json.dumps({
                            "type": "init",
                            "brain_topology": manager.get_static_brain_data(),
                            "active_controller": manager.active_controller_key,
                            "active_env": manager.active_env_key,
                        })
                    )
                elif action == "set_env":
                    manager.set_environment(cmd.get("env"))
                elif action == "reset":
                    manager.reset_sim()
                elif action == "pause":
                    manager.is_running = not manager.is_running
                elif action == "set_speed":
                    manager.speed_multiplier = int(cmd.get("speed", 1))
                elif action == "stimulus":
                    manager.inject_stimulus(cmd)
        except (WebSocketDisconnect, Exception):
            pass

    async def broadcast_loop():
        try:
            while True:
                if manager.is_running:
                    steps_to_run = manager.speed_multiplier
                    frame_data = None
                    for _ in range(steps_to_run):
                        frame_data = manager.step()

                    if frame_data:
                        await websocket.send_text(json.dumps(frame_data))

                delay = max(0.005, manager.step_delay / max(1, manager.speed_multiplier))
                await asyncio.sleep(delay)
        except (WebSocketDisconnect, Exception):
            pass

    listener_task = asyncio.create_task(incoming_listener())
    broadcaster_task = asyncio.create_task(broadcast_loop())

    done, pending = await asyncio.wait(
        [listener_task, broadcaster_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
