"""MaleCNS Connectome controller: Biological brain runtime driving agent behavior."""

from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch

from neurofly.controllers.base import BaseController
from neurofly.brain.connectome import Connectome
from neurofly.brain.loader import generate_drosophila_connectome
from neurofly.brain.malecns import MALECNS_DATASET_ID, load_malecns_v1_bulk
from neurofly.brain.simulator import BrainSimulator
from neurofly.sensory.encoder import SensoryEncoder
from neurofly.motor.decoder import MotorDecoder, MotorAction


class MaleCNSController(BaseController):
    """Biological Drosophila CNS controller using connectome graph and LIF dynamics."""

    def __init__(
        self,
        connectome: Optional[Connectome] = None,
        scale: str = "standard",
        ticks_per_step: int = 4,
        device: Optional[str] = None,
    ):
        self.scale = scale
        self.ticks_per_step = ticks_per_step

        # Initialize connectome and simulator
        self.connectome = connectome or generate_drosophila_connectome(scale=scale, device=device)
        self.simulator = BrainSimulator(self.connectome)

        # Sensory and motor interfaces
        self.encoder = SensoryEncoder(self.connectome)
        self.decoder = MotorDecoder(self.connectome)

    @classmethod
    def from_bulk_files(
        cls,
        weights_path: str | Path,
        annotations_path: Optional[str | Path] = None,
        neurotransmitters_path: Optional[str | Path] = None,
        *,
        min_synapses: int = 1,
        weight_transform: str = "log1p",
        weight_scale: float = 1.0,
        ticks_per_step: int = 4,
        device: Optional[str] = None,
    ) -> "MaleCNSController":
        """Build a controller backed by the official MaleCNS v1.0 bulk graph."""
        connectome = load_malecns_v1_bulk(
            weights_path=weights_path,
            annotations_path=annotations_path,
            neurotransmitters_path=neurotransmitters_path,
            min_synapses=min_synapses,
            weight_transform=weight_transform,
            weight_scale=weight_scale,
            device=device,
        )
        return cls(
            connectome=connectome,
            scale="v1.0",
            ticks_per_step=ticks_per_step,
            device=device,
        )

    @property
    def is_official_malecns(self) -> bool:
        return self.connectome.metadata.get("dataset_id") == MALECNS_DATASET_ID

    @property
    def name(self) -> str:
        if self.is_official_malecns:
            return "MaleCNS v1.0 Connectome"
        return f"Structured Drosophila baseline ({self.scale})"

    def reset(self):
        """Reset internal biological brain states."""
        self.simulator.reset()
        self.decoder.reset()

    def act(self, obs: Dict[str, Any]) -> MotorAction:
        """Process biological sensory stimuli through MaleCNS connectome to produce motor action."""
        visual = obs["visual"]
        olfactory = obs["olfactory"]
        tactile = obs["tactile"]

        # 1. Encode sensory signals into current injection vectors
        target_indices, current_values = self.encoder.encode(visual, olfactory, tactile)

        # 2. Step biological network across multiple millisecond ticks with sustained sensory transduction
        accumulated_spikes = torch.zeros(self.connectome.num_neurons, device=self.connectome.device)
        for _ in range(self.ticks_per_step):
            self.simulator.inject_current(target_indices, current_values)
            step_spikes = self.simulator.step()
            accumulated_spikes += step_spikes

        # 3. Decode descending neuron spiking into physical motor action
        action = self.decoder.decode(accumulated_spikes / max(1, self.ticks_per_step))
        return action

    def get_brain_telemetry(self) -> Dict[str, Any]:
        """Return real-time neural firing rates, active spikes, and raster for UI streaming."""
        active_spikes = self.simulator.get_active_spikes()
        neuropil_activity = self.simulator.get_neuropil_activity()

        return {
            "num_neurons": self.connectome.num_neurons,
            "num_synapses": self.connectome.num_synapses,
            "active_spikes_count": len(active_spikes),
            "active_spike_indices": active_spikes.tolist()[:100],  # cap for bandwidth
            "neuropil_rates_hz": neuropil_activity,
            "dataset_id": self.connectome.metadata.get("dataset_id", "synthetic-structured"),
            "official_malecns": self.is_official_malecns,
        }
