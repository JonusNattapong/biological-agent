"""BrainSimulator: High-level controller uniting Connectome graph and LIF dynamics."""

from collections import deque
from typing import Dict, List, Optional
import numpy as np
import torch

from neurofly.brain.connectome import Connectome, Neuropil
from neurofly.brain.neuron import LIFNeurons


class BrainSimulator:
    """Manages stepping of the biological brain network, current injections, and recording."""

    def __init__(
        self,
        connectome: Connectome,
        dt: float = 1.0,  # 1 millisecond
        buffer_size: int = 100,  # Spike history steps for raster
    ):
        self.connectome = connectome
        self.dt = dt
        self.num_neurons = connectome.num_neurons
        self.device = connectome.device

        # Spiking neuron population
        self.neurons = LIFNeurons(
            num_neurons=self.num_neurons,
            dt=dt,
            device=self.device,
        )

        # External injected current buffer for the next step
        self.i_inj = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)

        # Activity recording
        self.buffer_size = buffer_size
        self.spike_history = deque(maxlen=buffer_size)
        self.current_step = 0

        # Moving average firing rates (Hz)
        self.smoothed_firing_rate = torch.zeros(self.num_neurons, dtype=torch.float32, device=self.device)
        self.rate_alpha = 0.95

    def reset(self):
        """Reset neurons and simulation step counter."""
        self.neurons.reset()
        self.i_inj.zero_()
        self.spike_history.clear()
        self.smoothed_firing_rate.zero_()
        self.current_step = 0

    def inject_current(self, indices: torch.Tensor, current: torch.Tensor):
        """Inject current into specific neurons for the upcoming tick."""
        if indices.numel() > 0:
            self.i_inj[indices] = current

    def add_current(self, indices: torch.Tensor, current: torch.Tensor):
        """Add current to specific neurons for the upcoming tick."""
        if indices.numel() > 0:
            self.i_inj[indices] += current

    def step(self) -> torch.Tensor:
        """Advance the biological network by dt (1 millisecond).

        1. Propagate previous spikes through connectome weight matrix: I_syn = W @ spikes
        2. Advance LIF membrane dynamics with I_syn + I_inj
        3. Clear injected currents
        4. Record spikes

        Returns:
            spikes: Boolean/float tensor of shape (num_neurons,)
        """
        # Synaptic currents from previous step's spikes
        if self.neurons.spikes.any():
            synaptic_input = self.connectome.propagate_spikes(self.neurons.spikes)
        else:
            synaptic_input = None

        # Step LIF dynamics
        spikes = self.neurons.step(i_inj=self.i_inj, synaptic_input=synaptic_input)

        # Clear external injections
        self.i_inj.zero_()

        # Update spike history and smoothed firing rate
        self.spike_history.append(spikes.clone())
        self.smoothed_firing_rate = (
            self.rate_alpha * self.smoothed_firing_rate
            + (1.0 - self.rate_alpha) * (spikes * (1000.0 / self.dt))
        )
        self.current_step += 1

        return spikes

    def get_active_spikes(self) -> np.ndarray:
        """Return 0-indexed indices of neurons that fired in the latest step."""
        spiked_indices = torch.nonzero(self.neurons.spikes).squeeze(1)
        return spiked_indices.cpu().numpy()

    def get_neuropil_activity(self) -> Dict[str, float]:
        """Compute mean firing rate (Hz) for each neuropil."""
        activity: Dict[str, float] = {}
        for np_type in Neuropil:
            indices = self.connectome.get_neuropil_indices(np_type)
            if indices.numel() > 0:
                mean_rate = self.smoothed_firing_rate[indices].mean().item()
                activity[np_type.value] = round(mean_rate, 2)
            else:
                activity[np_type.value] = 0.0
        return activity

    def get_spike_raster(self) -> np.ndarray:
        """Return a binary matrix (time_steps, num_neurons) of recent spike history."""
        if not self.spike_history:
            return np.zeros((0, self.num_neurons), dtype=np.uint8)
        stacked = torch.stack(list(self.spike_history), dim=0)
        return stacked.cpu().numpy().astype(np.uint8)
