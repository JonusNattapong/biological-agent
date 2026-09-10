"""Vectorized Leaky Integrate-and-Fire (LIF) spiking neuron model with PyTorch sparse tensors."""

from typing import Optional
import torch


class LIFNeurons:
    """Vectorized Leaky Integrate-and-Fire (LIF) neuron population with biological heterogeneity.

    Dynamics:
        V[t+1] = V_rest + (V[t] - V_rest) * decay_v + (I_total * dt / C_m)
        where I_total = I_syn + I_inj + noise
    """

    def __init__(
        self,
        num_neurons: int,
        dt: float = 1.0,  # millisecond
        tau_m: float = 16.0,  # membrane time constant (ms)
        tau_syn: float = 4.0,  # synaptic decay (ms)
        c_m: float = 1.0,  # membrane capacitance (uF / cm^2)
        v_rest: float = -70.0,  # mV
        v_reset: float = -72.0,  # mV
        v_thresh: Optional[float] = None,
        v_thresh_base: float = -50.0,  # mV
        refractory_steps: Optional[int] = None,
        noise_std: float = 0.8,  # background synaptic noise
        device: Optional[str] = None,
    ):
        self.num_neurons = num_neurons
        self.dt = dt
        self.tau_m = tau_m
        self.tau_syn = tau_syn
        self.c_m = c_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.noise_std = noise_std
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Decay constants
        self.decay_v = float(torch.exp(torch.tensor(-dt / tau_m)).item())
        self.decay_syn = float(torch.exp(torch.tensor(-dt / tau_syn)).item())

        # Base threshold
        thresh_center = v_thresh if v_thresh is not None else v_thresh_base

        # Heterogeneous thresholds (center +- 1.5 mV) to eliminate artificial lockstep synchrony
        torch.manual_seed(42)
        if v_thresh is not None:
            self.v_thresh = torch.full((num_neurons,), thresh_center, device=self.device)
        else:
            self.v_thresh = thresh_center + torch.randn(num_neurons, device=self.device) * 1.2

        # Refractory period durations
        if refractory_steps is not None:
            self.refractory_durations = torch.full((num_neurons,), refractory_steps, dtype=torch.int32, device=self.device)
        else:
            self.refractory_durations = torch.randint(2, 6, (num_neurons,), dtype=torch.int32, device=self.device)

        # State vectors
        self.v = torch.full((num_neurons,), v_rest, dtype=torch.float32, device=self.device)
        self.i_syn = torch.zeros(num_neurons, dtype=torch.float32, device=self.device)
        self.refractory_timer = torch.zeros(num_neurons, dtype=torch.int32, device=self.device)
        self.spikes = torch.zeros(num_neurons, dtype=torch.float32, device=self.device)

    def reset(self):
        """Reset all neuron states to resting potential."""
        self.v.fill_(self.v_rest)
        self.i_syn.zero_()
        self.refractory_timer.zero_()
        self.spikes.zero_()

    def step(
        self,
        i_inj: Optional[torch.Tensor] = None,
        synaptic_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Advance the neuron states by dt.

        Args:
            i_inj: Injected external current tensor (num_neurons,)
            synaptic_input: Incoming synaptic currents from connectivity graph (num_neurons,)

        Returns:
            spikes: Boolean/float tensor indicating which neurons fired (num_neurons,)
        """
        # 1. Update synaptic currents with exponential decay
        self.i_syn.mul_(self.decay_syn)
        if synaptic_input is not None:
            self.i_syn.add_(synaptic_input)

        # 2. Total input current: synaptic + external injection + background noise
        i_total = self.i_syn.clone()
        if i_inj is not None:
            i_total.add_(i_inj)
        if self.noise_std > 0:
            noise = torch.randn(self.num_neurons, device=self.device) * self.noise_std
            i_total.add_(noise)

        # 3. Handle refractory state
        is_refractory = self.refractory_timer > 0
        self.refractory_timer[is_refractory] -= 1

        # 4. Membrane potential update for non-refractory neurons (Standard physical LIF)
        non_refractory = ~is_refractory
        self.v[non_refractory] = (
            self.v_rest
            + (self.v[non_refractory] - self.v_rest) * self.decay_v
            + i_total[non_refractory] * (self.dt / self.c_m)
        )
        self.v[non_refractory] = torch.clamp(self.v[non_refractory], min=-85.0, max=40.0)

        # 5. Check threshold and emit spikes
        spiked = (self.v >= self.v_thresh) & non_refractory
        self.spikes.zero_()
        self.spikes[spiked] = 1.0

        # 6. Reset membrane potential and start heterogeneous refractory timer
        self.v[spiked] = self.v_reset
        self.refractory_timer[spiked] = self.refractory_durations[spiked]

        return self.spikes
