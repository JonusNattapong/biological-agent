"""Unit tests for LIF neuron dynamics and Connectome sparse propagation."""

import pytest
import torch
import numpy as np

from neurofly.brain.neuron import LIFNeurons
from neurofly.brain.connectome import Connectome, Neuropil
from neurofly.brain.loader import generate_drosophila_connectome
from neurofly.brain.simulator import BrainSimulator


def test_lif_neuron_spiking_and_refractory():
    """Verify LIF neurons fire when depolarized past threshold and enter refractory period."""
    num_neurons = 5
    neurons = LIFNeurons(num_neurons=num_neurons, v_rest=-70.0, v_thresh=-50.0, refractory_steps=2, noise_std=0.0)

    # Initial state
    assert torch.allclose(neurons.v, torch.tensor(-70.0))

    # Sub-threshold injection
    i_sub = torch.full((num_neurons,), 5.0)
    spikes = neurons.step(i_inj=i_sub)
    assert spikes.sum().item() == 0.0
    assert torch.all(neurons.v > -70.0)

    # Strong suprathreshold injection
    i_strong = torch.full((num_neurons,), 60.0)
    spikes = neurons.step(i_inj=i_strong)
    assert spikes.sum().item() == num_neurons  # All spiked!

    # Next step: should be in refractory period
    assert torch.all(neurons.refractory_timer > 0)
    spikes_refractory = neurons.step(i_inj=i_strong)
    assert spikes_refractory.sum().item() == 0.0  # Cannot fire while refractory


def test_connectome_structure_and_neuropils():
    """Verify biological connectome generator partitions neuropils properly."""
    conn = generate_drosophila_connectome(scale="micro")
    assert conn.num_neurons == 320
    assert conn.num_synapses > 1000

    optic_l = conn.get_neuropil_indices(Neuropil.OPTIC_LOBE_LEFT)
    optic_r = conn.get_neuropil_indices(Neuropil.OPTIC_LOBE_RIGHT)
    cx = conn.get_neuropil_indices(Neuropil.CENTRAL_COMPLEX)
    dn = conn.get_neuropil_indices(Neuropil.DESCENDING_MOTOR)

    assert optic_l.numel() == 40
    assert optic_r.numel() == 40
    assert cx.numel() == 80
    assert dn.numel() == 30

    # Ensure 3D coordinates are valid
    assert conn.coordinates.shape == (320, 3)
    # Left optic lobe x should be negative, right should be positive
    assert np.mean(conn.coordinates[optic_l.cpu().numpy(), 0]) < 0
    assert np.mean(conn.coordinates[optic_r.cpu().numpy(), 0]) > 0


def test_brain_simulator_propagation():
    """Verify spikes in presynaptic neurons deliver synaptic currents to postsynaptic targets."""
    conn = generate_drosophila_connectome(scale="micro")
    sim = BrainSimulator(conn)

    # Inject large current into Optic Lobes
    optic_l = conn.get_neuropil_indices(Neuropil.OPTIC_LOBE_LEFT)
    sim.inject_current(optic_l, torch.full_like(optic_l, 80.0, dtype=torch.float32))

    # Step 1: Optic lobe fires
    spikes_step1 = sim.step()
    assert spikes_step1[optic_l].sum().item() > 0

    # Step 2: Spikes propagate to Central Complex and Descending Neurons
    spikes_step2 = sim.step()
    # Check that downstream neurons received current and some fired or got depolarized
    cx = conn.get_neuropil_indices(Neuropil.CENTRAL_COMPLEX)
    assert (sim.neurons.v[cx] > -70.0).any()
