"""Unit tests for sensory perception and motor decoding."""

import numpy as np
import torch

from neurofly.sensory.vision import CompoundEyeVision
from neurofly.sensory.olfaction import AntennaeOlfaction
from neurofly.sensory.mechanosensory import Mechanosensation
from neurofly.sensory.encoder import SensoryEncoder
from neurofly.motor.decoder import MotorDecoder
from neurofly.brain.loader import generate_drosophila_connectome
from neurofly.brain.connectome import Neuropil


def test_vision_looming_and_ommatidia():
    """Verify compound eyes detect approaching threat expansion and food positions."""
    vision = CompoundEyeVision(num_ommatidia_per_eye=16)

    fly_pos = np.array([0.0, 0.0])
    fly_heading = 0.0  # Facing +X

    # Food on the right side (+Y)
    food_positions = [np.array([50.0, 50.0])]
    # Threat approaching from the left (-Y)
    threat_pos = np.array([50.0, -40.0])

    # Tick 1: Initialize distance
    v1 = vision.perceive(fly_pos, fly_heading, food_positions, threat_pos)
    # Tick 2: Threat rushes closer!
    threat_pos_closer = np.array([20.0, -15.0])
    v2 = vision.perceive(fly_pos, fly_heading, food_positions, threat_pos_closer)

    # Looming on left eye should be detected
    assert v2.looming_threat_left > 0.1
    # Right eye intensity should be positive due to food on the right
    assert v2.right_intensity > 0.0


def test_olfaction_gradient():
    """Verify antennae sense spatial chemical concentration gradient."""
    olfaction = AntennaeOlfaction(antenna_span=4.0)

    fly_pos = np.array([0.0, 0.0])
    fly_heading = 0.0  # Normal is [-sin(0), cos(0)] = [0, 1]
    # Food is on the left side (+Y in global coordinates)
    food_positions = [np.array([0.0, 30.0])]

    stim = olfaction.perceive(fly_pos, fly_heading, food_positions)
    # Left antenna should be closer to food than right antenna
    assert stim.left_concentration > stim.right_concentration
    assert stim.gradient > 0.0


def test_sensory_encoder_and_motor_decoder():
    """Verify sensory encoder injects currents and motor decoder translates DN spikes."""
    conn = generate_drosophila_connectome(scale="micro")
    encoder = SensoryEncoder(conn)
    decoder = MotorDecoder(conn)

    vision = CompoundEyeVision()
    olfaction = AntennaeOlfaction()
    mech = Mechanosensation()

    v_stim = vision.perceive(np.zeros(2), 0.0, [np.array([10.0, 0.0])])
    o_stim = olfaction.perceive(np.zeros(2), 0.0, [np.array([10.0, 0.0])])
    t_stim = mech.perceive(100.0, False, 9999.0)

    indices, currents = encoder.encode(v_stim, o_stim, t_stim)
    assert indices.numel() > 0
    assert currents.numel() > 0

    # Test motor decoder with mock spikes in descending motor neurons
    mock_spikes = torch.zeros(conn.num_neurons)
    dn_indices = conn.get_neuropil_indices(Neuropil.DESCENDING_MOTOR)
    mock_spikes[dn_indices] = 1.0

    action = decoder.decode(mock_spikes)
    assert action.forward_velocity > 0.0
    assert isinstance(action.angular_velocity, float)
