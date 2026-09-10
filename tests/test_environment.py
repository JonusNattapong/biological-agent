"""Unit tests for Survival Arena environment physics and rewards."""

import numpy as np

from neurofly.environments.survival.arena import SurvivalArena
from neurofly.motor.decoder import MotorAction


def test_arena_reset_and_step():
    """Verify arena reset initializes entities and step updates physics."""
    arena = SurvivalArena(num_food=4, max_steps=50)
    obs = arena.reset()

    assert "visual" in obs
    assert "olfactory" in obs
    assert "tactile" in obs
    assert arena.fly_energy == 100.0
    assert len(arena.foods) == 4

    # Step forward
    action = MotorAction(forward_velocity=3.0, angular_velocity=0.0, feed_proboscis=False, raw_dn_activity=0.5)
    next_obs, reward, done, info = arena.step(action)

    assert info["steps_survived"] == 1
    assert info["total_distance"] == 3.0
    assert arena.fly_energy < 100.0  # Energy spent
    assert not done


def test_food_consumption():
    """Verify eating food replenishes energy and increases counter."""
    arena = SurvivalArena(num_food=1, max_steps=50)
    arena.reset()

    # Move food directly on top of fly
    arena.foods[0].pos = arena.fly_pos.copy()
    arena.fly_energy = 50.0

    action = MotorAction(forward_velocity=0.0, angular_velocity=0.0, feed_proboscis=True, raw_dn_activity=0.0)
    _, reward, _, info = arena.step(action)

    assert info["food_eaten"] == 1
    assert arena.fly_energy > 50.0  # Energy replenished
    assert reward > 10.0


def test_predator_catch_terminal():
    """Verify predator hitting fly ends episode."""
    arena = SurvivalArena(include_predator=True, max_steps=50)
    arena.reset()

    # Teleport predator onto fly
    arena.predator.pos = arena.fly_pos.copy()

    action = MotorAction(forward_velocity=0.0, angular_velocity=0.0, feed_proboscis=False, raw_dn_activity=0.0)
    _, _, done, info = arena.step(action)

    assert done is True
    assert info["death_reason"] == "eaten_by_predator"
