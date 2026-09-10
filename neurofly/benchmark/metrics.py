"""Metrics collection and evaluation for agent benchmark comparisons."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EpisodeResult:
    """Telemetry and outcome metrics from a single evaluation episode."""
    controller_name: str
    episode_id: int
    steps_survived: int
    total_distance: float
    food_eaten: int
    collisions: int
    final_energy: float
    total_reward: float
    death_reason: Optional[str]
    mean_speed: float
    mean_dn_activity: float


@dataclass
class BenchmarkSummary:
    """Aggregated statistics across multiple evaluation episodes."""
    controller_name: str
    num_episodes: int
    mean_survival_steps: float
    std_survival_steps: float
    mean_food_eaten: float
    std_food_eaten: float
    mean_distance: float
    mean_collisions: float
    mean_final_energy: float
    survival_rate_pct: float  # Percentage that survived max_steps without death
    mean_reward: float
