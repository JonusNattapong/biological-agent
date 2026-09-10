"""Comparative benchmark runner across multiple controllers."""

from typing import Callable, Dict, List, Optional
import numpy as np

from neurofly.benchmark.metrics import EpisodeResult, BenchmarkSummary
from neurofly.controllers.base import BaseController
from neurofly.environments.base import BaseEnvironment
from neurofly.environments.survival.arena import SurvivalArena


class BenchmarkRunner:
    """Runs standardized evaluation benchmarks comparing biological and artificial controllers."""

    def __init__(
        self,
        controllers: List[BaseController],
        env_factory: Optional[Callable[[], BaseEnvironment]] = None,
        num_episodes: int = 5,
        max_steps: int = 600,
    ):
        self.controllers = controllers
        self.env_factory = env_factory or (lambda: SurvivalArena(max_steps=max_steps))
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def run(self) -> Dict[str, BenchmarkSummary]:
        """Execute benchmark episodes for all controllers and return aggregated metrics."""
        summaries: Dict[str, BenchmarkSummary] = {}

        for controller in self.controllers:
            episode_results: List[EpisodeResult] = []

            for ep in range(self.num_episodes):
                env = self.env_factory()
                obs = env.reset()
                controller.reset()

                total_reward = 0.0
                dn_activities = []
                speeds = []

                for step in range(self.max_steps):
                    action = controller.act(obs)
                    obs, reward, done, info = env.step(action)

                    total_reward += reward
                    if hasattr(action, "forward_velocity"):
                        speeds.append(action.forward_velocity)
                    if hasattr(action, "raw_dn_activity"):
                        dn_activities.append(action.raw_dn_activity)

                    if done:
                        break

                res = EpisodeResult(
                    controller_name=controller.name,
                    episode_id=ep,
                    steps_survived=info.get("steps_survived", 0),
                    total_distance=info.get("total_distance", 0.0),
                    food_eaten=info.get("food_eaten", 0),
                    collisions=info.get("collisions", 0),
                    final_energy=info.get("energy", 0.0),
                    total_reward=round(total_reward, 2),
                    death_reason=info.get("death_reason"),
                    mean_speed=float(np.mean(speeds)) if speeds else 0.0,
                    mean_dn_activity=float(np.mean(dn_activities)) if dn_activities else 0.0,
                )
                episode_results.append(res)

            # Aggregate stats
            surv_steps = [r.steps_survived for r in episode_results]
            foods = [r.food_eaten for r in episode_results]
            dists = [r.total_distance for r in episode_results]
            cols = [r.collisions for r in episode_results]
            energies = [r.final_energy for r in episode_results]
            rewards = [r.total_reward for r in episode_results]
            survived_count = sum(1 for r in episode_results if r.death_reason == "time_limit_reached" or r.death_reason is None)

            summary = BenchmarkSummary(
                controller_name=controller.name,
                num_episodes=self.num_episodes,
                mean_survival_steps=float(np.mean(surv_steps)),
                std_survival_steps=float(np.std(surv_steps)),
                mean_food_eaten=float(np.mean(foods)),
                std_food_eaten=float(np.std(foods)),
                mean_distance=float(np.mean(dists)),
                mean_collisions=float(np.mean(cols)),
                mean_final_energy=float(np.mean(energies)),
                survival_rate_pct=float((survived_count / self.num_episodes) * 100.0),
                mean_reward=float(np.mean(rewards)),
            )
            summaries[controller.name] = summary

        return summaries

    @staticmethod
    def format_markdown_table(summaries: Dict[str, BenchmarkSummary]) -> str:
        """Format benchmark summaries into a GitHub Markdown table."""
        header = (
            "| Controller | Survival Steps | Food Eaten | Distance | Collisions | Final Energy | Survival % | Mean Reward |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        rows = []
        for name, s in summaries.items():
            row = (
                f"| **{s.controller_name}** "
                f"| {s.mean_survival_steps:.1f} +/- {s.std_survival_steps:.1f} "
                f"| {s.mean_food_eaten:.1f} +/- {s.std_food_eaten:.1f} "
                f"| {s.mean_distance:.1f} "
                f"| {s.mean_collisions:.1f} "
                f"| {s.mean_final_energy:.1f} "
                f"| {s.survival_rate_pct:.0f}% "
                f"| {s.mean_reward:.1f} |"
            )
            rows.append(row)
        return header + "\n".join(rows)
