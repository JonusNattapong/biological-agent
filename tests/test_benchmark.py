"""Unit tests for the Benchmark Runner."""

from neurofly.controllers import RandomController, TinyNNController, MaleCNSController
from neurofly.benchmark import BenchmarkRunner


def test_benchmark_runner():
    """Verify running comparative benchmarks across multiple controllers."""
    controllers = [
        RandomController(),
        TinyNNController(),
        MaleCNSController(scale="micro"),
    ]

    runner = BenchmarkRunner(controllers, num_episodes=2, max_steps=20)
    results = runner.run()

    assert len(results) == 3
    for name, summary in results.items():
        assert summary.num_episodes == 2
        assert summary.mean_survival_steps > 0
        assert summary.mean_distance >= 0.0

    md_table = BenchmarkRunner.format_markdown_table(results)
    assert "| Controller |" in md_table
    assert "Structured Drosophila baseline (micro)" in md_table
