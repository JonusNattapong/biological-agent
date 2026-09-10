"""One-click launcher for NeuroFly: Web Playground, Benchmark Suite, or CLI Demo."""

import argparse
import os
import sys

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def _build_brain_controller(
    *,
    weights: str | None = None,
    annotations: str | None = None,
    neurotransmitters: str | None = None,
    min_synapses: int = 1,
    weight_transform: str = "log1p",
    weight_scale: float = 1.0,
    synthetic_scale: str = "micro",
):
    from neurofly.controllers import MaleCNSController

    if weights:
        return MaleCNSController.from_bulk_files(
            weights_path=weights,
            annotations_path=annotations,
            neurotransmitters_path=neurotransmitters,
            min_synapses=min_synapses,
            weight_transform=weight_transform,
            weight_scale=weight_scale,
        )
    return MaleCNSController(scale=synthetic_scale)


def run_web(
    host: str = "0.0.0.0",
    port: int = 8000,
    *,
    weights: str | None = None,
    annotations: str | None = None,
    neurotransmitters: str | None = None,
    min_synapses: int = 1,
    weight_transform: str = "log1p",
    weight_scale: float = 1.0,
):
    """Launch the Web Playground server."""
    import uvicorn

    if weights:
        os.environ["NEUROFLY_MALECNS_WEIGHTS"] = weights
        if annotations:
            os.environ["NEUROFLY_MALECNS_ANNOTATIONS"] = annotations
        if neurotransmitters:
            os.environ["NEUROFLY_MALECNS_NEUROTRANSMITTERS"] = neurotransmitters
        os.environ["NEUROFLY_MALECNS_MIN_SYNAPSES"] = str(min_synapses)
        os.environ["NEUROFLY_MALECNS_WEIGHT_TRANSFORM"] = weight_transform
        os.environ["NEUROFLY_MALECNS_WEIGHT_SCALE"] = str(weight_scale)
        print("🧠 Using official MaleCNS v1.0 bulk connectivity")
    else:
        print("🧪 Using structured synthetic Drosophila baseline (no MaleCNS weights supplied)")

    print(f"🚀 Starting NeuroFly Web Playground at http://{host}:{port}")
    uvicorn.run("neurofly.web.server:app", host=host, port=port, reload=False)


def run_benchmark(
    num_episodes: int = 3,
    max_steps: int = 400,
    *,
    weights: str | None = None,
    annotations: str | None = None,
    neurotransmitters: str | None = None,
    min_synapses: int = 1,
    weight_transform: str = "log1p",
    weight_scale: float = 1.0,
):
    """Run comparative benchmark across all controllers."""
    from neurofly.controllers import (
        RandomController,
        TinyNNController,
        HeuristicController,
        MaleCNSController,
    )
    from neurofly.benchmark import BenchmarkRunner

    print("🔬 Initializing NeuroFly Benchmark Suite...")
    controllers = [
        RandomController(),
        TinyNNController(),
        HeuristicController(),
        _build_brain_controller(
            weights=weights,
            annotations=annotations,
            neurotransmitters=neurotransmitters,
            min_synapses=min_synapses,
            weight_transform=weight_transform,
            weight_scale=weight_scale,
            synthetic_scale="micro",
        ),
    ]

    print(f"Running {num_episodes} episodes x {max_steps} max steps across {len(controllers)} controllers...\n")
    runner = BenchmarkRunner(controllers, num_episodes=num_episodes, max_steps=max_steps)
    results = runner.run()

    table = BenchmarkRunner.format_markdown_table(results)
    print("\n" + "=" * 60)
    print("🏆 BENCHMARK RESULTS")
    print("=" * 60)
    print(table)

def run_cli(
    steps: int = 100,
    *,
    weights: str | None = None,
    annotations: str | None = None,
    neurotransmitters: str | None = None,
    min_synapses: int = 1,
    weight_transform: str = "log1p",
    weight_scale: float = 1.0,
):
    """Run CLI demo showing text-based telemetry."""
    from neurofly.environments import SurvivalArena

    print("🪰 Initializing biological controller & Survival Arena...")
    arena = SurvivalArena(max_steps=steps)
    agent = _build_brain_controller(
        weights=weights,
        annotations=annotations,
        neurotransmitters=neurotransmitters,
        min_synapses=min_synapses,
        weight_transform=weight_transform,
        weight_scale=weight_scale,
        synthetic_scale="micro",
    )
    print(f"Brain source: {agent.name}")

    obs = arena.reset()
    agent.reset()

    print(f"\nRunning simulation for {steps} steps...\n")
    for s in range(steps):
        action = agent.act(obs)
        obs, reward, done, info = arena.step(action)
        telem = agent.get_brain_telemetry()

        if s % 10 == 0:
            print(
                f"[Step {s:03d}] Pos: ({arena.fly_pos[0]:6.1f}, {arena.fly_pos[1]:6.1f}) | "
                f"Energy: {arena.fly_energy:5.1f}% | Food Eaten: {info['food_eaten']} | "
                f"Col: {info['collisions']} | Brain Spikes: {telem['active_spikes_count']}"
            )

        if done:
            print(f"\nSimulation ended at step {s}. Reason: {info.get('death_reason')}")
            break

    print(f"\nFinal Stats: Survived {info['steps_survived']} steps, Distance: {info['total_distance']}, Food: {info['food_eaten']}")

def main():
    parser = argparse.ArgumentParser(description="NeuroFly: Biological Brain Runtime & Benchmark Arena")
    parser.add_argument("--web", action="store_true", help="Launch interactive Web Playground")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-controller benchmark suite")
    parser.add_argument("--cli", action="store_true", help="Run terminal console simulation")
    parser.add_argument("--port", type=int, default=8000, help="Web port (default: 8000)")
    parser.add_argument("--steps", type=int, default=150, help="Simulation steps for CLI")
    parser.add_argument("--episodes", type=int, default=3, help="Benchmark episodes per controller")
    parser.add_argument("--malecns-weights", help="official MaleCNS v1.0 connectome-weights Feather path")
    parser.add_argument("--malecns-annotations", help="official MaleCNS v1.0 body-annotations Feather path")
    parser.add_argument("--malecns-neurotransmitters", help="official MaleCNS v1.0 body-neurotransmitters Feather path")
    parser.add_argument("--min-synapses", type=int, default=1, help="minimum MaleCNS edge synapse count")
    parser.add_argument(
        "--weight-transform",
        choices=["raw", "sqrt", "log1p", "binary"],
        default="log1p",
        help="map MaleCNS synapse counts to simulator weights",
    )
    parser.add_argument("--weight-scale", type=float, default=1.0, help="simulation weight multiplier")

    args = parser.parse_args()
    dataset_kwargs = {
        "weights": args.malecns_weights,
        "annotations": args.malecns_annotations,
        "neurotransmitters": args.malecns_neurotransmitters,
        "min_synapses": args.min_synapses,
        "weight_transform": args.weight_transform,
        "weight_scale": args.weight_scale,
    }

    if args.web:
        run_web(port=args.port, **dataset_kwargs)
    elif args.benchmark:
        run_benchmark(num_episodes=args.episodes, max_steps=args.steps, **dataset_kwargs)
    elif args.cli:
        run_cli(steps=args.steps, **dataset_kwargs)
    else:
        # Default: show help or launch web
        print("No mode specified. Use --web, --benchmark, or --cli.")
        parser.print_help()

if __name__ == "__main__":
    main()
