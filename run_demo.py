"""One-click launcher for NeuroFly: Web Playground, Benchmark Suite, or CLI Demo."""

import argparse
import sys
import time

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_web(host: str = "127.0.0.1", port: int = 8000):
    """Launch the Web Playground server."""
    import uvicorn
    print(f"🚀 Starting NeuroFly Web Playground at http://{host}:{port}")
    uvicorn.run("neurofly.web.server:app", host=host, port=port, reload=False)

def run_benchmark(num_episodes: int = 3, max_steps: int = 400):
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
        MaleCNSController(scale="micro"),
    ]

    print(f"Running {num_episodes} episodes x {max_steps} max steps across {len(controllers)} controllers...\n")
    runner = BenchmarkRunner(controllers, num_episodes=num_episodes, max_steps=max_steps)
    results = runner.run()

    table = BenchmarkRunner.format_markdown_table(results)
    print("\n" + "=" * 60)
    print("🏆 BENCHMARK RESULTS")
    print("=" * 60)
    print(table)

def run_cli(steps: int = 100):
    """Run CLI demo showing text-based telemetry."""
    from neurofly.environments import SurvivalArena
    from neurofly.controllers import MaleCNSController

    print("🪰 Initializing MaleCNS Connectome & Survival Arena...")
    arena = SurvivalArena(max_steps=steps)
    agent = MaleCNSController(scale="micro")

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

    args = parser.parse_args()

    if args.web:
        run_web(port=args.port)
    elif args.benchmark:
        run_benchmark(num_episodes=args.episodes)
    elif args.cli:
        run_cli(steps=args.steps)
    else:
        # Default: show help or launch web
        print("No mode specified. Use --web, --benchmark, or --cli.")
        parser.print_help()

if __name__ == "__main__":
    main()
