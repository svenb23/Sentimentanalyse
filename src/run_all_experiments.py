import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CONFIGS_DIR = BASE_DIR / "configs" / "experiments"

configs = list(CONFIGS_DIR.glob("*.yaml"))

print(f"Found {len(configs)} experiments\n")

for i, config in enumerate(configs, 1):
    print(f"\n{'='*60}")
    print(f"[{i}/{len(configs)}] Running: {config.stem}")
    print(f"{'='*60}")

    subprocess.run(["python", "src/run_experiment.py", "-c", str(config)], cwd=BASE_DIR)

print(f"\n{'='*60}")
print("All experiments completed!")
print("Run 'python src/compare_experiments.py' to compare results")
print(f"{'='*60}")
