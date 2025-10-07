import os
import subprocess
import sys
import json

# Default parameters
default_iterations = 20
INITIAL_DATA_FILE = "data/initial_synthetic_text.txt"
EVALUATOR_FILE = "evaluator.py"
CONFIG_FILE = "config.yaml"

# Parse command line arguments
if len(sys.argv) > 1:
    ITERATIONS = sys.argv[1]
else:
    ITERATIONS = str(default_iterations)

print("🧬 Starting KG Synthetic Data Evolution")
print("=======================================")
print(f"Iterations: {ITERATIONS}")
print(f"Initial Data: {INITIAL_DATA_FILE}")
print(f"Evaluator: {EVALUATOR_FILE}")
print(f"Config: {CONFIG_FILE}")
print("")

# Check if required files exist
def check_file(path, msg):
    if not os.path.isfile(path):
        print(f"Error: {msg}")
        sys.exit(1)

if not os.path.isfile(INITIAL_DATA_FILE):
    print(f"Error: {INITIAL_DATA_FILE} not found")
    print("Creating initial synthetic data...")
    check_file("generate_initial_data.py", "generate_initial_data.py not found")
    subprocess.run([sys.executable, "generate_initial_data.py"], check=True)
    if not os.path.isfile(INITIAL_DATA_FILE):
        print("Error: Failed to generate initial data")
        sys.exit(1)

check_file(EVALUATOR_FILE, f"{EVALUATOR_FILE} not found")
check_file(CONFIG_FILE, f"{CONFIG_FILE} not found")

# Check API key
env_key = os.environ.get("OPENAI_API_KEY")
if not env_key:
    print("Error: OPENAI_API_KEY environment variable not set")
    print("Please set your OpenRouter API key:")
    print("export OPENAI_API_KEY='your_api_key_here'")
    sys.exit(1)

print("✅ All prerequisites checked\n")

# Evaluate baseline
print("📊 Evaluating baseline synthetic data...")
subprocess.run([
    sys.executable, "-c",
    (
        f"import json; from evaluator import evaluate_stage2; "
        f"result = evaluate_stage2('{INITIAL_DATA_FILE}'); "
        f"print(f'Baseline faithfulness: {{result.metrics[\"faithfulness_score\"]:.3f}}'); "
        f"f = open('baseline_evaluation.json', 'w'); "
        f"json.dump(result.metrics, f, indent=2); "
        f"f.close()"
    )
], check=True)

print("\n🔄 Starting data evolution process...")
print("This may take 20-40 minutes depending on iterations and API speed\n")

# Run evolution
subprocess.run([
    sys.executable, "openevolve/openevolve-run.py", INITIAL_DATA_FILE, EVALUATOR_FILE,
    "--config", CONFIG_FILE,
    "--iterations", ITERATIONS
], check=True)

print("\n📊 Evaluating evolved synthetic data...\n")

# Find the best evolved data
def find_best_data():
    import glob
    files = glob.glob("openevolve_output_*/best/best_program.*")
    return files[0] if files else None

EVOLVED_DATA = find_best_data()
if EVOLVED_DATA:
    print(f"Found evolved data: {EVOLVED_DATA}")
    subprocess.run([
        sys.executable, "-c",
        (
            f"import json; from evaluator import evaluate_stage2; "
            f"result = evaluate_stage2('{EVOLVED_DATA}'); "
            f"print(f'Evolved faithfulness: {{result.metrics[\"faithfulness_score\"]:.3f}}'); "
            f"f = open('evolved_evaluation.json', 'w'); "
            f"json.dump(result.metrics, f, indent=2); "
            f"f.close()"
        )
    ], check=True)
else:
    print("⚠️  Warning: No evolved data found in output directory")

print("\n🎉 Evolution complete!\n")
print("📂 Check the results:")
print("   - Baseline: baseline_evaluation.json")
print("   - Evolved: evolved_evaluation.json")
print("   - Best data: openevolve_output_*/best/best_program.*")
print("   - Logs: openevolve_output_*/logs/\n")

if os.path.isfile("baseline_evaluation.json") and os.path.isfile("evolved_evaluation.json"):
    print("💡 Performance comparison:")
    try:
        with open("baseline_evaluation.json") as f:
            baseline = json.load(f)
        with open("evolved_evaluation.json") as f:
            evolved = json.load(f)
        baseline_score = baseline.get('faithfulness_score', 0)
        evolved_score = evolved.get('faithfulness_score', 0)
        improvement = evolved_score - baseline_score
        print(f'  Baseline Faithfulness: {baseline_score:.3f}')
        print(f'  Evolved Faithfulness: {evolved_score:.3f}')
        if baseline_score > 0:
            print(f'  Improvement: {improvement:+.3f} ({improvement/baseline_score*100:+.1f}%)')
        else:
            print(f'  Improvement: {improvement:+.3f}')
        if 'entity_coverage' in baseline and 'entity_coverage' in evolved:
            print(f'  Entity Coverage: {baseline["entity_coverage"]:.3f} → {evolved["entity_coverage"]:.3f}')
    except Exception as e:
        print(f'  Error comparing results: {e}')
