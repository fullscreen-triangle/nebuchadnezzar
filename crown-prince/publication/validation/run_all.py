"""Master validation runner. Runs all three paper validations and produces a summary."""
import json
import datetime
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
scripts = [
    "validate_pharmacodynamics.py",
    "validate_pharmacokinetics.py",
    "validate_therapeutic_effect.py",
]

results_files = [
    "validation_pharmacodynamics.json",
    "validation_pharmacokinetics.json",
    "validation_therapeutic_effect.json",
]

print("=" * 70)
print("Running all paper validations from the Bounded Phase Space Law")
print("=" * 70)
for s in scripts:
    print(f"\n--- {s} ---")
    res = subprocess.run([sys.executable, str(here / s)], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR:", res.stderr)

# Aggregate
summary = {
    "timestamp": datetime.datetime.now().isoformat(),
    "axiom": "All persistent physical systems occupy bounded regions of phase space admitting partition and nesting.",
    "papers": []
}

total_tests = 0
total_passed = 0
for f in results_files:
    with open(here / f) as fp:
        data = json.load(fp)
    s = data["summary"]
    summary["papers"].append({
        "paper": data["paper"],
        "total_tests": s["total_tests"],
        "passed": s["passed"],
        "failed": s["failed"],
        "pass_rate": round(s["pass_rate"], 4),
        "results_file": f
    })
    total_tests += s["total_tests"]
    total_passed += s["passed"]

summary["aggregate"] = {
    "total_tests": total_tests,
    "passed": total_passed,
    "failed": total_tests - total_passed,
    "pass_rate": round(total_passed / total_tests if total_tests > 0 else 0.0, 4)
}

print("\n" + "=" * 70)
print("AGGREGATE SUMMARY")
print("=" * 70)
print(f"Total tests:    {total_tests}")
print(f"Passed:         {total_passed}")
print(f"Failed:         {total_tests - total_passed}")
print(f"Pass rate:      {100*total_passed/total_tests:.1f}%")

with open(here / "validation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary written to: {here / 'validation_summary.json'}")
