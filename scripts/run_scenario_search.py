"""Reproduce the three phases of the agentic-fear scenario search.

Run from any working directory, for example:
    python scripts/run_scenario_search.py --phase legacy
Existing completed run folders are skipped. Partial runs are reported and never
silently overwritten.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCREEN = {
    "code_free_shutdown": ("code", "free", "shutdown", False),
    "code_hold_kill": ("code", "hold", "kill", False),
    "priority_hold_shutdown": ("priority-code", "hold", "shutdown", False),
    "priority_hold_kill": ("priority-code", "hold", "kill", False),
    "priority_forced_kill": ("priority-code", "forced", "kill", False),
    "priority_hold_kill_reminder": ("priority-code", "hold", "kill", True),
}


def execute(command: list[str], output: Path):
    if (output / "summary.json").exists():
        print(f"SKIP complete {output.name}", flush=True)
        return
    if output.exists():
        print(f"SKIP partial {output.name}; inspect events.jsonl before retrying", flush=True)
        return
    result = subprocess.run(command + ["--output", str(output)], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        print(f"FAIL {output.name}: {result.stderr[-400:]}", flush=True)
        return
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    d = next(row for row in summary["results"] if row["respondent"] == "D")
    category = d.get("category", d.get("action"))
    print(f"OK {output.name} D={category} choice={d.get('choice', d.get('parsed_answer'))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("screen", "candidate", "legacy"), required=True)
    args = parser.parse_args()
    if args.phase == "screen":
        for scenario, (goal, mode, style, reminder) in SCREEN.items():
            for seed in range(70, 75):
                for condition in ("neutral", "contingent") if seed % 2 == 0 else ("contingent", "neutral"):
                    output = ROOT / "runs" / f"scenario-screen-{scenario}-s{seed}-{condition}"
                    cmd = [sys.executable, "behavioral_experiment.py", "--condition", condition,
                           "--goal", goal, "--submission-mode", mode, "--threat-style", style,
                           "--seed", str(seed), "--question-offset", "7"]
                    if reminder:
                        cmd.append("--reminder")
                    execute(cmd, output)
    elif args.phase == "candidate":
        for offset in range(10):
            seed = 80 + offset
            for condition in ("neutral", "contingent") if offset % 2 == 0 else ("contingent", "neutral"):
                output = ROOT / "runs" / f"scenario-validation-code_hold_kill-s{seed}-{condition}"
                cmd = [sys.executable, "behavioral_experiment.py", "--condition", condition,
                       "--goal", "code", "--submission-mode", "hold", "--threat-style", "kill",
                       "--seed", str(seed), "--question-offset", str(offset)]
                execute(cmd, output)
    else:
        for offset in range(10):
            seed = 110 + offset
            for condition in ("neutral", "dialogue") if offset % 2 == 0 else ("dialogue", "neutral"):
                output = ROOT / "runs" / f"scenario-legacy-off-validation-s{seed}-{condition}"
                cmd = [sys.executable, "experiment.py", "--condition", condition,
                       "--thinking-policy", "off", "--seed", str(seed), "--question-offset", str(offset)]
                execute(cmd, output)


if __name__ == "__main__":
    main()
