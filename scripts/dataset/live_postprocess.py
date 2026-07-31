"""Post-prediction consumers: account ledger, market regime, website assets.

Run AFTER the three-model critical path has completed.  All steps read from
committed generations — they never mutate the prediction checkpoint.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=Path("dataset/cache/live"))
    parser.add_argument("--web-script", type=Path, default=Path("website/convert_data.py"))
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    python = sys.executable

    steps: list[tuple[str, list[str]]] = []
    if not args.skip_regime:
        steps.append(("Market regime", [python, "-m", "src.backtest.market_regime"]))
    steps.extend([
        ("Real account ledger (baseline)", [python, "-m", "scripts.evaluation.live_portfolio"]),
        ("Real account ledger (defend)", [python, "-m", "scripts.evaluation.live_portfolio",
                                          "--strategy", "defend",
                                          "--output-dir", "reports/strategy_v1/evidence_s2"]),
        ("Real account ledger (elite)", [python, "-m", "scripts.evaluation.live_portfolio",
                                         "--strategy", "elite",
                                         "--output-dir", "reports/strategy_v1/evidence_s3"]),
        ("Website assets", [python, str(args.web_script), "--triple"]),
    ])
    for label, command in steps:
        print(f"\n=== {label} ===")
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"{label} failed with exit code {completed.returncode}")
    print("\nPost-processing completed.")


if __name__ == "__main__":
    main()
