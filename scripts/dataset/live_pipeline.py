"""One-command transactional production refresh for Julong Quant."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineTimer:
    label: str = ""
    started: float = field(default_factory=time.perf_counter)
    steps: list[dict[str, Any]] = field(default_factory=list)

    def step(self, label: str) -> float:
        elapsed = time.perf_counter() - self.started
        self.steps.append({"step": label, "elapsed_s": round(elapsed, 3)})
        print(f"  [{elapsed:6.1f}s] {label}")
        return elapsed

    def done(self) -> float:
        total = time.perf_counter() - self.started
        boundary = self.label or "TOTAL"
        print(f"  [{total:6.1f}s] {boundary}")
        self.steps.append({"step": boundary, "elapsed_s": round(total, 3)})
        return total


def _run(label: str, command: list[str], cwd: Path) -> None:
    print(f"\n=== {label} ===")
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update partitions, publish prediction checkpoint, then refresh views"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--store-root", type=Path, default=Path("dataset/cache/live"))
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--skip-regime", action="store_true")
    parser.add_argument("--critical-only", action="store_true")
    parser.add_argument("--web-script", type=Path, default=Path("website/convert_data.py"))
    args = parser.parse_args()
    root = args.project_root.resolve()
    store_root = args.store_root
    if not store_root.is_absolute():
        store_root = root / store_root

    total = PipelineTimer(label="=== FULL PIPELINE ===")

    if not args.skip_data_update:
        t = PipelineTimer(label="Transactional market/factor update")
        from scripts.dataset.daily_update import update_live_store

        update_live_store(store_root=store_root)
        t.step("base + factor committed")
        t.done()

        import gc
        gc.collect()

        from scripts.dataset.daily_update import update_indices
        update_indices()
        total.step("index data updated")

    t_pred = PipelineTimer(label="Frozen-model live predictions")
    from src.predict.live_predictor_incremental import generate_live_predictions

    generate_live_predictions(
        exp_dir=root / "dataset/output/experiment_009",
        live_store_root=store_root,
        smooth_window=10,
    )
    t_pred.step("predictions + checkpoint exported")
    t_pred.done()

    checkpoint_elapsed = total.step("=== PREDICTION CHECKPOINT: Top 50 ready ===")

    if args.critical_only:
        total.done()
        return

    print("\n=== Export legacy monolithic snapshots ===")
    from scripts.dataset.daily_update import export_legacy
    export_legacy(store_root=store_root)
    total.step("legacy parquet synced from partitions")

    command = [
        sys.executable, "-m", "scripts.dataset.live_postprocess",
        "--store-root", str(store_root),
        "--web-script", str(args.web_script),
    ]
    if args.skip_regime:
        command.append("--skip-regime")
    _run("Independent account/regime/website consumers", command, root)

    total.step("postprocess complete")
    total.done()
    print("\nLive refresh completed successfully.")


if __name__ == "__main__":
    main()
