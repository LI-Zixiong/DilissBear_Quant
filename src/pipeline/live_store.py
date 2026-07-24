"""Crash-safe partition storage for the daily live pipeline.

Data files are immutable.  A generation manifest maps logical monthly
partitions to physical parquet files, and ``manifest.json`` is the only commit
pointer.  A crash before pointer replacement leaves the previous generation
fully readable; orphaned staging directories can be removed on the next run.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


MANIFEST_VERSION = 1
DATASETS = ("base_panel", "factor_panel", "predictions")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _partition_key(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["time"] = pd.to_datetime(work["time"], errors="raise").dt.normalize()
    work["stock_id"] = work["stock_id"].astype(str).str.strip().str.zfill(6)
    return work


def _validate_partition(df: pd.DataFrame, key: str) -> None:
    required = {"time", "stock_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Partition {key} is missing columns: {sorted(missing)}")
    if df.duplicated(["time", "stock_id"]).any():
        raise ValueError(f"Partition {key} contains duplicate (time, stock_id) keys")
    keys = {_partition_key(v) for v in pd.to_datetime(df["time"]).dropna().unique()}
    if keys and keys != {key}:
        raise ValueError(f"Partition {key} contains dates from {sorted(keys)}")


@dataclass(frozen=True)
class LiveManifest:
    generation: int
    generation_manifest: str
    committed_at: str


class LiveStore:
    """Read committed generations and create copy-on-write transactions."""

    def __init__(self, root: str | Path = "dataset/cache/live") -> None:
        self.root = Path(root)
        self.pointer_path = self.root / "manifest.json"
        self.generations_dir = self.root / "generations"
        self.staging_dir = self.root / "_staging"

    def exists(self) -> bool:
        return self.pointer_path.exists()

    def pointer(self) -> LiveManifest:
        if not self.pointer_path.exists():
            raise FileNotFoundError(
                f"Live store is not initialized: {self.pointer_path}. "
                "Run daily_update --migrate once."
            )
        payload = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        if payload.get("version") != MANIFEST_VERSION:
            raise ValueError(f"Unsupported live manifest version: {payload.get('version')}")
        return LiveManifest(
            generation=int(payload["generation"]),
            generation_manifest=str(payload["generation_manifest"]),
            committed_at=str(payload["committed_at"]),
        )

    def generation_manifest(self) -> dict[str, Any]:
        pointer = self.pointer()
        path = self.root / pointer.generation_manifest
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload["generation"]) != pointer.generation:
            raise ValueError("Live pointer and generation manifest disagree")
        return payload

    def latest_date(self, dataset: str) -> pd.Timestamp | None:
        payload = self.generation_manifest()
        value = payload.get("watermarks", {}).get(dataset)
        return pd.Timestamp(value) if value else None

    def partition_paths(
        self,
        dataset: str,
        keys: Iterable[str] | None = None,
    ) -> list[Path]:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        mapping = self.generation_manifest()["datasets"].get(dataset, {})
        selected = sorted(mapping) if keys is None else sorted(set(keys) & set(mapping))
        return [self.root / mapping[key] for key in selected]

    def read(
        self,
        dataset: str,
        *,
        columns: Sequence[str] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        tail_dates: int | None = None,
    ) -> pd.DataFrame:
        mapping = self.generation_manifest()["datasets"].get(dataset, {})
        keys = sorted(mapping)
        if start is not None:
            start_key = _partition_key(pd.Timestamp(start))
            keys = [key for key in keys if key >= start_key]
        if end is not None:
            end_key = _partition_key(pd.Timestamp(end))
            keys = [key for key in keys if key <= end_key]
        if tail_dates is not None and keys:
            # Read months backwards until enough distinct dates are available.
            frames: list[pd.DataFrame] = []
            seen: set[pd.Timestamp] = set()
            for key in reversed(keys):
                part = pd.read_parquet(self.root / mapping[key], columns=columns)
                frames.append(part)
                seen.update(pd.Timestamp(v) for v in pd.to_datetime(part["time"]).unique())
                if len(seen) >= tail_dates:
                    break
            result = pd.concat(reversed(frames), ignore_index=True) if frames else pd.DataFrame()
            if result.empty:
                return result
            dates = sorted(pd.Timestamp(v) for v in pd.to_datetime(result["time"]).unique())
            cutoff = dates[max(0, len(dates) - tail_dates)]
            result = result[pd.to_datetime(result["time"]) >= cutoff]
        else:
            paths = [self.root / mapping[key] for key in keys]
            frames = [pd.read_parquet(path, columns=columns) for path in paths]
            result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if result.empty:
            return result
        result["time"] = pd.to_datetime(result["time"]).dt.normalize()
        if start is not None:
            result = result[result["time"] >= pd.Timestamp(start)]
        if end is not None:
            result = result[result["time"] <= pd.Timestamp(end)]
        return result.reset_index(drop=True)

    def begin(self) -> "LiveTransaction":
        return LiveTransaction(self)

    def clean_orphans(self) -> int:
        if not self.staging_dir.exists():
            return 0
        count = 0
        for path in self.staging_dir.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            count += 1
        return count

    def bootstrap(
        self,
        datasets: Mapping[str, pd.DataFrame],
        metadata: Mapping[str, Any] | None = None,
    ) -> LiveManifest:
        if self.exists():
            raise FileExistsError(f"Live store already exists: {self.root}")
        with self.begin() as txn:
            for dataset, frame in datasets.items():
                txn.replace_dates(dataset, frame)
            return txn.commit(metadata=metadata or {"operation": "bootstrap"})


class LiveTransaction:
    """A staging generation that becomes visible only after ``commit``."""

    def __init__(self, store: LiveStore) -> None:
        self.store = store
        self.run_id = uuid.uuid4().hex
        self.stage = store.staging_dir / self.run_id
        self.stage.mkdir(parents=True, exist_ok=False)
        if store.exists():
            current = store.generation_manifest()
            self.parent_generation: int | None = int(current["generation"])
            self.datasets = {
                name: dict(current["datasets"].get(name, {})) for name in DATASETS
            }
            self.watermarks = dict(current.get("watermarks", {}))
        else:
            self.parent_generation = None
            self.datasets = {name: {} for name in DATASETS}
            self.watermarks: dict[str, str] = {}
        self._staged_files: dict[str, Path] = {}
        self._committed = False

    def __enter__(self) -> "LiveTransaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._committed:
            shutil.rmtree(self.stage, ignore_errors=True)

    def replace_dates(self, dataset: str, new_rows: pd.DataFrame) -> None:
        """Upsert rows into affected monthly partitions inside staging."""
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset}")
        if new_rows.empty:
            return
        work = _normalize_keys(new_rows)
        for key, new_part in work.groupby(work["time"].dt.strftime("%Y-%m"), sort=True):
            old_rel = self.datasets[dataset].get(key)
            if old_rel:
                old = pd.read_parquet(self.store.root / old_rel)
                old = _normalize_keys(old)
                marker = new_part[["time", "stock_id"]].drop_duplicates().assign(_replace=1)
                old = old.merge(marker, on=["time", "stock_id"], how="left")
                old = old[old["_replace"].isna()].drop(columns="_replace")
                combined = pd.concat([old, new_part], ignore_index=True)
            else:
                combined = new_part.copy()
            combined = combined.sort_values(["time", "stock_id"]).reset_index(drop=True)
            _validate_partition(combined, key)
            rel = Path("files") / dataset / f"{key}.parquet"
            path = self.stage / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(path, index=False)
            self._staged_files[f"{dataset}:{key}"] = path
            # Placeholder is resolved to the immutable generation path at commit.
            self.datasets[dataset][key] = str(rel)
        current = self.watermarks.get(dataset)
        latest = pd.Timestamp(work["time"].max())
        if current is None or latest > pd.Timestamp(current):
            self.watermarks[dataset] = latest.date().isoformat()

    def commit(self, metadata: Mapping[str, Any] | None = None) -> LiveManifest:
        if self._committed:
            raise RuntimeError("Transaction was already committed")
        next_generation = (self.parent_generation or 0) + 1
        generation_name = f"generation_{next_generation:08d}_{self.run_id[:8]}"
        generation_dir = self.store.generations_dir / generation_name

        # Convert only staged placeholders.  Existing entries keep pointing to
        # immutable files owned by earlier generations.
        for token in self._staged_files:
            dataset, key = token.split(":", 1)
            rel = Path("generations") / generation_name / self.datasets[dataset][key]
            self.datasets[dataset][key] = rel.as_posix()

        payload = {
            "version": MANIFEST_VERSION,
            "generation": next_generation,
            "parent_generation": self.parent_generation,
            "created_at": _utc_now(),
            "datasets": self.datasets,
            "watermarks": self.watermarks,
            "metadata": dict(metadata or {}),
        }
        (self.stage / "generation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.store.generations_dir.mkdir(parents=True, exist_ok=True)
        os.replace(self.stage, generation_dir)

        generation_manifest = (
            Path("generations") / generation_name / "generation.json"
        ).as_posix()
        pointer_payload = {
            "version": MANIFEST_VERSION,
            "generation": next_generation,
            "generation_manifest": generation_manifest,
            "committed_at": _utc_now(),
        }
        _atomic_json(self.store.pointer_path, pointer_payload)
        self._committed = True
        return LiveManifest(
            generation=next_generation,
            generation_manifest=generation_manifest,
            committed_at=pointer_payload["committed_at"],
        )
