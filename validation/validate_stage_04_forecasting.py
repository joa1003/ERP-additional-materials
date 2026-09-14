#!/usr/bin/env python3
"""Validate completion of the formal Stage 4 forecasting schedule or one bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LEADS = (1, 12, 48)
SEEDS = (42, 123, 2026, 31415)
STRATEGIES = (
    "lcl_source",
    "direct_transfer",
    "fine_tuning",
    "cer_scratch_limited",
    "cer_scratch_full",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--lead", type=int, choices=LEADS)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    args = ap.parse_args()
    if (args.lead is None) != (args.seed is None):
        ap.error("--lead and --seed must be supplied together")
    return args


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    pairs = [(args.lead, args.seed)] if args.lead is not None else [
        (lead, seed) for lead in LEADS for seed in SEEDS
    ]

    failures: list[str] = []
    checked = 0
    for lead, seed in pairs:
        for strategy in STRATEGIES:
            rel = Path(f"h{lead}") / strategy / f"seed_{seed}"
            table_dir = root / "outputs/tables/pooled_strategy_evaluation_formal" / rel
            metadata_dir = root / "outputs/metadata/pooled_strategy_evaluation_formal" / rel
            required = [
                table_dir / "test_predictions.parquet",
                table_dir / "test_pooled_metrics.csv",
                table_dir / "test_household_metrics.csv",
                metadata_dir / "run_metadata.md",
                metadata_dir / "run_status.json",
            ]
            for path in required:
                if not path.is_file() or path.stat().st_size == 0:
                    failures.append(f"missing/empty: {path.relative_to(root)}")
            status_path = metadata_dir / "run_status.json"
            if status_path.is_file():
                try:
                    payload = json.loads(status_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    failures.append(f"invalid JSON: {status_path.relative_to(root)}: {exc}")
                else:
                    expected = {
                        "status": "COMPLETE_PASS",
                        "strategy": strategy,
                        "seed": seed,
                        "horizon": lead,
                        "test_rows_used_for_training_or_selection": 0,
                    }
                    for key, value in expected.items():
                        if payload.get(key) != value:
                            failures.append(
                                f"status mismatch: {status_path.relative_to(root)}: "
                                f"{key}={payload.get(key)!r}, expected {value!r}"
                            )
            checked += 1

    scope = (
        f"h={args.lead}, seed={args.seed}" if args.lead is not None
        else "complete 3-lead x 4-seed x 5-strategy schedule"
    )
    print("=" * 96)
    print("STAGE 4 FORMAL FORECASTING COMPLETION AUDIT")
    print("=" * 96)
    print("Scope             :", scope)
    print("Runs checked      :", checked)
    print("Expected run files: predictions + pooled metrics + household metrics + metadata + status")

    if failures:
        print("Status            : FAIL")
        for item in failures[:100]:
            print("-", item)
        if len(failures) > 100:
            print(f"- ... {len(failures)-100} additional failures")
        return 1

    print("Status            : PASS_STAGE4_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
