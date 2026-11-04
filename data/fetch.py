"""Group data entry point; only the explicitly authorized Q1 offline mode exists."""

import argparse
import json
from pathlib import Path
import sys

from rich.console import Console
from rich.table import Table


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from q1.market_data import build_pipeline, load_offline_pipeline, load_registered_q1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", choices=["q1"], required=True)
    parser.add_argument("--offline", action="store_true", required=True,
                        help="Required: use committed verified snapshots; never download or rediscover sources")
    parser.add_argument("--output", type=Path, default=ROOT/"data/clean/q1/m2_pipeline")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest-sha256", help="Expected manifest hash for an unregistered output directory")
    args = parser.parse_args()
    if args.verify_only:
        if args.manifest_sha256:
            result = load_offline_pipeline(args.output, args.manifest_sha256)
        elif args.output.resolve() == ROOT/"data/clean/q1/m2_pipeline":
            result = load_registered_q1()
        else:
            parser.error("Supply --manifest-sha256 to verify an unregistered output")
    else:
        result = build_pipeline(args.output)
    coverage = result["coverage"]
    console = Console()
    console.print("[bold]Q1 OFFLINE PIPELINE — PARTIAL_HISTORICAL, not a complete examination panel[/bold]")
    table = Table("Evidence category", "Quotes", "Dates", "Expiries")
    for name in ["all_real_observations_including_separate_index_and_unknown_metadata_rows",
                 "target_etf_observations_not_automatically_eligible", "partial_diagnostic_preview_NOT_empirical",
                 "strictly_usable_for_empirical_analysis"]:
        item = coverage[name]
        table.add_row(name, str(item["quotes"]), str(item["dates"]), str(item["expiries"]))
    console.print(table)
    if "manifest_sha256" in result:
        console.print("Manifest SHA-256: " + result["manifest_sha256"])
    console.print(json.dumps({"data_readiness": coverage["dataset_readiness"],
                              "examination_historical_requirement_satisfied": coverage["examination_historical_requirement_satisfied"]}))


if __name__ == "__main__":
    main()
