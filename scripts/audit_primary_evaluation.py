#!/usr/bin/env python3
"""Run the independent report auditor on a complete primary evaluation only.

This does not finalize the experiment or read incomplete sensitivity branches.
The full report repeats these checks after all required branches finish.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path

import finalize_report as auditor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("."))
    args = parser.parse_args()
    project = args.project.resolve()
    bundle = auditor.audit_project(project)
    acceptance, contrasts = auditor.validate_statistics(bundle)
    paths = ["FINAL_EVAL_LOCK.json", "artifacts/PREDICTION_COMPLETENESS.json",
             "artifacts/predictions.parquet", "reports/final/evaluation_manifest.json",
             "reports/final/main_table.csv", "reports/final/metrics_by_source_seed.csv",
             "reports/final/comparisons.json", "reports/final/acceptance.json",
             "reports/final/bootstrap_pair.json", "reports/final/bootstrap_pair.npz",
             "reports/final/bootstrap_pigeonhole.json", "reports/final/bootstrap_pigeonhole.npz"]
    report = {
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS", "scope": "Complete primary evaluation only",
        "independent_metric_and_leakage_audit": bundle["audit"],
        "statistics_checks": ["bootstrap coverage and identities", "registered contrast coverage",
                              "independent Holm correction", "primary acceptance reconstruction"],
        "registered_contrasts": len(contrasts), "primary_acceptance": acceptance,
        "artifact_sha256": {path: auditor.sha256(project / path) for path in paths},
        "runner_sha256": auditor.sha256(__file__),
        "sensitivity_results_read": False, "experimental_completion_claimed": False,
        "limitation": "Bootstrap draws were checked, not independently regenerated; both training sensitivities and full final reporting remain required."
    }
    output = project / "reports/PRIMARY_EVALUATION_AUDIT.json"
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": "PASS", "checked_prediction_rows": len(bundle["predictions"]),
                      "registered_contrasts": len(contrasts), "report": str(output),
                      "experimental_completion_claimed": False}))


if __name__ == "__main__":
    main()
