#!/usr/bin/env python3
"""Verify and reproduce saved analysis on CPU, without importing the full pipeline.

The source file is trusted project code. Only an explicit allowlist of function
definitions is extracted with ast; no notebook cell or source module is run.
No dataset images, TensorFlow, model weights, or network access are needed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

PACKAGE = Path(__file__).resolve().parents[1]
METRICS = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]
FUNCTIONS = (
    "frame_hash", "task_frame", "make_partitions", "validate_partitions",
    "partition_assignments", "select_threshold", "metric_values",
    "validate_prediction_frame", "sample_cluster_indices", "cluster_bootstrap",
    "threshold_robustness", "_viewer_runs", "_viewer_json", "_viewer_csv",
    "_viewer_sha", "_viewer_object_sha", "_viewer_evidence",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_object(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_original_functions(source):
    # These CPU imports are deliberately independent of the full source imports.
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, f1_score, precision_score,
        recall_score, roc_auc_score,
    )
    from sklearn.model_selection import train_test_split

    tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    selected = []
    for name in FUNCTIONS:
        # Viewer helpers are also embedded in the recovery guard. Select their
        # final, public top-level definitions used by final_polish.
        scope = tree.body if name.startswith("_viewer_") else ast.walk(tree)
        matches = [node for node in scope
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        require(len(matches) == 1, f"Expected one original definition of {name}")
        require(not matches[0].decorator_list, f"Unexpected decorator on {name}")
        selected.append(matches[0])
    namespace = dict(
        np=np, pd=pd, Path=Path, csv=csv, hashlib=hashlib, json=json, math=math,
        accuracy_score=accuracy_score, confusion_matrix=confusion_matrix,
        f1_score=f1_score, precision_score=precision_score,
        recall_score=recall_score, roc_auc_score=roc_auc_score,
        train_test_split=train_test_split, METRIC_NAMES=METRICS,
        VIEWER_SEEDS=(17, 29, 42, 71, 101),
        VIEWER_TASK1_REVISION="fixed_image_tta_v2_padded_physical_batch",
    )
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    require("tensorflow" not in sys.modules and "keras" not in sys.modules,
            "CPU checker unexpectedly imported a model framework")
    return namespace


def verify_ledger(evidence):
    with (evidence / "artifact_verification.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), "Empty artifact ledger")
    require(len({row["artifact"] for row in rows}) == len(rows),
            "Duplicate artifact ledger entries")
    for row in rows:
        path = (evidence / row["artifact"]).resolve()
        require(path.is_relative_to(evidence), "Artifact path escapes evidence directory")
        require(path.is_file(), f"Missing recorded artifact: {row['artifact']}")
        require(path.stat().st_size == int(row["bytes"]),
                f"Recorded size mismatch: {row['artifact']}")
        require(sha256(path) == row["sha256"],
                f"Recorded SHA-256 mismatch: {row['artifact']}")
    return len(rows)


def compare_table(actual, expected, description):
    import pandas as pd

    require(set(actual.columns) == set(expected.columns),
            f"Different columns: {description}")
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected[actual.columns].reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-10, rtol=0,
        obj=description,
    )


def reproduce(args):
    original = load_original_functions(args.source)
    np, pd = original["np"], original["pd"]
    evidence, output = args.evidence, args.output
    require(evidence.is_dir(), f"Evidence directory missing: {evidence}")
    require(not output.is_relative_to(evidence) and not evidence.is_relative_to(output),
            "Output and evidence directories must be separate and nonoverlapping")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())),
            "Output must be a new or empty directory; nothing will be overwritten")
    status = json_object(evidence / "completion_status.json")
    require(status.get("analysis_complete") is True
            and status.get("evaluated_runs_verified") == 16,
            "Saved evidence does not record all 16 analyses complete")
    protocol = json_object(evidence / "protocol.json")
    fingerprint = json_object(evidence / "dataset_fingerprint.json")
    patch = json_object(evidence / "reporting_patch.json")
    artifact_count = verify_ledger(evidence)
    manifest = pd.read_csv(evidence / "audit/manifest.csv", keep_default_na=False)
    require(not manifest.image_id.duplicated().any()
            and len(manifest) == fingerprint["images"], "Manifest coverage mismatch")
    identity_columns = ["image_id", "file_sha256", "pixel_sha256", "qualified_key",
                        "split_group", "label_task1", "label_task2"]
    require(original["frame_hash"](manifest, identity_columns)
            == fingerprint["content_sha256"], "Manifest content fingerprint mismatch")
    runs = original["_viewer_runs"]()
    recorded = pd.read_csv(evidence / "all_run_results.csv")
    require(len(recorded) == 16 and not recorded.run_id.duplicated().any()
            and set(recorded.run_id) == {run["run_id"] for run in runs},
            "All-run table does not contain the exact 16-run schedule")
    recorded = recorded.set_index("run_id")
    completed, tables, interval_jobs = [], {}, []
    for run in runs:
        run_id = run["run_id"]
        directory = evidence / "runs" / run_id
        record, revision = original["_viewer_evidence"](
            directory, run, task1_repair_required=patch.get("revision") == "V3.1"
        )
        config = json_object(directory / "run_configuration.json")
        require(config["protocol"] == protocol
                and config["content_sha256"] == fingerprint["content_sha256"],
                f"Protocol or dataset fingerprint mismatch: {run_id}")
        assignments = pd.read_csv(directory / "split_assignments.csv", keep_default_na=False)
        columns = ["image_id", "label", "split_group", "partition"]
        require(original["frame_hash"](assignments, columns) == config["split_sha256"],
                f"Saved split hash mismatch: {run_id}")
        frame = original["task_frame"](manifest, run["task"])
        parts = original["make_partitions"](frame, run["design"], run["seed"])
        rebuilt = original["partition_assignments"](parts)
        # The original exported assignments preserve train/tuning/test order.
        require(original["frame_hash"](rebuilt, columns) == config["split_sha256"],
                f"Partitions cannot be reproduced with this environment: {run_id}")
        require(set(assignments.partition) == {"train", "tuning", "test"},
                f"Unexpected partition name: {run_id}")
        metadata_columns = ["image_id", "source_split", "qualified_key", "raw_key",
                            "pixel_sha256", "file_sha256", "split_group", "label", "partition"]
        compare_table(assignments[metadata_columns], rebuilt[metadata_columns],
                      f"Partition manifest metadata: {run_id}")
        positive = "PNEUMONIA" if run["task"] == "task1" else "VIRAL"
        predictions = {}
        for partition in ("tuning", "test"):
            predictions[partition] = pd.read_csv(
                directory / f"{partition}_predictions.csv", keep_default_na=False
            )
            original["validate_prediction_frame"](
                predictions[partition], parts[partition], positive
            )
        tuning, test = predictions["tuning"], predictions["test"]
        threshold = original["select_threshold"](tuning.y_true, tuning.probability)
        require(math.isclose(threshold, float(record["threshold"]), abs_tol=1e-12, rel_tol=0),
                f"Tuning-selected threshold differs: {run_id}")
        values = original["metric_values"](test.y_true, test.probability, threshold)
        for name, value in dict(threshold=threshold, **values).items():
            stored = record["threshold"] if name == "threshold" else record["metrics"][name]
            require(np.isclose(value, stored, atol=1e-10, rtol=0)
                    and np.isclose(value, recorded.loc[run_id, name], atol=1e-10, rtol=0),
                    f"Recomputed {name} differs: {run_id}")
        result = dict(**run, positive_class=positive, threshold=threshold, **values)
        compare_table(pd.DataFrame([result]), pd.read_csv(directory / "test_metrics.csv"),
                      f"Per-run metrics: {run_id}")
        sensitivity = original["threshold_robustness"](test)
        compare_table(sensitivity, pd.read_csv(directory / "threshold_sensitivity.csv"),
                      f"Threshold sensitivity: {run_id}")
        tables[run_id] = {"test_metrics.csv": pd.DataFrame([result]),
                          "threshold_sensitivity.csv": sensitivity,
                          "reproduced_split_assignments.csv": rebuilt[columns]}
        has_intervals = (run["design"] == "filename_grouped" and run["seed"] == 42
                         or run["design"] == "original_folders")
        if has_intervals:
            stored_intervals = pd.read_csv(directory / "bootstrap_intervals.csv")
            require(len(stored_intervals) == len(METRICS)
                    and set(stored_intervals.metric) == set(METRICS)
                    and stored_intervals.attempted_replicates.eq(2000).all(),
                    f"Invalid saved bootstrap interval table: {run_id}")
            interval_jobs.append((run, test, stored_intervals))
        completed.append(dict(**result, inference_revision=revision,
                              train_images=len(parts["train"]), tuning_images=len(tuning),
                              test_images=len(test), verified=True))
        print(f"Verified {run_id}: partitions, hashes, threshold and metrics", flush=True)
    # No output is written before every input run has passed the basic checks.
    output.mkdir(parents=True, exist_ok=True)
    if args.bootstrap:
        for run, test, stored_intervals in interval_jobs:
            run_id = run["run_id"]
            print(f"Recomputing {args.bootstrap} bootstrap attempts: {run_id}", flush=True)
            intervals = original["cluster_bootstrap"](
                test, replicates=args.bootstrap, seed=run["seed"]
            )
            if args.bootstrap == 2000:
                compare_table(intervals, stored_intervals, f"Bootstrap intervals: {run_id}")
            tables[run_id]["bootstrap_intervals.csv"] = intervals
    for run_id, per_run in tables.items():
        target = output / "runs" / run_id
        target.mkdir(parents=True)
        for name, table in per_run.items():
            table.to_csv(target / name, index=False, lineterminator="\n")
    pd.DataFrame(completed).to_csv(output / "verified_run_results.csv", index=False)
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "scipy")}
    report = dict(
        verified=True, verified_runs=len(completed), recorded_artifacts_verified=artifact_count,
        source_sha256=sha256(args.source), evidence_directory=str(evidence),
        dataset_images_in_manifest=len(manifest), partition_reproduction=True,
        tuning_thresholds_reproduced=True, metrics_reproduced=True,
        threshold_sensitivity_reproduced=True, bootstrap_attempts=args.bootstrap,
        bootstrap_runs_recomputed=len(interval_jobs) if args.bootstrap else 0,
        bootstrap_matches_saved=(True if args.bootstrap == 2000 else None),
        model_weights_loaded=False, tensorflow_imported="tensorflow" in sys.modules,
        dataset_image_bytes_reverified=False, python=platform.python_version(),
        packages=versions, original_protocol_software=protocol["software"],
        limitations=[
            "Hashes establish consistency with included records, not independent provenance.",
            "Model weights are excluded: recorded weight hashes are compared, but weights and predictions are not regenerated.",
            "Dataset images are excluded: manifest hashes are checked, not the original image bytes.",
            "Groups derive from filenames and exact duplicates; patient identities have not been independently verified.",
            "Five seeds are repeated holdouts, not five-fold cross-validation.",
            "Bootstrap intervals are conditional on the recorded grouping assumption.",
        ],
    )
    (output / "verification_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"PASS: {len(completed)} runs and {artifact_count} recorded artifacts. Output: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PACKAGE / "src/pneumonia_analysis.py")
    parser.add_argument("--evidence", type=Path, default=PACKAGE / "evidence")
    parser.add_argument("--output", required=True, type=Path,
                        help="New or empty directory outside the evidence tree")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="ATTEMPTS",
                        help="Optional attempts for the three published interval runs; use 2000 for exact comparison")
    args = parser.parse_args(argv)
    if args.bootstrap < 0 or 0 < args.bootstrap < 2:
        parser.error("--bootstrap must be 0 (off), or at least 2; publication comparison uses 2000")
    for name in ("source", "evidence", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    try:
        reproduce(args)
    except (OSError, ValueError, KeyError, AssertionError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")


if __name__ == "__main__":
    main()
