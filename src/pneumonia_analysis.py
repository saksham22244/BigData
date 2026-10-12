# %% [markdown]
# # Pneumonia classification: analysis and figure export
# 
# This notebook contains the 16-run experiment and the code used to prepare its tables and figures.
# The default setting, `MODE = "final_polish"`, reads the completed results from Google Drive and exports
# the seven manuscript figures. It does not train models, generate predictions or repeat bootstrap calculations.
# 
# To run it in Colab:
# 
# 1. Select a CPU runtime: **Runtime → Change runtime type → Hardware accelerator: None**.
# 2. Check the dataset and results paths in the configuration cell, then select **Runtime → Run all**.
# 3. Connect the Drive account containing the saved runs. Download `FINAL_FIGURES_AND_TABLES.zip` from the folder
#    printed at the end.
# 
# The notebook is self-contained. Each export uses a new dated folder; the saved analysis files remain in place.
# If the saved evidence is incomplete or inconsistent, the export stops.
# 
# Evaluation groups are derived from filenames. Patient identities have not been independently verified.


# %% [markdown]
# ## 1. Configuration
# 
# `final_polish` exports the completed study. `resume_analysis` enables the training and recovery workflow and
# requires the recorded software environment and a compatible GPU runtime.
# 
# The experiment uses five seeds—17, 29, 42, 71 and 101—for a total of 16 model runs. Keep `final_polish` selected
# when working with the completed results.


# %%
# Export the completed results on CPU.
MODE = "final_polish"
# Use "resume_analysis" only to resume training or evaluation.

CONFIG = {
    "MOUNT_GOOGLE_DRIVE": True,
    "DATASET_SOURCE": "/content/drive/MyDrive/Big Data/ChestXRay2017/chest_xray",
    "OUTPUT_ROOT": "/content/drive/MyDrive/Big Data/ChestXRay2017/pneumonia_reviewer_complete_v3",
    "SHOW_PREVIEW": True,
    # Settings for resume_analysis.
    "RESUME_EXISTING_EXPERIMENT": True,
    "LOCAL_DATASET": "/content/reviewer_v3_chest_xray",
    "LEGACY_OUTPUT_ROOTS": [
        "/content/drive/MyDrive/Big Data/ChestXRay2017/pneumonia_submission_revision_filename_group_v2/results"
    ],
    "REQUIRE_GPU": True,  # Used only in resume_analysis.
    "BOOTSTRAP_REPLICATES": 2000,
    "MAX_NEW_RUNS": None,
}
IMPLEMENTATION_SHA256 = (
    "311fdd4442a623add59a8b23d14ff56b85fa278f03117ab70d36e15c955b0e36"
)
if MODE not in {"final_polish", "resume_analysis"}:
    raise ValueError("MODE must be final_polish or resume_analysis.")
_MODE_AT_START = MODE
_ANALYSIS_SETUP_READY = False
_ANALYSIS_SUCCEEDED = False
print("Selected mode:", MODE)
if MODE == "final_polish":
    print(
        "Using saved results. Training, model inference and bootstrap computation are disabled."
    )
else:
    print(
        "Full recovery enabled: compatible completed runs are verified; missing work may need a GPU."
    )


# %% [markdown]
# ## 2. Analysis and recovery
# 
# This section runs only when `MODE = "resume_analysis"`. It includes dataset auditing, split construction,
# training, checkpoint recovery and evaluation. Compatibility checks cover the dataset, partitions, settings
# and software versions before saved work is reused. Restoring packages may require a session restart.
# 
# Task 1 selects its checkpoint by tuning loss. Both controlled Task 2 designs select the checkpoint with the
# highest tuning AUROC during fine-tuning. The filename-grouped seed-42 models supply the primary figures and
# confidence intervals. The other seeds provide descriptive repeated-holdout summaries.


# %% [markdown]
# ### 2.1. Drive connection


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    DRIVE_BOOTSTRAP_SHA256 = (
        "251bbd4853c4f610d0aa22b979dc00aba2164be1988b9060bb4308c30351f318"
    )
    """Connect Drive while preserving existing mount contents."""
    import os
    import json
    from pathlib import Path, PurePosixPath

    def _pneumonia_active_drive(path):
        path = Path(path)
        try:
            return (
                not path.is_symlink()
                and os.path.ismount(path)
                and ((path / "MyDrive").is_dir() or (path / "My Drive").is_dir())
            )
        except OSError as exc:
            raise RuntimeError(
                "A Drive mount is not responding. Preserve files and reconnect the runtime before continuing."
            ) from exc

    def _pneumonia_empty_mount(path):
        path = Path(path)
        if path.is_symlink():
            return False
        if not path.exists():
            return True
        return path.is_dir() and next(path.iterdir(), None) is None

    def _pneumonia_rebase_path(value, old_root, new_root):
        path, before, after = [
            PurePosixPath(str(item).replace("\\", "/"))
            for item in [value, old_root, new_root]
        ]
        try:
            relative = path.relative_to(before)
        except ValueError:
            return value
        if ".." in relative.parts:
            raise ValueError(
                "Drive paths must not escape the configured mount directory."
            )
        return str(after / relative)

    def ensure_pneumonia_drive(config, mount_callable=None, active_check=None):
        """Rebase only the mount prefix, preserving the exact MyDrive folder names."""
        requested = Path(config.get("DRIVE_MOUNTPOINT", "/content/drive"))
        if not requested.is_absolute():
            raise ValueError("DRIVE_MOUNTPOINT must be an absolute path.")
        active_check = active_check or _pneumonia_active_drive
        alternatives = [
            requested.parent / ("pneumonia_drive" + (f"_{i}" if i else ""))
            for i in range(16)
        ]
        candidates = list(dict.fromkeys([requested, *alternatives]))
        # Reuse active mounts: mounting again can stop an existing Drive process.
        # A new mount point must be an empty directory.
        target = next((path for path in candidates if active_check(path)), None)
        reused = target is not None
        if target is None:
            target = next(
                (path for path in candidates if _pneumonia_empty_mount(path)), None
            )
            if target is None:
                raise RuntimeError(
                    "No empty Drive mount location was found. Existing files were left untouched."
                )
            if not target.parent.is_dir():
                raise RuntimeError(
                    "The parent of the selected Drive mount does not exist."
                )
            if mount_callable is None:
                try:
                    from google.colab import drive
                except ImportError as exc:
                    raise RuntimeError(
                        "This preset uses Colab. For local Python, disable Drive mounting and set local paths."
                    ) from exc
                mount_callable = drive.mount
            print("Connecting Google Drive at:", target)
            if target != requested:
                print(
                    "Keeping existing files at",
                    requested,
                    "untouched; using an empty mount location.",
                )
            mount_callable(str(target))
            if not active_check(target):
                raise RuntimeError(
                    "Google Drive was not verified as mounted. No output folder or cache was created."
                )
        else:
            print("Reusing connected Google Drive at:", target)
        updated = dict(config)
        for key in ["DATASET_SOURCE", "OUTPUT_ROOT"]:
            updated[key] = _pneumonia_rebase_path(config[key], requested, target)
        updated["LEGACY_OUTPUT_ROOTS"] = [
            _pneumonia_rebase_path(path, requested, target)
            for path in config.get("LEGACY_OUTPUT_ROOTS", [])
        ]
        updated["DRIVE_MOUNTPOINT"] = str(target)
        # The analysis runner also calls mount(); authentication is complete here.
        # Skip the duplicate mount call.
        updated["MOUNT_GOOGLE_DRIVE"] = False
        updated["_DRIVE_MOUNT_GUARD"] = dict(
            version="V3.1.1",
            requested_mount=str(requested),
            active_mount=str(target),
            reused_active_mount=reused,
            local_conflicting_files_preserved=True,
        )
        return updated

    def initialize_paths(config):
        staged = dict(config)
        if staged.get("MOUNT_GOOGLE_DRIVE"):
            staged = ensure_pneumonia_drive(staged)
        elif staged.get("_DRIVE_MOUNT_GUARD"):
            if not _pneumonia_active_drive(
                staged["_DRIVE_MOUNT_GUARD"]["active_mount"]
            ):
                raise RuntimeError(
                    "The previously connected Drive is no longer mounted. Reconnect and Run all; no output files were created."
                )
        if not Path(staged["DATASET_SOURCE"]).is_dir():
            raise FileNotFoundError(
                "Dataset not found after connecting storage: "
                + staged["DATASET_SOURCE"]
                + ". Check the same Drive account and dataset path; existing checkpoints were not changed."
            )
        root = Path(staged["OUTPUT_ROOT"])
        existing_output = root.is_dir()
        root.mkdir(parents=True, exist_ok=True)
        os.environ["KERAS_HOME"] = str(root / "keras_cache")
        os.environ["MPLCONFIGDIR"] = str(
            Path(staged["LOCAL_DATASET"]).parent / "reviewer_v3_plot_cache"
        )
        if staged.get("_DRIVE_MOUNT_GUARD"):
            record = dict(
                staged["_DRIVE_MOUNT_GUARD"],
                bootstrap_source_sha256=DRIVE_BOOTSTRAP_SHA256,
                dataset_source=staged["DATASET_SOURCE"],
                output_root=staged["OUTPUT_ROOT"],
                training_and_inference_changed=False,
                existing_output_folder_found=existing_output,
            )
            temp = root / "drive_mount_status.json.tmp"
            temp.write_text(json.dumps(record, indent=2), encoding="utf-8")
            temp.replace(root / "drive_mount_status.json")
        config.update(staged)
        print("Dataset:", config["DATASET_SOURCE"])
        print("Existing experiment output folder:", config["OUTPUT_ROOT"])
        print("Output folder already existed:", existing_output)

    initialize_paths(CONFIG)


# %% [markdown]
# ### 2.2. Recorded software environment


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    RUNTIME_RESTORE_SHA256 = (
        "b755377769ecd00771ec4d80b08a017f86e7ecd3cf983bb3a928debbb7c88073"
    )
    """Restore the recorded package versions before importing the analysis libraries."""
    import hashlib
    import importlib.metadata
    import json
    import platform
    import re
    import subprocess
    import sys
    import time
    import uuid
    from pathlib import Path

    RECORDED_PACKAGES = (
        "tensorflow",
        "keras",
        "numpy",
        "pandas",
        "scikit-learn",
        "Pillow",
        "scipy",
    )
    RUNTIME_HELP_URL = (
        "https://research.google.com/colaboratory/runtime-version-faq.html"
    )
    _RESTART_ATTRIBUTE = "_pneumonia_v312_packages_changed_restart_required"

    def runtime_package_versions():
        versions = {}
        for name in RECORDED_PACKAGES:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        versions["python"] = platform.python_version()
        return versions

    def _runtime_event(root, event):
        directory = Path(root) / "runtime_recovery"
        directory.mkdir(parents=True, exist_ok=True)
        event = dict(
            event,
            recorded_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            training_protocol_changed=False,
            checkpoints_changed=False,
            bootstrap_sha256=globals().get("RUNTIME_RESTORE_SHA256", "development"),
        )
        target = directory / (
            time.strftime("%Y%m%dT%H%M%S", time.gmtime())
            + "_"
            + uuid.uuid4().hex[:12]
            + ".json"
        )
        target.write_text(json.dumps(event, indent=2, sort_keys=True), encoding="utf-8")
        return target

    def _recorded_runtime(config):
        root = Path(config["OUTPUT_ROOT"])
        protocol_path = root / "protocol.json"
        if not protocol_path.is_file():
            # A missing protocol must not start a replacement experiment.
            if config.get("RESUME_EXISTING_EXPERIMENT", True) or any(
                (root / "runs").glob("*")
            ):
                raise RuntimeError(
                    "Saved protocol.json is missing. Check the SAME Drive account and OUTPUT_ROOT. "
                    "This resume notebook will not start a replacement experiment or overwrite saved runs."
                )
            return None, None
        try:
            raw = protocol_path.read_bytes()
            protocol = json.loads(raw)
            if not isinstance(protocol, dict):
                raise ValueError("Expected a protocol object")
            recorded = protocol["software"]
            if not isinstance(recorded, dict):
                raise ValueError("Expected a recorded software object")
            if set(recorded) != {*RECORDED_PACKAGES, "python"}:
                raise ValueError("Unexpected or missing recorded software fields")
            for name, version in recorded.items():
                if not isinstance(version, str) or not re.fullmatch(
                    r"[0-9][0-9A-Za-z.!+_-]*", version
                ):
                    raise ValueError("Invalid recorded version for " + name)
            if not re.fullmatch(r"\d+\.\d+\.\d+", recorded["python"]):
                raise ValueError("Expected an exact Python version")
            if protocol.get("version") != "reviewer_complete_v3":
                raise ValueError("Not a reviewer_complete_v3 experiment")
            if protocol.get("implementation_sha256") != IMPLEMENTATION_SHA256:
                raise ValueError(
                    "Saved implementation is incompatible with this notebook"
                )
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError(
                "Cannot safely read the existing protocol; it was left unchanged: "
                + str(exc)
            ) from exc
        return recorded, hashlib.sha256(raw).hexdigest()

    def ensure_recorded_runtime(config, allow_install=False):
        """Restore runtime packages and require a restart after any installation attempt."""
        if getattr(sys, _RESTART_ATTRIBUTE, False):
            raise RuntimeError(
                "Packages were changed in this session. Choose Runtime > Restart session, "
                "then Run all again. Do not delete the runtime or the Drive results."
            )
        root = Path(config["OUTPUT_ROOT"])
        recorded, digest = _recorded_runtime(config)
        if recorded is None:
            return {"status": "new_experiment_explicitly_allowed"}
        actual = runtime_package_versions()
        mismatches = {
            name: {"recorded": version, "current": actual.get(name)}
            for name, version in recorded.items()
            if actual.get(name) != version
        }
        if recorded["python"] != actual["python"]:
            _runtime_event(
                root,
                dict(
                    status="python_runtime_mismatch",
                    protocol_sha256=digest,
                    differences=mismatches,
                ),
            )
            hint = {
                "3.12.13": "2026.07",
                "3.12.12": "2026.01",
                "3.11.13": "2025.07",
            }.get(recorded["python"])
            suggestion = (
                (
                    " A listed candidate is "
                    + hint
                    + "; verify its Python version after connecting."
                )
                if hint
                else ""
            )
            raise RuntimeError(
                "Python differs: saved "
                + recorded["python"]
                + ", current "
                + str(actual["python"])
                + ". "
                "In Runtime > Change runtime type > Runtime Version, choose a runtime with the saved Python "
                "version and a GPU, then Run all. pip cannot replace the notebook interpreter."
                + suggestion
                + " Current choices: "
                + RUNTIME_HELP_URL
                + " Existing models and protocol.json were left unchanged."
            )
        if mismatches:
            print("Restoring the environment recorded with your saved models:")
            for name, values in mismatches.items():
                print(" ", name, ":", values["current"], "->", values["recorded"])
            if not allow_install:
                raise RuntimeError(
                    "Software differs from protocol.json. Run the environment-restoration cell before "
                    "the analysis cells. No training has started."
                )
            pins = [name + "==" + recorded[name] for name in RECORDED_PACKAGES]
            directory = root / "runtime_recovery"
            directory.mkdir(parents=True, exist_ok=True)
            requirements = directory / ("requirements_" + digest[:16] + ".txt")
            requirements.write_text("\n".join(pins) + "\n", encoding="utf-8")
            log_path = directory / (
                "pip_"
                + time.strftime("%Y%m%dT%H%M%S", time.gmtime())
                + "_"
                + uuid.uuid4().hex[:8]
                + ".log"
            )
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--upgrade",
                "--only-binary=:all:",
                *pins,
            ]
            _runtime_event(
                root,
                dict(
                    status="restore_started",
                    protocol_sha256=digest,
                    differences=mismatches,
                    requirements_file=requirements.name,
                    pip_log=log_path.name,
                ),
            )
            # A failed installation can still change packages, so mark the session first.
            setattr(sys, _RESTART_ATTRIBUTE, True)
            print(
                "Installing the seven recorded package versions together. This can take several minutes."
            )
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                installed = runtime_package_versions()
                success = result.returncode == 0 and installed == recorded
                _runtime_event(
                    root,
                    dict(
                        status=(
                            "restart_required"
                            if success
                            else "install_failed_restart_required"
                        ),
                        protocol_sha256=digest,
                        installed=installed,
                        returncode=result.returncode,
                        pip_log=log_path.name,
                    ),
                )
            except Exception as exc:
                _runtime_event(
                    root,
                    dict(
                        status="installer_exception_restart_required",
                        protocol_sha256=digest,
                        error=type(exc).__name__ + ": " + str(exc),
                        pip_log=log_path.name,
                    ),
                )
                raise RuntimeError(
                    "Environment restoration was interrupted. Restart the session and Run all. "
                    "Models were not changed. Installer log: " + str(log_path)
                ) from exc
            if not success:
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-5000:]
                print(tail)
                raise RuntimeError(
                    "The recorded environment could not be fully installed. Restart the session; "
                    "if this repeats choose the matching past runtime. Do not edit protocol.json. "
                    "Installer log: " + str(log_path)
                )
            raise RuntimeError(
                "Recorded packages restored successfully. ONE RESTART IS REQUIRED: "
                "Runtime > Restart session, then Run all again. Saved models were not changed."
            )
        # Installed versions can match while older modules remain in memory.
        modules = {
            "tensorflow": "tensorflow",
            "keras": "keras",
            "numpy": "numpy",
            "pandas": "pandas",
            "scikit-learn": "sklearn",
            "Pillow": "PIL",
            "scipy": "scipy",
        }
        for package, module_name in modules.items():
            module = sys.modules.get(module_name)
            imported = (
                getattr(module, "__version__", None) if module is not None else None
            )
            if imported is not None and str(imported) != recorded[package]:
                raise RuntimeError(
                    "Loaded "
                    + module_name
                    + " differs from the installed recorded version. "
                    "Restart session and Run all before continuing."
                )
        event = dict(
            status="recorded_environment_verified",
            protocol_sha256=digest,
            software=actual,
        )
        _runtime_event(root, event)
        print(
            "Recorded Python and all seven package versions match. Saved training may be reused after evidence checks."
        )
        return event

    ensure_recorded_runtime(CONFIG, allow_install=True)


# %% [markdown]
# ### 2.3. Saved results


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    """Read saved prediction records and display existing results without modifying the analysis."""

    import csv
    import hashlib
    import html
    import json
    import math
    from pathlib import Path, PurePosixPath

    VIEWER_SEEDS = (17, 29, 42, 71, 101)
    VIEWER_TASK1_REVISION = "fixed_image_tta_v2_padded_physical_batch"

    def _viewer_runs():
        runs = [
            ("task2", design, seed)
            for seed in VIEWER_SEEDS
            for design in ("filename_grouped", "image_level")
        ]
        runs += [("task1", "filename_grouped", seed) for seed in VIEWER_SEEDS]
        runs += [("task2", "original_folders", 42)]
        return [
            dict(
                task=task,
                design=design,
                seed=seed,
                run_id=f"{task}_{design}_seed{seed}",
            )
            for task, design, seed in runs
        ]

    def _viewer_json(path):
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object: {Path(path).name}")
        return value

    def _viewer_csv(path):
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def _viewer_sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _viewer_object_sha(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _viewer_evidence(directory, run, task1_repair_required=False):
        """Check prediction coverage, signatures and file hashes without loading a model."""
        record = _viewer_json(directory / "evaluation_complete.json")
        training = _viewer_json(directory / "training_complete.json")
        config = _viewer_json(directory / "run_configuration.json")
        if (
            record.get("evaluation_complete") is not True
            or training.get("training_complete") is not True
        ):
            raise ValueError("Training/evaluation completion record is not true")
        digest = _viewer_object_sha(config)
        signature = record.get("signature", {})
        if training.get("signature_hash") != digest:
            raise ValueError("Recorded training configuration signature disagrees")
        inference_revision = (
            "single_pass_v1" if run["task"] == "task2" else "fixed_image_tta_v1"
        )
        if run["task"] == "task1":
            revised_digest = _viewer_object_sha(
                dict(config, evaluation_only_revision=VIEWER_TASK1_REVISION)
            )
            if signature.get("run_signature") == revised_digest:
                marker = _viewer_json(directory / "task1_inference_revision.json")
                if (
                    marker.get("inference_revision") != VIEWER_TASK1_REVISION
                    or marker.get("target_signature") != signature
                    or marker.get("training_changed") is not False
                ):
                    raise ValueError(
                        "Task 1 inference repair marker/signature is inconsistent"
                    )
                inference_revision = VIEWER_TASK1_REVISION
            elif signature.get("run_signature") == digest:
                if task1_repair_required:
                    raise ValueError(
                        "Old Task 1 predictions need V3.1 regeneration from saved trained weights"
                    )
            else:
                raise ValueError(
                    "Unrecognized Task 1 evaluation configuration signature"
                )
        elif signature.get("run_signature") != digest:
            raise ValueError("Recorded evaluation configuration signature disagrees")
        if signature.get("weights") != training.get(
            "selected_weights_sha256"
        ) or not signature.get("weights"):
            raise ValueError("Recorded selected-weight signatures disagree")
        if config.get("run") != run:
            raise ValueError("Run configuration does not match this scheduled run")
        assignments = _viewer_csv(directory / "split_assignments.csv")
        if not assignments or len({row["image_id"] for row in assignments}) != len(
            assignments
        ):
            raise ValueError("Missing or duplicate split assignments")
        positive = "PNEUMONIA" if run["task"] == "task1" else "VIRAL"
        threshold = float(record["threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid stored threshold")
        for partition in ("tuning", "test"):
            filename = f"{partition}_predictions.csv"
            if _viewer_sha(directory / filename) != record.get(filename):
                raise ValueError(f"Checksum mismatch: {filename}")
            predictions = _viewer_csv(directory / filename)
            expected = {
                row["image_id"]: row
                for row in assignments
                if row.get("partition") == partition
            }
            if (
                not expected
                or len(predictions) != len(expected)
                or {row["image_id"] for row in predictions} != set(expected)
            ):
                raise ValueError(f"Prediction coverage mismatch: {partition}")
            for row in predictions:
                original = expected[row["image_id"]]
                probability = float(row["probability"])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(f"Invalid probability: {partition}")
                if row["positive_class"] != positive or int(row["y_true"]) != int(
                    original["label"] == positive
                ):
                    raise ValueError(f"Class/label mismatch: {partition}")
                if any(
                    row[key] != original[key]
                    for key in ("label", "qualified_key", "raw_key", "split_group")
                ):
                    raise ValueError(f"Filename-group metadata mismatch: {partition}")
                if not math.isclose(float(row["threshold"]), threshold, abs_tol=1e-12):
                    raise ValueError(f"Threshold mismatch: {partition}")
                if int(row["predicted_index"]) != int(probability >= threshold):
                    raise ValueError(f"Prediction-label mismatch: {partition}")
        return record, inference_revision

    def inventory_saved_results(output_root):
        """Inspect the 16 scheduled runs and their saved prediction records.

        Checks cover file hashes, image coverage and recorded signatures.
        Model files and figure contents are not validated by this reader.
        The returned analysis status is the status recorded by the runner."""
        root = Path(output_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"Output folder not found: {root}. Mount Drive and check OUTPUT_ROOT."
            )
        result = dict(
            output_root=str(root),
            runs=[],
            metrics=[],
            diagnostics=[],
            saved_analysis_status=None,
        )
        repair_required = False
        patch_path = root / "reporting_patch.json"
        if patch_path.is_file():
            try:
                patch = _viewer_json(patch_path)
                repair_required = patch.get("revision") == "V3.1"
                if (
                    repair_required
                    and patch.get("task1_inference_revision") != VIEWER_TASK1_REVISION
                ):
                    result["diagnostics"].append(
                        "V3.1 reporting marker has an unexpected Task 1 inference revision; inspect it before using results."
                    )
            except (OSError, ValueError) as exc:
                result["diagnostics"].append(f"Cannot read reporting_patch.json: {exc}")
        status_path = root / "completion_status.json"
        if status_path.is_file():
            try:
                result["saved_analysis_status"] = _viewer_json(status_path)
            except (OSError, ValueError) as exc:
                result["diagnostics"].append(
                    f"Cannot read completion_status.json: {exc}"
                )
        for run in _viewer_runs():
            directory = root / "runs" / run["run_id"]
            primary = run["seed"] == 42 and run["design"] == "filename_grouped"
            figure_names = ["training_curves.png"]
            if primary or run["design"] == "original_folders":
                figure_names += ["roc_curve.png", "confusion_matrix.png"]
            if primary:
                figure_names += ["occlusion/occlusion_examples.csv"]
            row = dict(
                **run,
                training_record_present=(
                    directory / "training_complete.json"
                ).is_file(),
                evaluation_record_present=(
                    directory / "evaluation_complete.json"
                ).is_file(),
                prediction_evidence_verified=False,
                inference_revision="Not checked",
                missing_expected_figures="; ".join(
                    name for name in figure_names if not (directory / name).is_file()
                ),
                note="No evaluation record",
            )
            if row["evaluation_record_present"]:
                try:
                    record, inference_revision = _viewer_evidence(
                        directory, run, repair_required
                    )
                    row["prediction_evidence_verified"] = True
                    row["inference_revision"] = inference_revision
                    row["note"] = (
                        "Prediction files verified; model files not checked by viewer"
                    )
                    result["metrics"].append(
                        dict(
                            **run,
                            threshold=record["threshold"],
                            **record.get("metrics", {}),
                        )
                    )
                except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
                    row["note"] = f"Saved evidence needs inspection: {exc}"
            result["runs"].append(row)
        expected_dirs = {run["run_id"] for run in _viewer_runs()}
        if (root / "runs").is_dir():
            extras = sorted(
                path.name
                for path in (root / "runs").iterdir()
                if path.is_dir() and path.name not in expected_dirs
            )
            if extras:
                result["diagnostics"].append(
                    "Unrecognized run folders (not counted): " + ", ".join(extras)
                )
        return result

    def _viewer_html_table(rows, columns=None, limit=20):
        if not rows:
            return "<p>No rows available.</p>"
        columns = list(columns or rows[0].keys())
        escape = lambda value: html.escape(str(value))
        heading = "".join(f"<th>{escape(column)}</th>" for column in columns)
        body = "".join(
            "<tr>"
            + "".join(f"<td>{escape(row.get(column, ''))}</td>" for column in columns)
            + "</tr>"
            for row in rows[:limit]
        )
        suffix = (
            f"<p>Showing {min(limit, len(rows))} of {len(rows)} rows.</p>"
            if len(rows) > limit
            else ""
        )
        return f'<div style="overflow-x:auto"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>{suffix}'

    def _viewer_sample_path(dataset_root, image_id):
        """Resolve a manifest image path within the dataset directory."""
        relative = PurePosixPath(str(image_id).replace("\\", "/"))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or ":" in relative.parts[0]
        ):
            raise ValueError("Unsafe image_id path")
        root = Path(dataset_root).expanduser().resolve()
        path = root.joinpath(*relative.parts).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Image path is outside the dataset folder")
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("Unsupported sample image format")
        return path

    def show_saved_results(output_root, dataset_source=None, max_images=20):
        """Display saved plots and tables, and return the result inventory.

        Optional radiographs illustrate supplied dataset categories. They are
        not occlusion results or evidence of independently verified identities."""
        if (
            not isinstance(max_images, int)
            or isinstance(max_images, bool)
            or max_images < 0
        ):
            raise ValueError("max_images must be a nonnegative integer")
        from IPython.display import HTML, Image, FileLink, display

        inventory = inventory_saved_results(output_root)
        root = Path(inventory["output_root"])

        def message(text, level=None):
            tag = f"h{level}" if level else "p"
            display(HTML(f"<{tag}>{html.escape(str(text))}</{tag}>"))

        def show_table(path, limit=12):
            try:
                rows = _viewer_csv(path)
                message(path.relative_to(root).as_posix(), 4)
                display(HTML(_viewer_html_table(rows, limit=limit)))
                display(FileLink(str(path)))
            except (OSError, ValueError, csv.Error) as exc:
                message(f"Cannot display {path.name}: {exc}")

        verified = sum(run["prediction_evidence_verified"] for run in inventory["runs"])
        message("Saved V3 results — read-only gallery", 2)
        message(f"Output folder: {root}")
        message(
            f"{verified}/16 runs passed this viewer's saved-prediction evidence checks. This is not final analysis or submission approval."
        )
        message(
            "There are five seed values and 16 task/design/seed models. No training is started and no experiment files are changed."
        )
        status = inventory["saved_analysis_status"]
        if status:
            message(
                f"Runner's saved phase: {status.get('phase', 'not recorded')}; analysis_complete: {status.get('analysis_complete', 'not recorded')}. This viewer does not independently certify that status."
            )
            if status.get("error"):
                message("Saved runner error: " + str(status["error"]))
        message(
            "The runner's session counter can restart at zero while saved per-run files remain intact. This inventory checks each scheduled run's own saved evidence, rather than treating a session counter as the total completed history."
        )
        message("Run inventory", 3)
        display(
            HTML(
                _viewer_html_table(
                    inventory["runs"],
                    columns=[
                        "run_id",
                        "training_record_present",
                        "evaluation_record_present",
                        "prediction_evidence_verified",
                        "inference_revision",
                        "missing_expected_figures",
                        "note",
                    ],
                    limit=16,
                )
            )
        )
        message("Metrics from verified saved prediction records", 3)
        message(
            "Values below are stored metrics, not recomputed by this viewer. Compare tasks and repeated splits descriptively."
        )
        display(HTML(_viewer_html_table(inventory["metrics"], limit=16)))

        message("Saved figures", 3)
        message(
            "V3 writes figures to Drive instead of displaying them during training. Training curves are per run. ROC/confusion figures are for grouped seed 42 and the original-folder comparator; occlusion examples are for grouped seed 42. Figure 2 and final summaries are created after all 16 analyses finish."
        )
        preferred_runs = [
            "task1_filename_grouped_seed42",
            "task2_filename_grouped_seed42",
            "task2_original_folders_seed42",
        ]
        ordered = [root / "Figure_2.png", root / "figures" / "dataset_examples.png"]
        for run_id in preferred_runs:
            directory = root / "runs" / run_id
            ordered += [
                directory / name
                for name in (
                    "roc_curve.png",
                    "confusion_matrix.png",
                    "training_curves.png",
                )
            ]
            ordered += sorted((directory / "occlusion").glob("example_*.png"))
        ordered += [
            root / "runs" / run["run_id"] / "training_curves.png"
            for run in _viewer_runs()
            if run["run_id"] not in preferred_runs
        ]
        figures = [path for path in ordered if path.is_file()]
        message(
            f"{len(figures)} saved figure images found; showing up to {max_images}."
        )
        for path in figures[:max_images]:
            message(path.relative_to(root).as_posix(), 4)
            display(Image(data=path.read_bytes(), format="png", width=950))
            display(FileLink(str(path)))
            if path.with_suffix(".pdf").is_file():
                display(FileLink(str(path.with_suffix(".pdf"))))
        if not figures:
            message(
                "No saved PNG figures found. Check the output folder; inspect the error/status above. A metric row alone does not prove figure generation finished."
            )
        if len(figures) > max_images:
            message(
                "Increase max_images to display more training curves. All figure paths:"
            )
            for path in figures[max_images:]:
                display(FileLink(str(path)))

        message("Saved tables", 3)
        table_names = [
            "all_run_results.csv",
            "repeated_split_summary.csv",
            "task2_paired_seed_differences.csv",
            "split_counts_all_runs.csv",
        ]
        for name in table_names:
            if (root / name).is_file():
                show_table(root / name, limit=16)
        for run_id in preferred_runs:
            for name in (
                "threshold_sensitivity.csv",
                "bootstrap_intervals.csv",
                "occlusion/occlusion_examples.csv",
            ):
                path = root / "runs" / run_id / name
                if path.is_file():
                    show_table(path)
        audit_files = sorted((root / "audit").glob("*.csv"))
        if audit_files:
            message("Audit tables (links; full manifests can be large)", 3)
            for path in audit_files:
                display(FileLink(str(path)))

        if dataset_source is not None:
            message("Optional example chest radiographs", 3)
            message(
                "Deterministic filename examples from supplied NORMAL, BACTERIAL and VIRAL categories. These are sample images, not representative performance estimates or verified etiologies."
            )
            candidates = {}
            for run in _viewer_runs():
                path = root / "runs" / run["run_id"] / "split_assignments.csv"
                if path.is_file():
                    try:
                        for row in _viewer_csv(path):
                            category = row.get("label_task2") or row.get("label")
                            if category in ("NORMAL", "BACTERIAL", "VIRAL"):
                                image_id = row["image_id"]
                                if (
                                    category not in candidates
                                    or image_id < candidates[category]
                                ):
                                    candidates[category] = image_id
                    except (OSError, KeyError, csv.Error) as exc:
                        message(f"Cannot read sample-image mapping: {exc}")
            for category in ("NORMAL", "BACTERIAL", "VIRAL"):
                if category not in candidates:
                    message(f"No saved filename mapping found for {category}.")
                    continue
                image_id = candidates[category]
                try:
                    path = _viewer_sample_path(dataset_source, image_id)
                    message(f"{category}: {image_id}", 4)
                    display(
                        Image(
                            data=path.read_bytes(),
                            format="png" if path.suffix.lower() == ".png" else "jpeg",
                            width=450,
                        )
                    )
                except (OSError, ValueError) as exc:
                    message(
                        f"Sample unavailable: {exc}. Set dataset_source to the original folder containing train/val/test."
                    )
        for diagnostic in inventory["diagnostics"]:
            message(diagnostic)
        message(
            "Missing plots require completion or repair of the reporting stage. This gallery intentionally does not retrain, recalculate occlusion, bypass an error, or label an incomplete analysis complete."
        )
        return inventory


# %% [markdown]
# ### 2.4. Dataset audit and split construction


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    """Audit the dataset and construct filename groups with exact-duplicate constraints."""
    import hashlib
    import json
    import re
    import shutil
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split

    def write_json(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temp.replace(path)

    def write_csv(path, frame):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(path.name + ".tmp")
        frame.to_csv(temp, index=False)
        temp.replace(path)

    def file_hash(path):
        h = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def object_hash(value):
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def frame_hash(frame, columns):
        # Preserve row order because seeded image-level splits depend on it.
        return hashlib.sha256(
            frame[columns].fillna("").to_csv(index=False, lineterminator="\n").encode()
        ).hexdigest()

    def parse_filename(filename):
        stem = Path(filename).stem.lower()
        match = re.match(r"^(person\d+)_(bacteria|virus)_", stem)
        if match:
            return (
                match[1] + "_" + match[2],
                match[1],
                "person_number_plus_subtype",
                False,
            )
        for pattern, rule in [
            (r"^(normal\d+-im-\d+)", "normal_number_im"),
            (r"^(normal\d+-\d+)", "normal_number"),
            (r"^(im-\d+-\d+)", "im_number"),
        ]:
            match = re.match(pattern, stem)
            if match:
                return match[1], match[1], rule, False
        prefix = stem.split("_")[0]
        return prefix, prefix, "unverified_prefix_fallback", True

    def attach_duplicate_groups(manifest, audit_dir):
        """Join qualified keys linked by identical decoded images, retaining every row."""
        audit_dir = Path(audit_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        out = manifest.copy()
        duplicates = out[out.duplicated("pixel_sha256", keep=False)].copy()
        write_csv(
            audit_dir / "exact_pixel_duplicates.csv",
            duplicates.drop(columns="filepath", errors="ignore"),
        )
        # Stop on contradictory labels rather than choosing one for an exact copy.
        conflicts = out.groupby("pixel_sha256")[
            ["label_task1", "label_task2"]
        ].nunique()
        bad_hashes = conflicts.index[(conflicts > 1).any(axis=1)]
        bad = out[out.pixel_sha256.isin(bad_hashes)]
        write_csv(
            audit_dir / "conflicting_exact_image_labels.csv",
            bad.drop(columns="filepath", errors="ignore"),
        )
        if len(bad):
            raise ValueError(
                "Exact duplicate images have contradictory labels. See conflicting_exact_image_labels.csv; resolve against source records before training."
            )
        parent = {key: key for key in sorted(out.qualified_key.unique())}

        def find(key):
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        for _, group in out.groupby("pixel_sha256", sort=True):
            keys = sorted(group.qualified_key.unique())
            for key in keys[1:]:
                a, b = sorted([find(keys[0]), find(key)])
                parent[b] = a
        out["split_group"] = out.qualified_key.map(
            lambda key: "fg_" + hashlib.sha256(find(key).encode()).hexdigest()[:24]
        )
        conflict_groups = out.groupby("split_group")[
            ["label_task1", "label_task2"]
        ].nunique()
        bad_groups = conflict_groups.index[(conflict_groups > 1).any(axis=1)]
        write_csv(
            audit_dir / "conflicting_group_labels.csv",
            out[out.split_group.isin(bad_groups)].drop(
                columns="filepath", errors="ignore"
            ),
        )
        if len(bad_groups):
            raise ValueError(
                "A derived split group contains conflicting labels; inspect conflicting_group_labels.csv."
            )
        write_csv(
            audit_dir / "group_key_mapping.csv",
            out[
                ["image_id", "qualified_key", "raw_key", "pixel_sha256", "split_group"]
            ],
        )
        return out

    def prepare_manifest(source_root, local_root, audit_dir):
        source_root, local_root, audit_dir = map(
            Path, [source_root, local_root, audit_dir]
        )
        if not source_root.exists():
            raise FileNotFoundError(f"Configure DATASET_SOURCE: {source_root}")
        if source_root.resolve() == local_root.resolve():
            raise ValueError("Use different source and local dataset directories.")
        records, invalid, layout_roots = [], [], set()
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
            }:
                continue
            relative = path.relative_to(source_root)
            # Ignore macOS metadata files.
            if "__MACOSX" in relative.parts or path.name.startswith("._"):
                continue
            folder, source = path.parent.name.upper(), path.parent.parent.name.upper()
            if folder not in {"NORMAL", "PNEUMONIA"} or source not in {
                "TRAIN",
                "VAL",
                "TEST",
            }:
                invalid.append(
                    {
                        "image_id": relative.as_posix(),
                        "reason": "Unexpected folder layout",
                    }
                )
                continue
            layout_roots.add(tuple(relative.parts[:-3]))
            # Compare content; equal file sizes do not guarantee identical images.
            digest = file_hash(path)
            local = local_root / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() or file_hash(local) != digest:
                temp = local.with_name(local.name + ".copying")
                shutil.copy2(path, temp)
                if file_hash(temp) != digest:
                    raise IOError(f"Dataset copy verification failed: {relative}")
                temp.replace(local)
            try:
                with Image.open(local) as opened:
                    rgb = opened.convert("RGB")
                    pixel_digest = hashlib.sha256(
                        str(rgb.size).encode() + rgb.tobytes()
                    ).hexdigest()
            except Exception as exc:
                invalid.append({"image_id": relative.as_posix(), "reason": str(exc)})
                continue
            qualified, raw, rule, fallback = parse_filename(path.name)
            lower = path.name.lower()
            subtype = (
                "BACTERIAL"
                if "bacteria" in lower
                else ("VIRAL" if "virus" in lower else "")
            )
            if folder == "PNEUMONIA" and not subtype:
                invalid.append(
                    {
                        "image_id": relative.as_posix(),
                        "reason": "Pneumonia subtype cannot be parsed",
                    }
                )
                continue
            records.append(
                dict(
                    image_id=relative.as_posix(),
                    filepath=str(local),
                    filename=path.name,
                    source_split=source,
                    folder_label=folder,
                    label_task1=folder,
                    label_task2=subtype if folder == "PNEUMONIA" else "",
                    qualified_key=qualified,
                    raw_key=raw,
                    parse_rule=rule,
                    parse_fallback=fallback,
                    file_sha256=digest,
                    pixel_sha256=pixel_digest,
                )
            )
        write_csv(
            audit_dir / "invalid_images.csv",
            pd.DataFrame(invalid, columns=["image_id", "reason"]),
        )
        if invalid:
            raise ValueError(
                "Invalid/ambiguous images found; inspect invalid_images.csv. No silent exclusion."
            )
        if len(layout_roots) != 1:
            raise ValueError(
                "Multiple nested dataset roots were found. Point DATASET_SOURCE to exactly one train/val/test dataset."
            )
        if not records:
            raise ValueError("No eligible images found under DATASET_SOURCE.")
        manifest = pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)
        if manifest.image_id.duplicated().any():
            raise ValueError("Duplicate source image IDs.")
        manifest = attach_duplicate_groups(manifest, audit_dir)
        write_csv(audit_dir / "manifest.csv", manifest.drop(columns="filepath"))
        return manifest

    def task_frame(manifest, task):
        frame = manifest.copy()
        if task == "task2":
            frame = frame[frame.label_task2.isin(["BACTERIAL", "VIRAL"])].copy()
            frame["label"] = frame.label_task2
        elif task == "task1":
            frame["label"] = frame.label_task1
        else:
            raise ValueError(task)
        return frame.sort_values("image_id").reset_index(drop=True)

    def make_partitions(frame, design, seed):
        frame = frame.sort_values("image_id").reset_index(drop=True)

        def grouped_split(data, fraction, split_seed):
            counts = data.groupby("split_group").label.nunique()
            if (counts != 1).any():
                raise ValueError("Mixed labels in grouping units.")
            groups = data.groupby("split_group", as_index=False).label.first()
            a, b = train_test_split(
                groups,
                test_size=fraction,
                random_state=split_seed,
                stratify=groups.label,
            )
            return (
                data[data.split_group.isin(a.split_group)].copy(),
                data[data.split_group.isin(b.split_group)].copy(),
            )

        if design == "original_folders":
            training = frame[frame.source_split == "TRAIN"]
            test = frame[frame.source_split == "TEST"].copy()
            train, tuning = grouped_split(training, 0.15, seed)
        elif design == "filename_grouped":
            train, remaining = grouped_split(frame, 0.30, seed)
            tuning, test = grouped_split(remaining, 0.50, seed)
        elif design == "image_level":
            train, remaining = train_test_split(
                frame, test_size=0.30, random_state=seed, stratify=frame.label
            )
            tuning, test = train_test_split(
                remaining, test_size=0.50, random_state=seed, stratify=remaining.label
            )
        else:
            raise ValueError(design)
        parts = {
            name: part.sort_values("image_id").reset_index(drop=True)
            for name, part in [("train", train), ("tuning", tuning), ("test", test)]
        }
        validate_partitions(parts, design, frame)
        return parts

    def validate_partitions(parts, design, expected_frame):
        ids = [set(parts[name].image_id) for name in ["train", "tuning", "test"]]
        if any(ids[a] & ids[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
            raise ValueError("An image is in more than one partition.")
        expected = (
            expected_frame
            if design != "original_folders"
            else expected_frame[expected_frame.source_split.isin(["TRAIN", "TEST"])]
        )
        if set.union(*ids) != set(expected.image_id):
            raise ValueError("Split coverage differs from the eligible dataset.")
        for name, part in parts.items():
            if part.image_id.duplicated().any() or part.label.nunique() != 2:
                raise ValueError(
                    f"Invalid {name} partition: duplicate IDs or missing class."
                )
        if design == "filename_grouped":
            for column in ["qualified_key", "split_group", "pixel_sha256"]:
                sets = [
                    set(parts[name][column]) for name in ["train", "tuning", "test"]
                ]
                if any(sets[a] & sets[b] for a, b in [(0, 1), (0, 2), (1, 2)]):
                    raise ValueError(f"{column} crosses a grouped partition.")

    def partition_assignments(parts):
        return pd.concat(
            [p.assign(partition=name) for name, p in parts.items()], ignore_index=True
        )

    def original_overlap_table(manifest):
        rows = []
        for category in ["ALL", "NORMAL", "BACTERIAL", "VIRAL"]:
            data = (
                manifest
                if category == "ALL"
                else manifest[
                    (manifest.folder_label == category)
                    | (manifest.label_task2 == category)
                ]
            )
            train, test = (
                data[data.source_split == "TRAIN"],
                data[data.source_split == "TEST"],
            )
            for key in ["raw_key", "qualified_key", "split_group", "pixel_sha256"]:
                overlap = set(train[key]) & set(test[key])
                rows.append(
                    dict(
                        category=category,
                        key_rule=key,
                        train_groups=train[key].nunique(),
                        test_groups=test[key].nunique(),
                        overlap_groups=len(overlap),
                        percent_test_groups=100
                        * len(overlap)
                        / max(test[key].nunique(), 1),
                        test_group_denominator=test[key].nunique(),
                        affected_train_images=int(train[key].isin(overlap).sum()),
                        affected_test_images=int(test[key].isin(overlap).sum()),
                        affected_images_both_folders=int(
                            train[key].isin(overlap).sum()
                            + test[key].isin(overlap).sum()
                        ),
                        test_image_denominator=len(test),
                        percent_test_images=100
                        * int(test[key].isin(overlap).sum())
                        / max(len(test), 1),
                    )
                )
        return pd.DataFrame(rows)

    def export_data_audits(manifest, audit_dir):
        audit_dir = Path(audit_dir)
        write_csv(
            audit_dir / "original_folder_overlaps.csv", original_overlap_table(manifest)
        )
        rules = (
            manifest.groupby("parse_rule")
            .agg(images=("image_id", "size"), fallback=("parse_fallback", "max"))
            .reset_index()
        )
        rules["percent_images"] = 100 * rules.images / len(manifest)
        examples = (
            manifest.groupby("parse_rule")[["filename", "qualified_key", "raw_key"]]
            .first()
            .reset_index()
        )
        write_csv(
            audit_dir / "parsing_coverage.csv", rules.merge(examples, on="parse_rule")
        )
        group_rows = []
        for key in ["raw_key", "qualified_key", "split_group"]:
            per_group = (
                manifest.groupby(key)
                .agg(images=("image_id", "size"), subtypes=("label_task2", "nunique"))
                .reset_index()
            )
            write_csv(audit_dir / f"{key}_distribution.csv", per_group)
            group_rows.append(
                dict(
                    key_rule=key,
                    groups=len(per_group),
                    multi_image_groups=int((per_group.images > 1).sum()),
                    maximum_images=int(per_group.images.max()),
                    median_images=float(per_group.images.median()),
                    mixed_subtype_groups=int((per_group.subtypes > 1).sum()),
                )
            )
        write_csv(audit_dir / "group_summary.csv", pd.DataFrame(group_rows))
        collisions = []
        for a in ["BACTERIAL", "VIRAL"]:
            for b in ["BACTERIAL", "VIRAL"]:
                train = manifest[
                    (manifest.source_split == "TRAIN") & (manifest.label_task2 == a)
                ]
                test = manifest[
                    (manifest.source_split == "TEST") & (manifest.label_task2 == b)
                ]
                overlap = set(train.raw_key) & set(test.raw_key)
                collisions.append(
                    dict(
                        train_subtype=a,
                        test_subtype=b,
                        overlapping_raw_prefixes=len(overlap),
                        affected_train_images=int(train.raw_key.isin(overlap).sum()),
                        affected_test_images=int(test.raw_key.isin(overlap).sum()),
                        test_prefix_denominator=test.raw_key.nunique(),
                        percent_test_prefixes=100
                        * len(overlap)
                        / max(test.raw_key.nunique(), 1),
                    )
                )
        write_csv(
            audit_dir / "raw_cross_subtype_overlaps.csv", pd.DataFrame(collisions)
        )
        write_json(
            audit_dir / "grouping_interpretation.json",
            {
                "identity_verified": False,
                "fallback_images": int(manifest.parse_fallback.sum()),
                "image_level_fallback_for_grouped_design": False,
                "interpretation": "Subtype-qualified purity is partly imposed by construction. Raw-prefix collisions do not establish shared patients. Neither key independently verifies identity.",
                "duplicate_policy": "Keep all copies; link same-label exact decoded images into indivisible grouped split units. Conflicting labels stop the analysis. Original folder assignments are unchanged.",
            },
        )

    def export_partition_audit(parts, design, out_dir):
        out_dir = Path(out_dir)
        assignment = partition_assignments(parts)
        write_csv(
            out_dir / "split_assignments.csv", assignment.drop(columns="filepath")
        )
        rows = []
        for name, frame in parts.items():
            for label, subset in frame.groupby("label"):
                rows.append(
                    dict(
                        partition=name,
                        label=label,
                        images=len(subset),
                        filename_groups=subset.qualified_key.nunique(),
                        split_units=subset.split_group.nunique(),
                        design=design,
                    )
                )
        write_csv(out_dir / "split_counts.csv", pd.DataFrame(rows))
        overlaps = []
        for a, b in [("train", "tuning"), ("train", "test"), ("tuning", "test")]:
            for key in ["raw_key", "qualified_key", "split_group", "pixel_sha256"]:
                shared = set(parts[a][key]) & set(parts[b][key])
                overlaps.append(
                    dict(
                        first=a,
                        second=b,
                        key_rule=key,
                        overlapping_groups=len(shared),
                        second_group_denominator=parts[b][key].nunique(),
                        percent_second_groups=100
                        * len(shared)
                        / max(parts[b][key].nunique(), 1),
                        affected_first_images=int(parts[a][key].isin(shared).sum()),
                        affected_second_images=int(parts[b][key].isin(shared).sum()),
                    )
                )
        write_csv(out_dir / "partition_overlaps.csv", pd.DataFrame(overlaps))


# %% [markdown]
# ### 2.5. Checkpoints and interruption recovery


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    """Save Keras state with a verified previous checkpoint for recovery."""
    import time
    import uuid
    import tensorflow as tf

    def _snapshot_valid(directory, state, signature):
        if state.get("signature_hash") != object_hash(signature):
            raise ValueError(
                "Checkpoint protocol/data/split/software signature differs. Use a new experiment name; old files are preserved."
            )
        for key in ["model_file", "best_file"]:
            name = state[key]
            if Path(name).name != name:
                raise ValueError("Invalid checkpoint filename.")
            path = directory / name
            if not path.is_file() or file_hash(path) != state[key + "_sha256"]:
                raise IOError(f"Checkpoint checksum mismatch: {path.name}")
        return state

    def read_stage_state(directory, signature):
        directory = Path(directory)
        pointer = directory / "pointer.json"
        if not pointer.exists():
            # Only committed snapshots count as completed epochs.
            return None
        try:
            names = json.loads(pointer.read_text(encoding="utf-8"))
        except Exception as exc:
            raise IOError(
                "Checkpoint pointer is unreadable; preserve the directory for inspection."
            ) from exc
        failures = []
        for which in ["current", "previous"]:
            name = names.get(which)
            if not name:
                continue
            if Path(name).name != name:
                raise ValueError("Invalid snapshot filename in pointer.")
            try:
                state = json.loads((directory / name).read_text(encoding="utf-8"))
                _snapshot_valid(directory, state, signature)
                state["_loaded_snapshot"] = name
                state["_rolled_back"] = which == "previous"
                return state
            except ValueError as exc:
                if "signature differs" in str(exc):
                    raise
                failures.append(str(exc))
            except Exception as exc:
                failures.append(str(exc))
        raise IOError("Neither committed checkpoint verifies: " + "; ".join(failures))

    def _restore_best(model, directory, state):
        with np.load(Path(directory) / state["best_file"], allow_pickle=False) as saved:
            model.set_weights([saved[f"arr_{i}"] for i in range(len(saved.files))])

    def _commit_epoch(directory, model, best_weights, state):
        directory = Path(directory)
        token = uuid.uuid4().hex
        saved = {k: v for k, v in state.items() if not k.startswith("_")}
        saved["model_file"] = token + ".keras"
        saved["best_file"] = token + ".npz"
        model.save(directory / saved["model_file"])
        np.savez(directory / saved["best_file"], *best_weights)
        for key in ["model_file", "best_file"]:
            saved[key + "_sha256"] = file_hash(directory / saved[key])
        snapshot_name = token + ".json"
        write_json(directory / snapshot_name, saved)
        _snapshot_valid(directory, saved, saved["signature"])
        # Retain the recovered snapshot as the backup for the next commit.
        previous = state.get("_loaded_snapshot")
        write_json(
            directory / "pointer.json", {"current": snapshot_name, "previous": previous}
        )
        saved["_loaded_snapshot"] = snapshot_name
        saved["_rolled_back"] = False
        retained = {snapshot_name, saved["model_file"], saved["best_file"]}
        if previous:
            old = json.loads((directory / previous).read_text(encoding="utf-8"))
            retained.update([previous, old["model_file"], old["best_file"]])
        # Retire only UUID-named snapshots directly inside this checkpoint directory.
        for path in directory.iterdir():
            if (
                re.fullmatch(r"[0-9a-f]{32}\.(json|keras|npz)", path.name)
                and path.name not in retained
            ):
                path.unlink()
        return saved

    class PersistentStage(tf.keras.callbacks.Callback):
        def __init__(
            self,
            directory,
            state,
            monitor,
            mode,
            es_patience,
            es_min_delta,
            lr_patience,
            lr_factor,
            min_lr,
            test_interrupt_after=None,
        ):
            super().__init__()
            self.directory, self.state = Path(directory), state
            self.monitor = monitor
            self.early = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                mode=mode,
                patience=es_patience,
                min_delta=es_min_delta,
                restore_best_weights=True,
            )
            self.reduce = tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                mode=mode,
                patience=lr_patience,
                factor=lr_factor,
                min_lr=min_lr,
            )
            self.test_interrupt_after = test_interrupt_after

        def on_train_begin(self, logs=None):
            for key, callback in [("early", self.early), ("reduce", self.reduce)]:
                callback.set_model(self.model)
                callback.set_params(self.params)
                callback.on_train_begin(logs)
                if (
                    hasattr(callback, "_set_monitor_op")
                    and getattr(callback, "monitor_op", None) is None
                ):
                    callback._set_monitor_op()
                for attribute, value in self.state.get(key, {}).items():
                    setattr(callback, attribute, value)
            if self.state.get("best_file"):
                with np.load(
                    self.directory / self.state["best_file"], allow_pickle=False
                ) as saved:
                    self.early.best_weights = [
                        saved[f"arr_{i}"] for i in range(len(saved.files))
                    ]

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            if not np.isfinite(logs.get(self.monitor, np.nan)):
                raise ValueError(
                    "Missing/nonfinite validation monitor; epoch was not committed."
                )
            self.early.on_epoch_end(epoch, logs)
            self.reduce.on_epoch_end(epoch, logs)

            def capture(callback, attrs):
                values = {}
                for attr in attrs:
                    value = getattr(callback, attr, None)
                    if value is not None:
                        values[attr] = float(value) if attr == "best" else int(value)
                return values

            state = self.state.copy()
            state.update(
                epoch=epoch + 1,
                training_complete=bool(
                    self.model.stop_training or epoch + 1 >= state["epochs"]
                ),
                early=capture(
                    self.early, ["best", "wait", "stopped_epoch", "best_epoch"]
                ),
                reduce=capture(self.reduce, ["best", "wait", "cooldown_counter"]),
                optimizer_iterations=int(self.model.optimizer.iterations.numpy()),
                learning_rate=float(
                    tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                ),
            )
            state["history"] = state.get("history", []) + [
                dict(epoch=epoch + 1, **{k: float(v) for k, v in logs.items()})
            ]
            self.state = _commit_epoch(
                self.directory, self.model, self.early.best_weights, state
            )
            print(
                f"Saved to durable storage: {state['stage_name']} epoch {epoch+1}/{state['epochs']}"
            )
            if self.test_interrupt_after == epoch + 1:
                raise InterruptedError(
                    "Deliberate post-commit interruption for recovery smoke test."
                )

        def on_train_end(self, logs=None):
            self.early.on_train_end(logs)

    def run_stage(
        model_factory,
        train_data,
        val_data,
        class_weight,
        directory,
        signature,
        stage_name,
        epochs,
        monitor,
        mode,
        es_patience,
        es_min_delta=0.0,
        lr_patience=3,
        lr_factor=0.3,
        min_lr=1e-6,
        verbose=2,
        _test_interrupt_after=None,
    ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        actual_signature = dict(
            signature,
            stage=dict(
                name=stage_name,
                epochs=epochs,
                monitor=monitor,
                mode=mode,
                es_patience=es_patience,
                es_min_delta=es_min_delta,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                min_lr=min_lr,
            ),
        )
        state = read_stage_state(directory, actual_signature)
        if state is not None:
            model = tf.keras.models.load_model(directory / state["model_file"])
            if int(model.optimizer.iterations.numpy()) != state["optimizer_iterations"]:
                raise ValueError("Optimizer step count did not restore correctly.")
            events_path = directory / "recovery_events.json"
            events = json.loads(events_path.read_text()) if events_path.exists() else []
            if not state["training_complete"] or state["_rolled_back"]:
                events.append(
                    dict(
                        time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        completed_epoch=state["epoch"],
                        rolled_back=state["_rolled_back"],
                        stochastic_replay_exact=False,
                    )
                )
                write_json(events_path, events)
            if state["training_complete"]:
                _restore_best(model, directory, state)
                write_csv(directory / "history.csv", pd.DataFrame(state["history"]))
                return model, state
        else:
            model = model_factory()
            state = dict(
                signature=actual_signature,
                signature_hash=object_hash(actual_signature),
                stage_name=stage_name,
                epochs=epochs,
                epoch=0,
                training_complete=False,
                history=[],
            )
        callback = PersistentStage(
            directory,
            state,
            monitor,
            mode,
            es_patience,
            es_min_delta,
            lr_patience,
            lr_factor,
            min_lr,
            _test_interrupt_after,
        )
        model.fit(
            train_data,
            validation_data=val_data,
            initial_epoch=state["epoch"],
            epochs=epochs,
            class_weight=class_weight,
            callbacks=[callback],
            verbose=verbose,
        )
        state = callback.state
        if not state["training_complete"]:
            raise RuntimeError(
                "Training returned without a committed completion record."
            )
        _restore_best(model, directory, state)
        write_csv(directory / "history.csv", pd.DataFrame(state["history"]))
        return model, state

    def recovery_smoke_test(workdir):
        """Test optimizer state, callbacks, rollback and stage transitions using a small Keras model."""
        workdir = Path(workdir) / ("smoke_" + uuid.uuid4().hex)
        workdir.mkdir(parents=True)
        tf.keras.utils.set_random_seed(813)
        x = np.arange(32, dtype="float32").reshape(16, 2) / 32
        y = np.asarray([0, 1] * 8, dtype="float32")
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        dataset = (
            tf.data.Dataset.from_tensor_slices((x, y)).batch(4).with_options(options)
        )

        def factory():
            tf.keras.utils.set_random_seed(813)
            model = tf.keras.Sequential(
                [
                    tf.keras.Input((2,)),
                    tf.keras.layers.Dense(3, activation="tanh"),
                    tf.keras.layers.Dense(1, activation="sigmoid"),
                ]
            )
            # Tiny updates create a plateau for testing learning-rate reduction.
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-12),
                loss="binary_crossentropy",
                metrics=[tf.keras.metrics.AUC(name="auc")],
            )
            return model

        checks = []
        with tf.device("/CPU:0"):
            for monitor, mode in [("val_loss", "min"), ("val_auc", "max")]:
                directory = workdir / monitor
                kwargs = dict(
                    model_factory=factory,
                    train_data=dataset,
                    val_data=dataset,
                    class_weight=None,
                    directory=directory,
                    signature={"test": "native-v1", "tensorflow": tf.__version__},
                    stage_name="head",
                    epochs=4,
                    monitor=monitor,
                    mode=mode,
                    es_patience=8,
                    lr_patience=3,
                    min_lr=0.0,
                    verbose=0,
                )
                try:
                    run_stage(**kwargs, _test_interrupt_after=2)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError("Interruption was not triggered.")
                pointer = json.loads((directory / "pointer.json").read_text())
                halfway = json.loads((directory / pointer["current"]).read_text())
                assert halfway["epoch"] == 2 and halfway["optimizer_iterations"] == 8
                assert halfway["early"]["wait"] == 1 and halfway["reduce"]["wait"] == 1
                model, done = run_stage(**kwargs)
                assert (
                    done["training_complete"]
                    and done["epoch"] == 4
                    and done["optimizer_iterations"] == 16
                )
                assert done["early"]["wait"] == 3 and done["reduce"]["wait"] == 0
                assert np.isclose(done["learning_rate"], 3e-13, rtol=1e-5, atol=0.0)
                uninterrupted, reference = run_stage(
                    **dict(kwargs, directory=directory / "uninterrupted_reference")
                )
                for recovered_var, reference_var in zip(
                    model.optimizer.variables, uninterrupted.optimizer.variables
                ):
                    np.testing.assert_allclose(
                        recovered_var.numpy(),
                        reference_var.numpy(),
                        rtol=1e-6,
                        atol=1e-8,
                    )

                def must_not_build():
                    raise AssertionError("Completed stage was rebuilt.")

                same, skipped = run_stage(**dict(kwargs, model_factory=must_not_build))
                assert skipped["optimizer_iterations"] == 16
                np.testing.assert_allclose(model(x).numpy(), same(x).numpy(), atol=1e-7)
                try:
                    run_stage(**dict(kwargs, signature={"changed": True}))
                except ValueError:
                    pass
                else:
                    raise AssertionError("Changed signature was accepted.")
                # Corrupt the test snapshot and check recovery from the previous epoch.
                pointer = json.loads((directory / "pointer.json").read_text())
                current = json.loads((directory / pointer["current"]).read_text())
                (directory / current["model_file"]).write_bytes(
                    b"deliberately-corrupt-smoke-checkpoint"
                )
                rolled, final = run_stage(**kwargs)
                assert (
                    final["training_complete"] and final["optimizer_iterations"] == 16
                )
                assert any(
                    event["rolled_back"]
                    for event in json.loads(
                        (directory / "recovery_events.json").read_text()
                    )
                )

                # Check that the next stage starts with its own state and optimizer.
                def second_factory():
                    rolled.compile(
                        optimizer=tf.keras.optimizers.Adam(0.001),
                        loss="binary_crossentropy",
                        metrics=[tf.keras.metrics.AUC(name="auc")],
                    )
                    return rolled

                _, second = run_stage(
                    **dict(
                        kwargs,
                        model_factory=second_factory,
                        directory=directory / "second_stage",
                        stage_name="finetune",
                        epochs=1,
                    )
                )
                assert second["optimizer_iterations"] == 4
                checks.append(
                    dict(
                        monitor=monitor,
                        interruption_restore=True,
                        optimizer_restore=True,
                        callback_restore=True,
                        completion_skip=True,
                        mismatch_rejected=True,
                        rollback=True,
                        stage_transition=True,
                    )
                )
        report = dict(
            passed=True,
            tensorflow=tf.__version__,
            keras=getattr(tf.keras, "__version__", "unknown"),
            checks=checks,
        )
        write_json(workdir / "smoke_test_report.json", report)
        tf.keras.backend.clear_session()
        return report


# %% [markdown]
# ### 2.6. Thresholds, metrics and confidence intervals


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    """Calculate evaluation metrics, grouped intervals and reporting tables."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        roc_curve,
    )
    import matplotlib

    matplotlib.use("Agg")  # Use a noninteractive backend for saved figures.
    import matplotlib.pyplot as plt

    METRIC_NAMES = [
        "accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "roc_auc",
    ]
    SEEDS = [17, 29, 42, 71, 101]

    def planned_runs():
        runs = [
            dict(task="task2", design=design, seed=seed)
            for seed in SEEDS
            for design in ["filename_grouped", "image_level"]
        ]
        runs += [
            dict(task="task1", design="filename_grouped", seed=seed) for seed in SEEDS
        ]
        runs += [dict(task="task2", design="original_folders", seed=42)]
        for run in runs:
            run["run_id"] = f"{run['task']}_{run['design']}_seed{run['seed']}"
        return runs

    def select_threshold(y, probability):
        y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
        best, threshold = (-1.0, -1.0), 0.50
        for candidate in np.round(np.arange(0.10, 0.901, 0.01), 2):
            prediction = (probability >= candidate).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
            score = (
                0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)),
                accuracy_score(y, prediction),
            )
            if (
                score > best
            ):  # An ascending grid resolves the final tie in favor of the lower threshold.
                best, threshold = score, float(candidate)
        return threshold

    def metric_values(y, probability, threshold):
        y, probability = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
        prediction = (probability >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
        return dict(
            accuracy=float(accuracy_score(y, prediction)),
            sensitivity=float(recall_score(y, prediction, zero_division=0)),
            specificity=float(tn / max(tn + fp, 1)),
            precision=float(precision_score(y, prediction, zero_division=0)),
            f1=float(f1_score(y, prediction, zero_division=0)),
            roc_auc=(
                float(roc_auc_score(y, probability))
                if len(np.unique(y)) == 2
                else float("nan")
            ),
        )

    def validate_prediction_frame(predictions, expected, positive_class):
        required = {
            "image_id",
            "label",
            "split_group",
            "y_true",
            "probability",
            "threshold",
            "predicted_index",
            "positive_class",
        }
        if not required.issubset(predictions.columns):
            raise ValueError("Missing prediction fields.")
        if predictions.image_id.duplicated().any() or set(predictions.image_id) != set(
            expected.image_id
        ):
            raise ValueError("Predictions omit, repeat or add images.")
        merged = predictions.set_index("image_id").loc[expected.image_id]
        for column in ["label", "split_group"]:
            if not np.array_equal(
                merged[column].to_numpy(), expected[column].to_numpy()
            ):
                raise ValueError(
                    f"Prediction {column} disagrees with split assignment."
                )
        if not np.array_equal(
            merged.y_true.to_numpy(),
            expected.label.eq(positive_class).astype(int).to_numpy(),
        ):
            raise ValueError("Prediction truth/positive class mismatch.")
        if not merged.positive_class.eq(positive_class).all():
            raise ValueError("Positive class mismatch.")
        for column in ["probability", "threshold"]:
            values = pd.to_numeric(merged[column], errors="raise")
            if not np.isfinite(values).all() or not values.between(0, 1).all():
                raise ValueError("Invalid probability/threshold.")
        if merged.threshold.nunique() != 1:
            raise ValueError("Different thresholds within one prediction set.")
        if not np.array_equal(
            merged.predicted_index.to_numpy(),
            (merged.probability >= merged.threshold).astype(int).to_numpy(),
        ):
            raise ValueError("Stored predictions disagree with their threshold.")
        return True

    def sample_cluster_indices(groups, rng):
        keys = list(groups)
        sampled = rng.choice(len(keys), size=len(keys), replace=True)
        return np.concatenate([groups[keys[i]] for i in sampled])

    def cluster_bootstrap(predictions, replicates=2000, seed=42):
        frame = predictions.reset_index(drop=True)
        groups = {
            key: np.asarray(index)
            for key, index in frame.groupby("split_group").groups.items()
        }
        if not groups or frame.split_group.isna().any():
            raise ValueError("Valid split-unit groups required for cluster bootstrap.")
        rng = np.random.default_rng(seed)
        threshold = float(frame.threshold.iloc[0])
        point = metric_values(frame.y_true, frame.probability, threshold)
        samples = []
        for _ in range(replicates):
            sample = frame.iloc[sample_cluster_indices(groups, rng)]
            if sample.y_true.nunique() == 2:
                samples.append(
                    metric_values(sample.y_true, sample.probability, threshold)
                )
        if len(samples) < min(100, replicates // 2):
            raise ValueError("Too few valid two-class cluster-bootstrap resamples.")
        draws = pd.DataFrame(samples)
        return pd.DataFrame(
            [
                dict(
                    metric=metric,
                    estimate=point[metric],
                    lower_95=float(draws[metric].quantile(0.025)),
                    upper_95=float(draws[metric].quantile(0.975)),
                    method="percentile",
                    resampling_unit="filename_group_with_exact_duplicate_constraints",
                    attempted_replicates=replicates,
                    effective_replicates=len(draws),
                    bootstrap_seed=seed,
                    positive_class=frame.positive_class.iloc[0],
                )
                for metric in METRIC_NAMES
            ]
        )

    def threshold_robustness(predictions):
        selected = float(predictions.threshold.iloc[0])
        candidates = sorted(
            set(
                [
                    0.50,
                    selected,
                    max(0.0, round(selected - 0.05, 2)),
                    min(1.0, round(selected + 0.05, 2)),
                ]
            )
        )
        return pd.DataFrame(
            [
                dict(
                    threshold=t,
                    tuning_selected=bool(np.isclose(t, selected)),
                    default_050=bool(np.isclose(t, 0.50)),
                    positive_class=predictions.positive_class.iloc[0],
                    **metric_values(predictions.y_true, predictions.probability, t),
                )
                for t in candidates
            ]
        )

    def save_figure(fig, stem):
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white"
        )
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def export_run_reports(
        predictions, directory, run, primary=False, bootstrap_replicates=2000
    ):
        directory = Path(directory)
        values = metric_values(
            predictions.y_true,
            predictions.probability,
            float(predictions.threshold.iloc[0]),
        )
        row = dict(
            **run,
            positive_class=predictions.positive_class.iloc[0],
            threshold=float(predictions.threshold.iloc[0]),
            **values,
        )
        write_csv(directory / "test_metrics.csv", pd.DataFrame([row]))
        write_csv(
            directory / "threshold_sensitivity.csv", threshold_robustness(predictions)
        )
        if primary or run["design"] == "original_folders":
            write_csv(
                directory / "bootstrap_intervals.csv",
                cluster_bootstrap(predictions, bootstrap_replicates, run["seed"]),
            )
            title = f"{run['task'].upper()} {run['design'].replace('_', ' ')}\nPositive class: {row['positive_class']}"
            negative = "NORMAL" if run["task"] == "task1" else "BACTERIAL"
            cm = confusion_matrix(
                predictions.y_true, predictions.predicted_index, labels=[0, 1]
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cm, cmap="Blues")
            for (i, j), count in np.ndenumerate(cm):
                ax.text(
                    j,
                    i,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if count > cm.max() / 2 else "black",
                )
            ax.set(
                xticks=[0, 1],
                yticks=[0, 1],
                xticklabels=[negative, row["positive_class"]],
                yticklabels=[negative, row["positive_class"]],
                xlabel="Predicted dataset label",
                ylabel="Supplied dataset label",
                title=title,
            )
            save_figure(fig, directory / "confusion_matrix")
            fpr, tpr, _ = roc_curve(predictions.y_true, predictions.probability)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f"AUROC = {values['roc_auc']:.4f}")
            ax.plot([0, 1], [0, 1], "--", color="gray")
            ax.set(
                xlabel="False-positive rate", ylabel="True-positive rate", title=title
            )
            ax.legend()
            save_figure(fig, directory / "roc_curve")
        return row

    def export_summary(rows, directory):
        frame = pd.DataFrame(rows)
        expected = {run["run_id"] for run in planned_runs()}
        if (
            frame.empty
            or frame.run_id.duplicated().any()
            or set(frame.run_id) != expected
        ):
            raise ValueError(
                "All 16 distinct evaluated runs are required for final summary."
            )
        for column in METRIC_NAMES:
            if not np.isfinite(frame[column]).all():
                raise ValueError("Nonfinite final metric.")
        write_csv(Path(directory) / "all_run_results.csv", frame)
        repeated = frame[frame.design != "original_folders"]
        summary = repeated.groupby(["task", "design"])[METRIC_NAMES].agg(
            ["count", "mean", "std", "min", "max"]
        )
        summary.columns = ["_".join(item) for item in summary.columns]
        write_csv(Path(directory) / "repeated_split_summary.csv", summary.reset_index())
        task2 = repeated[repeated.task == "task2"]
        grouped = task2[task2.design == "filename_grouped"].set_index("seed")[
            METRIC_NAMES
        ]
        images = task2[task2.design == "image_level"].set_index("seed")[METRIC_NAMES]
        paired = (
            (grouped - images).add_prefix("filename_group_minus_image_").reset_index()
        )
        write_csv(Path(directory) / "task2_paired_seed_differences.csv", paired)
        return frame

    def export_workflow_figure(directory, manifest, complete_count):
        lines = [
            f"Dataset labels: {len(manifest):,} images (not microbiologically adjudicated)",
            "Audit filename rules, original folders, exact duplicates and conflicting labels",
            "Qualified filename groups + exact-duplicate constraints; identity unverified",
            "Task 2: 5 seeds x 2 split designs; Task 1: 5 grouped seeds; original folders: 1 model",
            f"Training and evaluation evidence: {complete_count}/16 verified runs",
            "Tuning-only threshold selection; held-out predictions; filename-group bootstrap",
            "Descriptive comparisons and reproducible artifacts; manuscript review remains separate",
        ]
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.axis("off")
        for i, line in enumerate(lines):
            y = 0.92 - 0.135 * i
            ax.text(
                0.5,
                y,
                line,
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
                bbox=dict(
                    boxstyle="round,pad=0.65", facecolor="#edf3f8", edgecolor="#37546c"
                ),
            )
            if i < len(lines) - 1:
                ax.annotate(
                    "",
                    xy=(0.5, y - 0.095),
                    xytext=(0.5, y - 0.038),
                    arrowprops=dict(arrowstyle="->"),
                )
        ax.set_title(
            "Figure 2. Analysis workflow — separate from the S1 checklist",
            fontsize=14,
            pad=15,
        )
        save_figure(fig, Path(directory) / "Figure_2")

    REVIEW_ITEMS = [
        (
            "R1 major 1 / R2 #4",
            "Dataset-derived subtype labels",
            "Metadata and Methods notes explicitly distinguish supplied labels from microbiological etiology.",
            "Update Abstract and Methods; no antibiotic-treatment or verified-unseen-patient claims.",
        ),
        (
            "R1 major 2 / R2 #1",
            "Grouping reliability and duplicate policy",
            "Parsing coverage, examples, fallbacks, group distributions, exact-image conflicts and duplicate constraints are audited.",
            "Identity cannot be verified from filenames; disclose this limitation.",
        ),
        (
            "R1 major 3 / R2 #6",
            "Threshold criterion and robustness",
            "Tuning balanced accuracy; accuracy then lower-threshold tie-breakers; 0.50 and +/-0.05 test tables.",
            "Describe operating points without claiming clinical-utility optimization.",
        ),
        (
            "R1 major 4 / R2 #3",
            "Repeated controlled comparisons",
            "Five Task 1 grouped runs, five paired Task 2 designs, original-folder comparator; descriptive summaries.",
            "Populate manuscript only after all 16 runs validate.",
        ),
        (
            "R1 major 5",
            "Replication oversampling",
            "Replication oversampling disabled; training class weights used consistently.",
            "Explain this change from earlier manuscript and the memorization risk of earlier replication.",
        ),
        (
            "R1 major 6",
            "Fine-tuning reproducibility",
            "DenseNet upper 100 layer positions; BatchNorm frozen; stage budgets, actual epochs, LR and trainable layers recorded.",
            "Transfer actual settings and completed epochs to Methods.",
        ),
        (
            "R1 minor 1",
            "90.70 percent comparator citation",
            "Unverified numeric literature benchmark removed.",
            "Verify source/endpoint before any optional reintroduction; do not invent a citation.",
        ),
        (
            "R1 minor 2",
            "Occlusion block size and stride",
            "48-pixel block and stride, median fill, consistent inference recorded.",
            "Include exact settings and qualitative-only interpretation in Methods.",
        ),
        (
            "R1 minor 3",
            "Figure 2 versus S1",
            "Figure 2 is exported as the workflow; S1 remains a separate checklist.",
            "Correct manuscript and Supporting Information cross-references.",
        ),
        (
            "R1 minor 4",
            "Repeated wording",
            "Reporting notes use restrained descriptive framing.",
            "Edit repeated Abstract/Discussion sentence.",
        ),
        (
            "R2 #2",
            "Original TRAIN/TEST overlap",
            "Raw/qualified/duplicate keys, category denominators and affected images; VAL excluded.",
            "Report both assumptions without equating keys with patient identities.",
        ),
        (
            "R2 #5",
            "Cluster confidence intervals",
            "2000 attempted whole-split-group bootstrap draws; effective count and percentile intervals.",
            "State unit and conditional grouping assumption in Methods.",
        ),
        (
            "R2 #7",
            "Task 1 versus Task 2 interpretation",
            "Different architectures, resolutions, augmentation and TTA are documented.",
            "Describe the comparison; do not infer intrinsic task difficulty experimentally.",
        ),
        (
            "R2 #8",
            "Public code and derived data",
            "Portable assignments, probabilities, seeds, configuration, histories and metrics exported.",
            "Deposit code and output bundle publicly; add persistent repository URL/DOI and availability statement.",
        ),
        (
            "R2 #9",
            "Novelty and related work",
            "Framed as a dataset audit / evaluation-design replication.",
            "Evaluate Ouerhani et al. doi:10.3390/jimaging12020081 and revise related work; code does not certify literature review.",
        ),
        (
            "R2 additional 1",
            "Terminology",
            "Filename-grouped evaluation, AUROC and chest radiograph used; no identity verification claimed.",
            "Standardize manuscript abbreviations and remove patient-safe wording.",
        ),
        (
            "R2 additional 2",
            "Sensitivity and recall",
            "Reports sensitivity once; no duplicate recall result column.",
            "Remove redundant sensitivity/recall columns from manuscript tables.",
        ),
        (
            "R2 additional 3",
            "Task 2 positive class",
            "VIRAL is explicit in prediction/metric tables and figure labels.",
            "Retain designation in manuscript captions.",
        ),
        (
            "R2 additional 4",
            "Screening claims",
            "No population-screening or treatment claims.",
            "Revise manuscript screening language and avoid asserting clinical deployment.",
        ),
        (
            "R2 additional 5",
            "Occlusion example selection",
            "Fixed seed by supplied class and correctness; same predictor for selection and map.",
            "Describe selection and non-localization interpretation.",
        ),
        (
            "R2 additional 6",
            "Methods specificity",
            "Detailed actual data, split, threshold and uncertainty settings exported.",
            "Reduce elementary equations/definitions; prioritize study-specific methods.",
        ),
        (
            "Journal 1",
            "Style and naming",
            "Artifacts have descriptive stable filenames.",
            "Check current PLOS ONE manuscript style and naming requirements.",
        ),
        (
            "Journal 2",
            "Figure 2 replacement",
            "Readable 300-DPI PNG and vector PDF generated.",
            "Inspect final figure and replace the low-resolution submission image.",
        ),
        (
            "Journal 3",
            "Response / marked / clean manuscript",
            "Reviewer evidence checklist and response draft provided.",
            "Produce and verify separate response letter, marked manuscript and clean manuscript.",
        ),
        (
            "Journal 4",
            "Suggested citations and disclosures",
            "No forced citation or fabricated disclosure.",
            "Evaluate suggested work for relevance; update financial disclosure only if applicable.",
        ),
    ]

    def reviewer_checklist(directory, complete_count):
        directory = Path(directory)
        rows = [
            dict(
                reviewer=ref,
                requirement=title,
                implementation=implementation,
                experiment_status=(
                    "all_16_evaluations_validated"
                    if complete_count == 16
                    else f"pending_{complete_count}_of_16"
                ),
                remaining_manuscript_or_external_action=action,
            )
            for ref, title, implementation, action in REVIEW_ITEMS
        ]
        write_csv(directory / "reviewer_requirement_checklist.csv", pd.DataFrame(rows))
        text = [
            "# Reviewer response working draft",
            "",
            "This is an evidence guide, not a completed response letter.",
            f"Verified evaluated runs: {complete_count}/16. No missing results are fabricated.",
            "",
        ]
        for ref, title, implementation, action in REVIEW_ITEMS:
            text += [
                f"## {ref}: {title}",
                implementation,
                "Remaining action: " + action,
                "",
            ]
        (directory / "reviewer_response_working_draft.md").write_text(
            "\n".join(text), encoding="utf-8"
        )


# %% [markdown]
# ### 2.7. Models, training and inference


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    """Define the models, training procedures and deterministic evaluation functions."""
    import gc
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator,
        load_img,
        img_to_array,
    )

    TASK1_AUGMENTATION = dict(
        rotation_range=12,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.15,
        shear_range=0.05,
        brightness_range=[0.85, 1.15],
        horizontal_flip=True,
        fill_mode="nearest",
    )
    TASK2_AUGMENTATION = dict(
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.10,
        brightness_range=[0.9, 1.1],
        horizontal_flip=True,
        fill_mode="nearest",
    )

    def model_spec(task):
        if task == "task1":
            return dict(
                task=task,
                architecture="InceptionV3",
                pretrained="imagenet",
                size=[299, 299],
                batch_size=32,
                augmentation=TASK1_AUGMENTATION,
                head_epochs=30,
                finetune_epochs=0,
                head_lr=0.001,
                monitor="val_loss",
                mode="min",
                es_patience=10,
                es_min_delta=0.0001,
                lr_patience=4,
                TTA_passes=3,
                TTA_rule="fixed_transform_per_image_id_and_pass_v1",
                positive_class="PNEUMONIA",
                classes=["NORMAL", "PNEUMONIA"],
                oversampling=False,
                class_weight="balanced_training_images",
                selection="minimum tuning loss",
                head="global_average_pool,dropout0.5,sigmoid",
            )
        return dict(
            task=task,
            architecture="DenseNet121",
            pretrained="imagenet",
            size=[320, 320],
            batch_size=16,
            augmentation=TASK2_AUGMENTATION,
            head_epochs=12,
            finetune_epochs=25,
            head_lr=0.001,
            finetune_lr=0.000003,
            monitor="val_auc",
            mode="max",
            es_patience=7,
            es_min_delta=0.0,
            lr_patience=3,
            upper_layers=100,
            backbone_batch_normalization_frozen=True,
            TTA_passes=0,
            positive_class="VIRAL",
            classes=["BACTERIAL", "VIRAL"],
            oversampling=False,
            class_weight="balanced_training_images",
            selection="maximum tuning AUROC within fine-tuning stage",
            head="global_average_pool,BN,dropout0.4,dense256_relu,BN,dropout0.3,sigmoid",
        )

    def preprocessing(task):
        return (
            tf.keras.applications.inception_v3.preprocess_input
            if task == "task1"
            else tf.keras.applications.densenet.preprocess_input
        )

    def build_network(task, pretrained="imagenet"):
        spec = model_spec(task)
        constructor = (
            tf.keras.applications.InceptionV3
            if task == "task1"
            else tf.keras.applications.DenseNet121
        )
        backbone = constructor(
            include_top=False, weights=pretrained, input_shape=(*spec["size"], 3)
        )
        backbone.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D(name="classifier_gap")(
            backbone.output
        )
        if task == "task1":
            x = tf.keras.layers.Dropout(0.5, name="classifier_dropout")(x)
        else:
            x = tf.keras.layers.BatchNormalization(name="classifier_bn1")(x)
            x = tf.keras.layers.Dropout(0.4, name="classifier_dropout1")(x)
            x = tf.keras.layers.Dense(256, activation="relu", name="classifier_dense")(
                x
            )
            x = tf.keras.layers.BatchNormalization(name="classifier_bn2")(x)
            x = tf.keras.layers.Dropout(0.3, name="classifier_dropout2")(x)
        output = tf.keras.layers.Dense(
            1, activation="sigmoid", name="classifier_output"
        )(x)
        model = tf.keras.Model(backbone.input, output)
        compile_network(model, spec["head_lr"])
        return model

    def compile_network(model, lr):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
        )

    def unfreeze_dense_upper(model, upper_layers=100):
        backbone_layers = [
            layer for layer in model.layers if not layer.name.startswith("classifier_")
        ]
        upper = {layer.name for layer in backbone_layers[-upper_layers:]}
        for layer in backbone_layers:
            layer.trainable = layer.name in upper and not isinstance(
                layer, tf.keras.layers.BatchNormalization
            )
        return dict(
            upper_layer_positions=upper_layers,
            actual_trainable_backbone_layers=int(
                sum(layer.trainable for layer in backbone_layers)
            ),
            trainable_parameters=int(
                sum(np.prod(weight.shape) for weight in model.trainable_weights)
            ),
        )

    def weights_hash(model):
        hasher = hashlib.sha256()
        for weight in model.get_weights():
            hasher.update(str(weight.shape).encode())
            hasher.update(str(weight.dtype).encode())
            hasher.update(weight.tobytes())
        return hasher.hexdigest()

    def training_generators(parts, task, seed):
        spec = model_spec(task)
        train_aug = ImageDataGenerator(
            preprocessing_function=preprocessing(task), **spec["augmentation"]
        )
        evaluation = ImageDataGenerator(preprocessing_function=preprocessing(task))

        def flow(generator, frame, shuffle):
            result = generator.flow_from_dataframe(
                frame,
                x_col="filepath",
                y_col="label",
                classes=spec["classes"],
                target_size=tuple(spec["size"]),
                batch_size=spec["batch_size"],
                class_mode="binary",
                shuffle=shuffle,
                seed=seed,
            )
            if result.n != len(frame):
                raise ValueError("A training/evaluation generator omitted images.")
            return result

        return flow(train_aug, parts["train"], True), flow(
            evaluation, parts["tuning"], False
        )

    def train_run(parts, run, directory, signature):
        spec = model_spec(run["task"])
        tf.keras.utils.set_random_seed(run["seed"])
        train_gen, tuning_gen = training_generators(parts, run["task"], run["seed"])
        counts = parts["train"].label.value_counts()
        weights = {
            index: len(parts["train"]) / (2 * counts[label])
            for index, label in enumerate(spec["classes"])
        }
        args = dict(
            train_data=train_gen,
            val_data=tuning_gen,
            class_weight=weights,
            signature=signature,
            monitor=spec["monitor"],
            mode=spec["mode"],
            es_patience=spec["es_patience"],
            es_min_delta=spec["es_min_delta"],
            lr_patience=spec["lr_patience"],
        )
        model, head = run_stage(
            lambda: build_network(run["task"]),
            directory=Path(directory) / "head",
            stage_name="head",
            epochs=spec["head_epochs"],
            **args,
        )
        stages = {"head": {k: v for k, v in head.items() if not k.startswith("_")}}
        architecture = dict(
            backbone_frozen_in_head=True, head_epochs_completed=head["epoch"]
        )
        if run["task"] == "task2":
            initial_hash = weights_hash(model)
            architecture.update(unfreeze_dense_upper(model, spec["upper_layers"]))

            def fine_factory():
                compile_network(model, spec["finetune_lr"])
                return model

            args["signature"] = dict(signature, finetune_initial_weights=initial_hash)
            model, fine = run_stage(
                fine_factory,
                directory=Path(directory) / "finetune",
                stage_name="finetune",
                epochs=spec["finetune_epochs"],
                **args,
            )
            architecture["finetune_epochs_completed"] = fine["epoch"]
            stages["finetune"] = {
                k: v for k, v in fine.items() if not k.startswith("_")
            }
        if not all(state["training_complete"] for state in stages.values()):
            raise ValueError("Missing stage completion.")
        record = dict(
            training_complete=True,
            signature_hash=object_hash(signature),
            stages=stages,
            selected_weights_sha256=weights_hash(model),
            selection=spec["selection"],
            architecture=architecture,
        )
        write_json(Path(directory) / "training_complete.json", record)
        write_json(
            Path(directory) / "training_specification.json", dict(spec, **architecture)
        )
        # Combine all stages in the saved training history.
        histories = pd.concat(
            [
                pd.DataFrame(state["history"]).assign(stage=name)
                for name, state in stages.items()
            ],
            ignore_index=True,
        )
        write_csv(Path(directory) / "training_history.csv", histories)
        curve_files = [
            Path(directory) / ("training_curves" + suffix)
            for suffix in [".png", ".pdf"]
        ]
        curve_marker = Path(directory) / "training_curves_signature.json"
        hist_hash = file_hash(Path(directory) / "training_history.csv")
        cached = json.loads(curve_marker.read_text()) if curve_marker.exists() else {}
        if cached.get("history") != hist_hash or not all(
            p.exists() and file_hash(p) == cached.get(p.name) for p in curve_files
        ):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for axis, metric in zip(axes, ["loss", "auc"]):
                for name in [metric, "val_" + metric]:
                    if name in histories:
                        axis.plot(
                            np.arange(1, len(histories) + 1),
                            histories[name],
                            label=name,
                        )
                axis.set(
                    xlabel="Completed epoch across stages",
                    ylabel=metric,
                    title=run["run_id"],
                )
                axis.legend()
            save_figure(fig, Path(directory) / "training_curves")
            write_json(
                curve_marker,
                dict(history=hist_hash, **{p.name: file_hash(p) for p in curve_files}),
            )
        del train_gen, tuning_gen
        gc.collect()
        return model, record

    def fixed_tta_parameters(image_id, image_shape, run_seed, passes=3):
        generator = ImageDataGenerator(**TASK1_AUGMENTATION)
        transforms = []
        for index in range(passes):
            seed = int.from_bytes(
                hashlib.sha256(
                    f"{run_seed}|{image_id}|{index}|tta-v1".encode()
                ).digest()[:4],
                "big",
            )
            transforms.append(generator.get_random_transform(image_shape, seed=seed))
        return transforms

    def predict_image_variants(model, images, task, image_id, seed):
        """Apply the same fixed transforms to an image and its occluded variants."""
        images = np.asarray(images, dtype="float32")
        spec = model_spec(task)
        preprocess = preprocessing(task)
        if task == "task1":
            transforms = fixed_tta_parameters(
                image_id, images.shape[1:], seed, spec["TTA_passes"]
            )
            augmenter = ImageDataGenerator(**TASK1_AUGMENTATION)
            probabilities = []
            for parameters in transforms:
                batch = np.stack(
                    [
                        augmenter.apply_transform(image.copy(), parameters)
                        for image in images
                    ]
                )
                probabilities.append(
                    np.asarray(model.predict_on_batch(preprocess(batch))).reshape(-1)
                )
            return np.mean(probabilities, axis=0)
        return np.asarray(model.predict_on_batch(preprocess(images.copy()))).reshape(-1)

    def predict_frame(model, frame, task, seed, probability_path, prediction_signature):
        """Save each prediction batch so inference can resume after interruption."""
        probability_path = Path(probability_path)
        sidecar = probability_path.with_suffix(".signature.json")
        signature = dict(
            prediction_signature,
            inference="fixed_image_tta_v1" if task == "task1" else "single_pass_v1",
        )
        if sidecar.exists() and json.loads(sidecar.read_text()) != signature:
            raise ValueError("Partial prediction signature mismatch.")
        write_json(sidecar, signature)
        existing = (
            pd.read_csv(probability_path)
            if probability_path.exists()
            else pd.DataFrame(columns=["image_id", "probability"])
        )
        if existing.image_id.duplicated().any() or not set(existing.image_id).issubset(
            set(frame.image_id)
        ):
            raise ValueError("Invalid cached inference IDs.")
        if len(existing) and (
            not np.isfinite(existing.probability).all()
            or not existing.probability.between(0, 1).all()
        ):
            raise ValueError("Invalid cached probabilities.")
        done = set(existing.image_id)
        pending = frame[~frame.image_id.isin(done)]
        batch_size = model_spec(task)["batch_size"]
        records = existing.to_dict("records")
        for start in range(0, len(pending), batch_size):
            batch = pending.iloc[start : start + batch_size]
            if task == "task1":
                # Apply each fixed augmentation pass to the complete batch.
                originals = [
                    img_to_array(
                        load_img(p, target_size=tuple(model_spec(task)["size"]))
                    )
                    for p in batch.filepath
                ]
                augmentation = ImageDataGenerator(**TASK1_AUGMENTATION)
                transforms = [
                    fixed_tta_parameters(i, image.shape, seed)
                    for i, image in zip(batch.image_id, originals)
                ]
                passes = []
                for pass_index in range(3):
                    arrays = np.stack(
                        [
                            augmentation.apply_transform(
                                image.copy(), transform[pass_index]
                            )
                            for image, transform in zip(originals, transforms)
                        ]
                    )
                    passes.append(
                        np.asarray(
                            model.predict_on_batch(preprocessing(task)(arrays))
                        ).reshape(-1)
                    )
                probs = np.mean(passes, axis=0)
            else:
                arrays = np.stack(
                    [
                        img_to_array(
                            load_img(p, target_size=tuple(model_spec(task)["size"]))
                        )
                        for p in batch.filepath
                    ]
                )
                probs = np.asarray(
                    model.predict_on_batch(preprocessing(task)(arrays))
                ).reshape(-1)
            if len(probs) != len(batch) or not np.isfinite(probs).all():
                raise ValueError("Nonfinite or wrong-length inference.")
            records.extend(
                dict(image_id=image_id, probability=float(p))
                for image_id, p in zip(batch.image_id, probs)
            )
            write_csv(probability_path, pd.DataFrame(records))
        result = (
            pd.DataFrame(records)
            .set_index("image_id")
            .loc[frame.image_id]
            .reset_index()
        )
        return result.probability.to_numpy()

    def prediction_table(frame, probabilities, threshold, task):
        positive = model_spec(task)["positive_class"]
        out = frame[
            ["image_id", "filename", "label", "qualified_key", "raw_key", "split_group"]
        ].copy()
        out["y_true"] = out.label.eq(positive).astype(int)
        out["probability"] = np.asarray(probabilities, dtype=float)
        out["threshold"] = threshold
        out["predicted_index"] = (out.probability >= threshold).astype(int)
        out["positive_class"] = positive
        return out

    def evaluate_run(model, record, parts, run, directory, signature):
        directory = Path(directory)
        positive = model_spec(run["task"])["positive_class"]
        inference_signature = dict(
            run_signature=object_hash(signature),
            weights=record["selected_weights_sha256"],
        )
        paths = [
            directory / "tuning_predictions.csv",
            directory / "test_predictions.csv",
        ]
        complete_path = directory / "evaluation_complete.json"
        if complete_path.exists():
            completed = json.loads(complete_path.read_text())
            valid_files = all(
                path.exists() and file_hash(path) == completed.get(path.name)
                for path in paths
            )
            if completed.get("signature") == inference_signature and valid_files:
                tuning, test = [pd.read_csv(path) for path in paths]
                validate_prediction_frame(tuning, parts["tuning"], positive)
                validate_prediction_frame(test, parts["test"], positive)
                threshold = select_threshold(tuning.y_true, tuning.probability)
                if not np.isclose(threshold, test.threshold.iloc[0]) or not np.isclose(
                    threshold, tuning.threshold.iloc[0]
                ):
                    raise ValueError(
                        "Stored threshold cannot be reproduced from tuning predictions."
                    )
                values = metric_values(test.y_true, test.probability, threshold)
                if any(
                    not np.isclose(values[key], completed["metrics"][key], atol=1e-12)
                    for key in METRIC_NAMES
                ):
                    raise ValueError("Stored metrics cannot be reproduced.")
                return test, completed
            # Archive previous evaluations before regenerating signature-checked predictions.
            complete_path.replace(
                directory / ("evaluation_previous_" + uuid.uuid4().hex + ".json")
            )
        # Rollback can change the selected weights. Archive the previous predictions
        # before generating results for the recovered model.
        expected_partial_signature = dict(
            inference_signature,
            inference=(
                "fixed_image_tta_v1" if run["task"] == "task1" else "single_pass_v1"
            ),
        )
        for name in [
            "tuning_probabilities.partial.csv",
            "test_probabilities.partial.csv",
        ]:
            probability_path = directory / name
            sidecar = probability_path.with_suffix(".signature.json")
            if (
                sidecar.exists()
                and json.loads(sidecar.read_text()) != expected_partial_signature
            ):
                token = uuid.uuid4().hex
                sidecar.replace(directory / (sidecar.name + ".stale_" + token))
                if probability_path.exists():
                    probability_path.replace(
                        directory / (probability_path.name + ".stale_" + token)
                    )
        tuning_probs = predict_frame(
            model,
            parts["tuning"],
            run["task"],
            run["seed"],
            directory / "tuning_probabilities.partial.csv",
            inference_signature,
        )
        threshold = select_threshold(
            parts["tuning"].label.eq(positive).astype(int), tuning_probs
        )
        test_probs = predict_frame(
            model,
            parts["test"],
            run["task"],
            run["seed"],
            directory / "test_probabilities.partial.csv",
            inference_signature,
        )
        tuning = prediction_table(parts["tuning"], tuning_probs, threshold, run["task"])
        test = prediction_table(parts["test"], test_probs, threshold, run["task"])
        validate_prediction_frame(tuning, parts["tuning"], positive)
        validate_prediction_frame(test, parts["test"], positive)
        write_csv(paths[0], tuning)
        write_csv(paths[1], test)
        complete = dict(
            evaluation_complete=True,
            signature=inference_signature,
            metrics=metric_values(test.y_true, test.probability, threshold),
            threshold=threshold,
            **{path.name: file_hash(path) for path in paths},
        )
        write_json(complete_path, complete)
        return test, complete

    def export_occlusion(model, predictions, expected, run, directory, weights_digest):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        selection = predictions.assign(
            correct=predictions.y_true.eq(predictions.predicted_index)
        )
        selected = []
        for truth in [0, 1]:
            for correct in [True, False]:
                pool = selection[
                    (selection.y_true == truth) & (selection.correct == correct)
                ].sort_values("image_id")
                if len(pool):
                    row = pool.sample(1, random_state=42).iloc[0]
                    selected.append((row, len(pool)))
        if not selected:
            raise ValueError("No eligible occlusion examples.")
        records = []
        for index, (row, pool_size) in enumerate(selected, 1):
            stem = directory / f"example_{index}"
            meta_path = stem.with_suffix(".json")
            cache_key = object_hash(
                dict(
                    image_id=row.image_id,
                    weights=weights_digest,
                    seed=run["seed"],
                    threshold=float(row.threshold),
                    task=run["task"],
                    patch=48,
                    stride=48,
                    predictor="fixed_image_tta_v1",
                )
            )
            if (
                meta_path.exists()
                and stem.with_suffix(".png").exists()
                and stem.with_suffix(".pdf").exists()
            ):
                saved = json.loads(meta_path.read_text())
                if saved.get("cache_key") == cache_key:
                    records.append(saved)
                    continue
            filepath = expected.set_index("image_id").loc[row.image_id, "filepath"]
            image = img_to_array(
                load_img(filepath, target_size=tuple(model_spec(run["task"])["size"]))
            )
            base = float(
                predict_image_variants(
                    model, image[None], run["task"], row.image_id, run["seed"]
                )[0]
            )
            if not np.isclose(base, float(row.probability), atol=2e-5):
                raise ValueError(
                    "Occlusion predictor disagrees with saved evaluated probability."
                )
            target = int(row.y_true) if bool(row.correct) else int(row.predicted_index)
            target_name = model_spec(run["task"])["classes"][target]
            heatmap, counts = np.zeros(image.shape[:2]), np.zeros(image.shape[:2])
            masks, locations = [], []
            for y in range(0, image.shape[0], 48):
                for x in range(0, image.shape[1], 48):
                    y2, x2 = min(y + 48, image.shape[0]), min(x + 48, image.shape[1])
                    variant = image.copy()
                    variant[y:y2, x:x2] = np.median(image)
                    masks.append(variant)
                    locations.append((y, y2, x, x2))
            for start in range(0, len(masks), 16):
                probabilities = predict_image_variants(
                    model,
                    masks[start : start + 16],
                    run["task"],
                    row.image_id,
                    run["seed"],
                )
                for location, probability in zip(
                    locations[start : start + 16], probabilities
                ):
                    y, y2, x, x2 = location
                    drop = (
                        base - float(probability)
                        if target == 1
                        else float(probability) - base
                    )
                    heatmap[y:y2, x:x2] += drop
                    counts[y:y2, x:x2] += 1
            heatmap = np.maximum(heatmap / np.maximum(counts, 1), 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
            for ax in axes:
                ax.imshow(image.astype("uint8"))
                ax.axis("off")
            axes[0].set_title(
                f"{'Correct' if row.correct else 'Incorrect'} dataset-label prediction\nP({row.positive_class})={base:.4f}; threshold={row.threshold:.2f}"
            )
            axes[1].imshow(heatmap, cmap="jet", alpha=0.35)
            axes[1].set_title(
                f"Occlusion score decrease: {target_name}\nSame inference as evaluation; not validated localization"
            )
            save_figure(fig, stem)
            saved = dict(
                cache_key=cache_key,
                image_id=row.image_id,
                correct=bool(row.correct),
                probability=base,
                positive_class=row.positive_class,
                threshold=float(row.threshold),
                target_label=target_name,
                selection_seed=42,
                selection_pool_size=pool_size,
                patch=48,
                stride=48,
                inference=(
                    "fixed TTA, three passes"
                    if run["task"] == "task1"
                    else "single pass"
                ),
            )
            write_json(meta_path, saved)
            records.append(saved)
        write_csv(directory / "occlusion_examples.csv", pd.DataFrame(records))


# %% [markdown]
# ### 2.8. Run schedule and completion checks


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    """Run the 16-model schedule and validate the evidence before marking it complete."""
    import os
    import sys
    import platform
    import importlib.metadata
    import zipfile

    METHODS_NOTES = """# Methods and interpretation notes

This is an internally evaluated dataset audit / evaluation-design replication. BACTERIAL and VIRAL are supplied filename-derived dataset labels, not a uniform microbiological reference standard. No treatment-selection, verified-unseen-patient or clinical-deployment claim is supported.

Pneumonia keys are subtype-qualified (personN_bacteria versus personN_virus). Their subtype purity is partly imposed by construction. Raw-prefix overlap does not establish shared patient identity. Neither candidate key independently verifies patient identity. Parsing coverage quantifies syntactic extraction, not clinically unambiguous IDs.

Generated grouped partitions keep qualified filename keys together and join keys linked by identical decoded RGB images. All same-label copies remain; contradictory exact-image labels stop execution. These exact-duplicate constraints are a documented change from the previous notebook. Original TRAIN/TEST assignments are preserved for the descriptive comparator; its original VAL folder is excluded. Raw, qualified, exact-image and constrained-unit overlaps are reported separately.

Task 1 has five filename-grouped repetitions. Task 2 has five paired seeds across image-level and filename-grouped designs. Nominal fractions are 70/15/15, with actual image and group counts exported because group sizes vary. The original-folder comparator uses original TRAIN split into 85/15 training/tuning and original TEST. Five repeated splits reuse one dataset and are not independent external cohorts. Mean/SD/range and paired differences are descriptive, not causal estimates of leakage.

Grouped seed 42 models are reused for the main results. Task 1 uses InceptionV3, 299x299 images, a frozen backbone, dropout 0.5, 30 maximum epochs and minimum tuning-loss selection. Task 2 uses DenseNet121, 320x320 images, 12 head-training and 25 fine-tuning maximum epochs, 100 upper backbone layer positions with backbone BatchNorm frozen, and fine-tuning LR 3e-6. Actual epochs and trainable layers are exported. All Task 2 designs use the best tuning-AUROC weights within fine-tuning. This changes the earlier single-split across-both-stages selection rule and is versioned accordingly.

Replication oversampling is disabled and balanced training class weights are used. Explain this change from older manuscript experiments and discuss the memorization risk of replication if older results are discussed. Task 1 and Task 2 differ in architecture, resolution, augmentation, training and inference; their performance comparison is descriptive rather than proof of intrinsic task difficulty.

Threshold selection maximizes tuning balanced accuracy on 0.10 through 0.90 by 0.01. Accuracy breaks ties, then the lower threshold. It does not optimize clinical utilities. Default 0.50 and +/-0.05 test robustness tables are descriptive and are not used to retune on test data. VIRAL is the positive class for Task 2; PNEUMONIA is positive for Task 1.

Primary and original-folder confidence intervals use 2000 attempted whole-constrained-filename-group bootstrap draws, retain every image in each sampled group and preserve repeated draws. Percentile 2.5/97.5 intervals and effective valid two-class counts are exported. These intervals are conditional on the uncertain grouping assumption.

Task 1 inference averages three deterministic image-ID-specific augmentation passes. The same fixed transforms are used for tuning, test inference and occluded variants. This reproducibility improvement is versioned and cannot be silently mixed with old TTA outputs. Occlusion samples at most one image from each supplied-class/correctness stratum using seed42, uses 48-pixel blocks/stride and median fill, and reports its selection pool. Incorrect examples explain the predicted class. Maps probe model behavior; they are not validated disease localization.

Checkpoints restore optimizer, callback and epoch state, but not exact stochastic training/input-iteration state. Recovery events are recorded; bitwise equivalence to an uninterrupted full training run is not claimed. Legacy metric rows alone are not reproducibly complete. Missing predictions are regenerated from compatible completed stages; insufficient legacy evidence requires a new run.

The unverified 90.70% literature benchmark is removed. Evaluate related work including Ouerhani et al., doi:10.3390/jimaging12020081, before revising the literature discussion. Figure 2 is the workflow, separate from the S1 checklist. Publish the notebook/Python code together with the output bundle and a persistent repository URL/DOI; writing to Drive is not public release. Update Abstract/Methods, Results, Discussion, cross-references and clean/marked manuscripts from verified outputs. The generated response document is only a working draft.
"""

    def software_versions():
        packages = [
            "tensorflow",
            "keras",
            "numpy",
            "pandas",
            "scikit-learn",
            "Pillow",
            "scipy",
        ]
        versions = {
            package: importlib.metadata.version(package) for package in packages
        }
        versions["python"] = platform.python_version()
        return versions

    def output_status(root, rows, phase, error=None):
        expected = {run["run_id"] for run in planned_runs()}
        actual = [row["run_id"] for row in rows]
        if len(actual) != len(set(actual)) or not set(actual).issubset(expected):
            raise ValueError("Duplicate or unexpected run in completion evidence.")
        complete = set(actual) == expected and phase == "complete"
        status = dict(
            analysis_complete=complete,
            submission_ready=False,
            phase=phase,
            evaluated_runs_verified=len(actual),
            expected_runs=16,
            missing_runs=sorted(expected - set(actual)),
            patient_identity_verified=False,
            public_release_verified=False,
            manuscript_package_verified=False,
            error=error,
        )
        write_json(Path(root) / "completion_status.json", status)
        write_csv(
            Path(root) / "partial_run_results.csv",
            pd.DataFrame(
                rows,
                columns=[
                    "run_id",
                    "task",
                    "design",
                    "seed",
                    "positive_class",
                    "threshold",
                    *METRIC_NAMES,
                ],
            ),
        )
        reviewer_checklist(Path(root), len(actual))
        return status

    def prepare_environment(config):
        if config["MOUNT_GOOGLE_DRIVE"]:
            try:
                from google.colab import drive
            except ImportError as exc:
                raise RuntimeError(
                    "This configuration mounts Google Drive. In local Python set MOUNT_GOOGLE_DRIVE=False and configure local paths."
                ) from exc
            drive.mount("/content/drive")
        source = Path(config["DATASET_SOURCE"])
        if not source.is_dir():
            raise FileNotFoundError(
                f"Dataset not found. Set DATASET_SOURCE to the folder containing train/val/test: {source}"
            )
        root = Path(config["OUTPUT_ROOT"])
        root.mkdir(parents=True, exist_ok=True)
        # Cache pretrained weights on Drive.
        os.environ["KERAS_HOME"] = str(root / "keras_cache")
        gpus = tf.config.list_physical_devices("GPU")
        if config["REQUIRE_GPU"] and not gpus:
            raise RuntimeError(
                "No GPU connected. Select a Colab GPU and reconnect. No training has started."
            )
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        tf.keras.mixed_precision.set_global_policy("float32")
        session_path = root / "session_environments.json"
        sessions = json.loads(session_path.read_text()) if session_path.exists() else []
        sessions.append(
            dict(
                time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                tensorflow=tf.__version__,
                float_policy="float32",
                devices=[
                    dict(
                        name=gpu.name,
                        details=str(tf.config.experimental.get_device_details(gpu)),
                    )
                    for gpu in gpus
                ],
            )
        )
        write_json(session_path, sessions)
        print("TensorFlow:", tf.__version__, "GPUs:", gpus)
        return root

    def inventory_legacy(config, root):
        records = []
        for directory in config.get("LEGACY_OUTPUT_ROOTS", []):
            directory = Path(directory)
            if directory.exists():
                # Record legacy files without importing or modifying them.
                for pattern in [
                    "controlled_repeated_split_progress.csv",
                    "*prediction*.csv",
                    "*.keras",
                ]:
                    for path in directory.glob(pattern):
                        records.append(
                            dict(
                                filename=path.name,
                                evidence_type=path.suffix,
                                reuse_status="not_imported_without_matching_v3_content_split_protocol_and_completion_evidence",
                            )
                        )
        write_csv(
            root / "legacy_evidence_inventory.csv",
            pd.DataFrame(
                records, columns=["filename", "evidence_type", "reuse_status"]
            ),
        )

    def validate_artifacts(root, rows):
        records = []
        for run in planned_runs():
            directory = root / "runs" / run["run_id"]
            required = [
                "training_complete.json",
                "evaluation_complete.json",
                "training_history.csv",
                "training_specification.json",
                "split_assignments.csv",
                "split_counts.csv",
                "partition_overlaps.csv",
                "tuning_predictions.csv",
                "test_predictions.csv",
                "test_metrics.csv",
                "threshold_sensitivity.csv",
            ]
            primary = run["seed"] == 42 and run["design"] == "filename_grouped"
            if primary or run["design"] == "original_folders":
                required += [
                    "bootstrap_intervals.csv",
                    "confusion_matrix.png",
                    "confusion_matrix.pdf",
                    "roc_curve.png",
                    "roc_curve.pdf",
                ]
            if primary:
                required += ["occlusion/occlusion_examples.csv"]
            for name in required:
                path = directory / name
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(
                        f'Missing required artifact: {run["run_id"]}/{name}'
                    )
                if path.suffix == ".png":
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        if min(image.info.get("dpi", (0, 0))) < 299:
                            raise ValueError("Figure DPI check failed.")
                records.append(
                    dict(
                        artifact=str(path.relative_to(root)).replace("\\", "/"),
                        sha256=file_hash(path),
                        bytes=path.stat().st_size,
                    )
                )
        if len(rows) != 16:
            raise ValueError("Missing evaluated runs.")
        write_csv(root / "artifact_verification.csv", pd.DataFrame(records))
        return records

    def package_outputs(root):
        target = root / "reviewer_output_bundle.zip"
        temp = root / "reviewer_output_bundle.pending.zip"
        allowed = {".csv", ".json", ".md", ".png", ".pdf", ".txt"}
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix not in allowed:
                    continue
                relative = path.relative_to(root)
                # Exclude checkpoints, cache files and recovery-test artifacts from the bundle.
                if any(
                    part in {"keras_cache", "self_test", "head", "finetune"}
                    for part in relative.parts
                ):
                    continue
                bundle.write(path, relative.as_posix())
        temp.replace(target)
        return target

    def run_project(config):
        if config["BOOTSTRAP_REPLICATES"] != 2000:
            raise ValueError(
                "The publication protocol requires 2000 attempted bootstrap draws."
            )
        if config["MAX_NEW_RUNS"] is not None and (
            not isinstance(config["MAX_NEW_RUNS"], int) or config["MAX_NEW_RUNS"] < 1
        ):
            raise ValueError(
                "MAX_NEW_RUNS must be None or a positive integer; it changes scheduling, not model budgets."
            )
        root = prepare_environment(config)
        rows = []
        output_status(root, rows, "validating")
        try:
            environment = software_versions()
            protocol = dict(
                version="reviewer_complete_v3",
                implementation_sha256=IMPLEMENTATION_SHA256,
                software=environment,
                grouping="qualified_filename_plus_exact_pixel_duplicate_components_v3",
                seeds=SEEDS,
                task1=model_spec("task1"),
                task2=model_spec("task2"),
                threshold="max tuning balanced accuracy; accuracy then lower threshold; grid0.10..0.90step0.01",
                bootstrap_attempts=2000,
                evaluation="fixed_image_id_TTA_v1",
            )
            protocol_path = root / "protocol.json"
            if (
                protocol_path.exists()
                and json.loads(protocol_path.read_text()) != protocol
            ):
                raise ValueError(
                    "Protocol/software changed. Restore recorded versions in protocol.json or choose a new OUTPUT_ROOT. Existing results are preserved."
                )
            write_json(protocol_path, protocol)
            (root / "METHODS_AND_LIMITATIONS.md").write_text(
                METHODS_NOTES, encoding="utf-8"
            )
            inventory_legacy(config, root)
            # Check native checkpoint recovery before starting the full training schedule.
            smoke = recovery_smoke_test(root / "self_test")
            write_json(root / "native_recovery_self_test.json", smoke)
            manifest = prepare_manifest(
                config["DATASET_SOURCE"],
                config["LOCAL_DATASET"],
                root / "audit_pending",
            )
            content_fingerprint = frame_hash(
                manifest,
                [
                    "image_id",
                    "file_sha256",
                    "pixel_sha256",
                    "qualified_key",
                    "split_group",
                    "label_task1",
                    "label_task2",
                ],
            )
            data_path = root / "dataset_fingerprint.json"
            dataset_identity = dict(
                content_sha256=content_fingerprint, images=len(manifest)
            )
            if (
                data_path.exists()
                and json.loads(data_path.read_text()) != dataset_identity
            ):
                raise ValueError(
                    "Dataset contents changed. Use a new experiment output folder."
                )
            write_json(data_path, dataset_identity)
            (root / "audit").mkdir(parents=True, exist_ok=True)
            for audit_file in (root / "audit_pending").glob("*.csv"):
                shutil.copy2(audit_file, root / "audit" / audit_file.name)
            export_data_audits(manifest, root / "audit")
            runs = planned_runs()

            def unfinished_checkpoint(run):
                directory = root / "runs" / run["run_id"]
                return (
                    any(
                        (directory / stage / "pointer.json").exists()
                        for stage in ["head", "finetune"]
                    )
                    and not (directory / "evaluation_complete.json").exists()
                )

            runs.sort(
                key=lambda run: not unfinished_checkpoint(run)
            )  # Keep the requested schedule order.
            new_runs = 0
            split_rows = []
            for run in runs:
                directory = root / "runs" / run["run_id"]
                directory.mkdir(parents=True, exist_ok=True)
                parts = make_partitions(
                    task_frame(manifest, run["task"]), run["design"], run["seed"]
                )
                signature = dict(
                    protocol=protocol,
                    content_sha256=content_fingerprint,
                    run=run,
                    split_sha256=frame_hash(
                        partition_assignments(parts),
                        ["image_id", "label", "split_group", "partition"],
                    ),
                )
                run_config_path = directory / "run_configuration.json"
                if (
                    run_config_path.exists()
                    and json.loads(run_config_path.read_text()) != signature
                ):
                    raise ValueError(f'Run configuration mismatch: {run["run_id"]}')
                write_json(run_config_path, signature)
                had_evaluation = (directory / "evaluation_complete.json").exists()
                if (
                    not had_evaluation
                    and config["MAX_NEW_RUNS"] is not None
                    and new_runs >= config["MAX_NEW_RUNS"]
                ):
                    # A shortened session leaves the analysis incomplete.
                    print("Session target reached. Run all again to continue.")
                    output_status(root, rows, "session_target_reached")
                    return root
                print("\nRUN:", run["run_id"])
                tf.keras.backend.clear_session()
                model = None
                try:
                    export_partition_audit(parts, run["design"], directory)
                    model, training_record = train_run(parts, run, directory, signature)
                    predictions, evaluation_record = evaluate_run(
                        model, training_record, parts, run, directory, signature
                    )
                    primary = run["design"] == "filename_grouped" and run["seed"] == 42
                    report_signature = dict(
                        test_sha256=evaluation_record["test_predictions.csv"],
                        implementation=IMPLEMENTATION_SHA256,
                    )
                    report_marker = directory / "reports_complete.json"
                    reports = (
                        json.loads(report_marker.read_text())
                        if report_marker.exists()
                        else {}
                    )
                    report_files = [
                        "test_metrics.csv",
                        "threshold_sensitivity.csv",
                        "training_curves.png",
                        "training_curves.pdf",
                    ]
                    if primary or run["design"] == "original_folders":
                        report_files += [
                            "bootstrap_intervals.csv",
                            "confusion_matrix.png",
                            "confusion_matrix.pdf",
                            "roc_curve.png",
                            "roc_curve.pdf",
                        ]
                    valid_reports = reports.get(
                        "signature"
                    ) == report_signature and all(
                        (directory / name).is_file()
                        and file_hash(directory / name)
                        == reports.get("files", {}).get(name)
                        for name in report_files
                    )
                    if valid_reports:
                        values = evaluation_record["metrics"]
                        row = dict(
                            **run,
                            positive_class=model_spec(run["task"])["positive_class"],
                            threshold=evaluation_record["threshold"],
                            **values,
                        )
                    else:
                        row = export_run_reports(
                            predictions,
                            directory,
                            run,
                            primary,
                            config["BOOTSTRAP_REPLICATES"],
                        )
                        write_json(
                            report_marker,
                            dict(
                                signature=report_signature,
                                files={
                                    name: file_hash(directory / name)
                                    for name in report_files
                                },
                            ),
                        )
                    if primary:
                        export_occlusion(
                            model,
                            predictions,
                            parts["test"],
                            run,
                            directory / "occlusion",
                            training_record["selected_weights_sha256"],
                        )
                    rows.append(row)
                    counts = pd.read_csv(directory / "split_counts.csv").assign(
                        run_id=run["run_id"], task=run["task"], seed=run["seed"]
                    )
                    split_rows.append(counts)
                    write_csv(
                        root / "split_counts_all_runs.csv",
                        pd.concat(split_rows, ignore_index=True),
                    )
                    output_status(root, rows, "running")
                    print(
                        f'Verified results this session: {len(rows)}/16; {run["run_id"]} complete.'
                    )
                    if not had_evaluation:
                        new_runs += 1
                finally:
                    del model
                    tf.keras.backend.clear_session()
                    gc.collect()
            export_summary(rows, root)
            export_workflow_figure(root, manifest, len(rows))
            validate_artifacts(root, rows)
            output_status(root, rows, "complete")
            bundle = package_outputs(root)
            print(
                "All 16 analyses validated. Manuscript preparation and public release remain separate."
            )
            print("Output bundle:", bundle)
            return root
        except Exception as exc:
            output_status(
                root, rows, "stopped_requires_attention", f"{type(exc).__name__}: {exc}"
            )
            raise


# %% [markdown]
# ### 2.9. Task 1 evaluation compatibility


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    REPORTING_PATCH_SHA256 = (
        "7da9a3ef23ef71b413199a762b10462ecc99d5edb184c47f1e8fcd1a2b7a3811"
    )
    """Apply the Task 1 evaluation update after the original definitions.
    
    Model construction, training signatures, checkpoint loading and split
    assignments retain the recorded protocol."""
    BASE_V3_IMPLEMENTATION_SHA256 = (
        "311fdd4442a623add59a8b23d14ff56b85fa278f03117ab70d36e15c955b0e36"
    )
    TASK1_INFERENCE_REVISION = "fixed_image_tta_v2_padded_physical_batch"
    _v3_predict_frame = predict_frame
    _v3_predict_image_variants = predict_image_variants
    _v3_evaluate_run = evaluate_run
    _v3_export_occlusion = export_occlusion
    _v3_export_data_audits = export_data_audits
    _v3_run_project = run_project

    def predict_task1_batch(model, images, image_ids, seed):
        """Use fixed-size batches for tuning, test and occlusion predictions.

        Transforms depend on the original image identifier. Padding repeats
        the last real input and is removed from the returned predictions."""
        images = np.asarray(images, dtype="float32")
        if len(images) != len(image_ids) or not len(images) or images.ndim != 4:
            raise ValueError(
                "Task 1 inference requires one ID per nonempty RGB image batch."
            )
        spec = model_spec("task1")
        size = spec["batch_size"]
        augmenter = ImageDataGenerator(**TASK1_AUGMENTATION)
        output = []
        for start in range(0, len(images), size):
            real = images[start : start + size]
            ids = list(image_ids[start : start + size])
            count = len(real)
            padded = np.concatenate(
                [real, np.repeat(real[-1:], size - count, axis=0)], axis=0
            )
            ids.extend([ids[-1]] * (size - count))
            params = [
                fixed_tta_parameters(key, image.shape, seed, spec["TTA_passes"])
                for key, image in zip(ids, padded)
            ]
            passes = []
            for index in range(spec["TTA_passes"]):
                transformed = np.stack(
                    [
                        augmenter.apply_transform(image.copy(), values[index])
                        for image, values in zip(padded, params)
                    ]
                )
                probabilities = np.asarray(
                    model.predict_on_batch(preprocessing("task1")(transformed))
                ).reshape(-1)
                if len(probabilities) != size or not np.isfinite(probabilities).all():
                    raise ValueError("Invalid fixed-batch Task 1 predictions.")
                passes.append(probabilities)
            output.extend(np.mean(passes, axis=0)[:count].tolist())
        output = np.asarray(output, dtype=float)
        if not ((0 <= output) & (output <= 1)).all():
            raise ValueError("Task 1 probabilities are outside [0, 1].")
        return output

    def predict_image_variants(model, images, task, image_id, seed):
        if task != "task1":
            return _v3_predict_image_variants(model, images, task, image_id, seed)
        return predict_task1_batch(model, images, [image_id] * len(images), seed)

    def predict_frame(model, frame, task, seed, probability_path, prediction_signature):
        if task != "task1":
            return _v3_predict_frame(
                model, frame, task, seed, probability_path, prediction_signature
            )
        probability_path = Path(probability_path)
        sidecar = probability_path.with_suffix(".signature.json")
        # The outer signature records the revision; retain the existing leaf name
        # so partial predictions remain compatible with the archive logic.
        signature = dict(prediction_signature, inference="fixed_image_tta_v1")
        if sidecar.exists() and json.loads(sidecar.read_text()) != signature:
            raise ValueError(
                "Task 1 partial inference signature differs; preserve it for inspection."
            )
        if probability_path.exists() and not sidecar.exists():
            raise ValueError("Partial predictions lack their provenance signature.")
        write_json(sidecar, signature)
        existing = (
            pd.read_csv(probability_path)
            if probability_path.exists()
            else pd.DataFrame(columns=["image_id", "probability"])
        )
        if existing.image_id.duplicated().any() or not set(existing.image_id).issubset(
            set(frame.image_id)
        ):
            raise ValueError("Invalid cached Task 1 image IDs.")
        if len(existing) and (
            not np.isfinite(existing.probability).all()
            or not existing.probability.between(0, 1).all()
        ):
            raise ValueError("Invalid cached Task 1 probabilities.")
        pending = frame[~frame.image_id.isin(set(existing.image_id))]
        records = existing.to_dict("records")
        size = model_spec("task1")["batch_size"]
        for start in range(0, len(pending), size):
            batch = pending.iloc[start : start + size]
            images = np.stack(
                [
                    img_to_array(
                        load_img(path, target_size=tuple(model_spec("task1")["size"]))
                    )
                    for path in batch.filepath
                ]
            )
            values = predict_task1_batch(model, images, batch.image_id.tolist(), seed)
            records.extend(
                dict(image_id=key, probability=float(value))
                for key, value in zip(batch.image_id, values)
            )
            write_csv(probability_path, pd.DataFrame(records))
        return (
            pd.DataFrame(records)
            .set_index("image_id")
            .loc[frame.image_id]
            .probability.to_numpy()
        )

    def _archive_task1_evaluation(directory, new_signature):
        """Archive the previous evaluation before repair, leaving model checkpoints in place."""
        marker = directory / "task1_inference_revision.json"
        previous = json.loads(marker.read_text()) if marker.exists() else {}
        if previous.get("target_signature") == new_signature:
            return
        candidates = [
            "evaluation_complete.json",
            "tuning_predictions.csv",
            "test_predictions.csv",
            "tuning_probabilities.partial.csv",
            "test_probabilities.partial.csv",
            "tuning_probabilities.partial.signature.json",
            "test_probabilities.partial.signature.json",
            "test_metrics.csv",
            "threshold_sensitivity.csv",
            "bootstrap_intervals.csv",
            "reports_complete.json",
            "roc_curve.png",
            "roc_curve.pdf",
            "confusion_matrix.png",
            "confusion_matrix.pdf",
        ]
        archive = directory / "evaluation_archive" / ("before_v3_1_" + uuid.uuid4().hex)
        copied = []
        for name in candidates:
            source = directory / name
            if source.is_file():
                archive.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, archive / name)
                copied.append(name)
        if (directory / "occlusion").is_dir():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copytree(directory / "occlusion", archive / "occlusion")
            copied.append("occlusion/")
        write_json(
            marker,
            dict(
                inference_revision=TASK1_INFERENCE_REVISION,
                target_signature=new_signature,
                archived_files=copied,
                archive=str(archive.relative_to(directory)) if copied else None,
                training_changed=False,
                reason="Unify physical batch shape for Task 1 evaluation and occlusion.",
            ),
        )
        if copied:
            print(
                "Task 1 evaluation repair: old reports archived; reusing trained weights and regenerating predictions."
            )

    def evaluate_run(model, record, parts, run, directory, signature):
        if run["task"] != "task1":
            return _v3_evaluate_run(model, record, parts, run, directory, signature)
        revised = dict(signature, evaluation_only_revision=TASK1_INFERENCE_REVISION)
        target_signature = dict(
            run_signature=object_hash(revised),
            weights=record["selected_weights_sha256"],
        )
        _archive_task1_evaluation(Path(directory), target_signature)
        return _v3_evaluate_run(model, record, parts, run, directory, revised)

    def export_occlusion(model, predictions, expected, run, directory, weights_digest):
        if run["task"] != "task1":
            return _v3_export_occlusion(
                model, predictions, expected, run, directory, weights_digest
            )
        # Record numerical differences without relaxing the consistency check.
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        selection = predictions.assign(
            correct=predictions.y_true.eq(predictions.predicted_index)
        )
        checks = []
        for truth in [0, 1]:
            for correct in [True, False]:
                pool = selection[
                    (selection.y_true == truth) & (selection.correct == correct)
                ].sort_values("image_id")
                if not len(pool):
                    continue
                row = pool.sample(1, random_state=42).iloc[0]
                filepath = expected.set_index("image_id").loc[row.image_id, "filepath"]
                image = img_to_array(
                    load_img(filepath, target_size=tuple(model_spec("task1")["size"]))
                )
                value = float(
                    predict_image_variants(
                        model, image[None], "task1", row.image_id, run["seed"]
                    )[0]
                )
                checks.append(
                    dict(
                        image_id=row.image_id,
                        stored_probability=float(row.probability),
                        reproduced_probability=value,
                        absolute_difference=abs(value - float(row.probability)),
                        probability_matches=bool(
                            np.isclose(value, float(row.probability), atol=2e-5)
                        ),
                        classification_matches=bool(
                            (value >= float(row.threshold)) == bool(row.predicted_index)
                        ),
                        physical_batch_size=model_spec("task1")["batch_size"],
                        inference_revision=TASK1_INFERENCE_REVISION,
                    )
                )
        write_csv(directory / "inference_consistency.csv", pd.DataFrame(checks))
        if not checks or not all(
            x["probability_matches"] and x["classification_matches"] for x in checks
        ):
            raise ValueError(
                "Task 1 inference consistency check failed. Inspect occlusion/inference_consistency.csv; training is preserved."
            )
        # Include the inference revision in the figure cache key.
        cache_digest = object_hash(
            dict(
                selected_weights=weights_digest,
                inference_revision=TASK1_INFERENCE_REVISION,
            )
        )
        _v3_export_occlusion(model, predictions, expected, run, directory, cache_digest)
        marker = pd.read_csv(directory / "occlusion_examples.csv")
        marker["selected_weights_sha256"] = weights_digest
        marker["inference_revision"] = TASK1_INFERENCE_REVISION
        write_csv(directory / "occlusion_examples.csv", marker)

    def export_data_audits(manifest, audit_dir):
        _v3_export_data_audits(manifest, audit_dir)
        # Export the recorded sample radiographs.
        rows = []
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, label in zip(axes, ["NORMAL", "BACTERIAL", "VIRAL"]):
            pool = (
                manifest[manifest.label_task1.eq("NORMAL")]
                if label == "NORMAL"
                else manifest[manifest.label_task2.eq(label)]
            )
            if pool.empty:
                plt.close(fig)
                raise ValueError(
                    "Missing dataset category for representative-image figure."
                )
            row = pool.sort_values("image_id").iloc[0]
            with Image.open(row.filepath) as opened:
                ax.imshow(opened.convert("RGB"))
            ax.axis("off")
            ax.set_title(label + "\nSupplied dataset label")
            rows.append(
                dict(
                    category=label,
                    image_id=row.image_id,
                    selection="First lexicographic image_id within supplied category; illustrative only.",
                )
            )
        fig.suptitle(
            "Example chest radiographs — dataset labels are not microbiological confirmation"
        )
        stem = Path(audit_dir).parent / "figures" / "dataset_examples"
        save_figure(fig, stem)
        write_csv(stem.with_suffix(".csv"), pd.DataFrame(rows))

    def run_project(config):
        root = Path(config["OUTPUT_ROOT"])
        root.mkdir(parents=True, exist_ok=True)
        if IMPLEMENTATION_SHA256 != BASE_V3_IMPLEMENTATION_SHA256:
            raise ValueError(
                "This reporting repair supports the unchanged delivered V3 training implementation only."
            )
        write_json(
            root / "reporting_patch.json",
            dict(
                revision="V3.1",
                source_sha256=REPORTING_PATCH_SHA256,
                training_implementation_sha256=BASE_V3_IMPLEMENTATION_SHA256,
                task1_inference_revision=TASK1_INFERENCE_REVISION,
                training_changed=False,
                figures_inline=True,
                old_task1_predictions_recomputed=True,
            ),
        )
        return _v3_run_project(config)

    METHODS_NOTES += """

V3.1 reporting repair: Task 1 tuning, test and occlusion inference use one shared
implementation with the configured training batch size as a fixed physical
inference batch shape. Short batches are padded with copies of the final input;
padding predictions are discarded. The original three image-ID-specific transforms
are unchanged. Prior Task 1 evaluation artifacts are archived, then predictions,
tuning-selected thresholds and reports are regenerated from the same selected
weights. Training, split assignments and training checkpoint signatures are
unchanged; reporting_patch.json records the repair source hash separately.
The original GPU mismatch was observed in an occlusion check; its numeric size was
not logged, so batch-dependent numerics remain a plausible cause, not a verified
diagnosis. Strict consistency and classification checks remain in place, with
per-example numeric diagnostics. Clinical results are not inferred from test data.
"""


# %% [markdown]
# ### 2.10. Storage check


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    # Check the Drive connection before the runner writes results.
    _drive_ready_run_project = run_project

    def run_project(config):
        initialize_paths(config)
        return _drive_ready_run_project(config)


# %% [markdown]
# ### 2.11. Resume checks and result packaging


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    FINAL_REPORT_PATCH_SHA256 = (
        "a2ae468f50e5537060452b8ca94111be96c4786b474b3e34aebf5685c7230dc9"
    )
    """Update report packaging while retaining saved model and inference signatures."""

    def export_workflow_figure(directory, manifest, complete_count):
        lines = [
            f"Dataset: {len(manifest):,} chest radiographs with supplied dataset labels",
            "Audit filename rules, original folders, exact duplicates and conflicting labels",
            "Qualified filename groups + exact-duplicate constraints; identity unverified",
            "Task 2: 5 seeds x 2 split designs; Task 1: 5 grouped seeds; original folders: 1 model",
            f"Training and evaluation evidence: {complete_count}/16 verified runs",
            "Tuning-only threshold selection; held-out predictions; filename-group bootstrap",
            "Descriptive comparisons; subtype labels lack uniform microbiological confirmation",
        ]
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.axis("off")
        for i, line in enumerate(lines):
            y = 0.92 - 0.135 * i
            ax.text(
                0.5,
                y,
                line,
                ha="center",
                va="center",
                fontsize=11,
                wrap=True,
                bbox=dict(
                    boxstyle="round,pad=0.65", facecolor="#edf3f8", edgecolor="#37546c"
                ),
            )
            if i < len(lines) - 1:
                ax.annotate(
                    "",
                    xy=(0.5, y - 0.095),
                    xytext=(0.5, y - 0.038),
                    arrowprops=dict(arrowstyle="->"),
                )
        ax.set_title(
            "Figure 2. Analysis workflow - separate from the S1 checklist",
            fontsize=14,
            pad=15,
        )
        save_figure(fig, Path(directory) / "Figure_2")

    def package_outputs(root):
        """Package current evidence and retain earlier evaluations in their archive folders."""
        root = Path(root)
        note = """# Current reviewer evidence bundle

This bundle contains current analyses and their audit records. Read completion_status.json;
all 16 verified analyses are necessary, and submission_ready remains false.

Old evaluation archives, stale/previous partial reports, temporary files, downloaded caches,
and native model snapshots are excluded from this shareable report ZIP. They remain on Drive.
Keep the original Drive folder to resume or reproduce saved-model inference. Deposit this full
notebook and matching Python source alongside the public evidence bundle. The ZIP alone does
not publish code, data, or models and does not establish journal acceptance.

Use filename-grouped evaluation. Qualified-key purity and raw-prefix overlaps do not verify
patient identity. BACTERIAL/VIRAL are supplied dataset labels, not uniformly microbiologically
confirmed causes. Repeated-split comparisons are descriptive. Revise the manuscript and response
letter from verified evidence; complete the literature assessment and public repository deposit.
"""
        (root / "BUNDLE_README.md").write_text(note, encoding="utf-8")
        target, pending = (
            root / "reviewer_output_bundle.zip",
            root / "reviewer_output_bundle.pending.zip",
        )
        allowed = {".csv", ".json", ".md", ".png", ".pdf", ".txt"}
        excluded = {
            "keras_cache",
            "self_test",
            "head",
            "finetune",
            "evaluation_archive",
            "audit_pending",
        }
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix not in allowed:
                    continue
                relative = path.relative_to(root)
                if any(part in excluded for part in relative.parts):
                    continue
                if any(
                    token in path.name
                    for token in (".stale_", "_previous_", ".pending.", ".tmp")
                ):
                    continue
                bundle.write(path, relative.as_posix())
        with zipfile.ZipFile(pending) as bundle:
            failure = bundle.testzip()
            if failure:
                raise RuntimeError("Output ZIP integrity check failed: " + failure)
        pending.replace(target)
        return target

    _v312_previous_run_project = run_project

    def run_project(config):
        # Check compatibility before the runner updates its session records.
        initialize_paths(config)
        ensure_recorded_runtime(config, allow_install=False)
        expected_protocol = dict(
            version="reviewer_complete_v3",
            implementation_sha256=IMPLEMENTATION_SHA256,
            software=software_versions(),
            grouping="qualified_filename_plus_exact_pixel_duplicate_components_v3",
            seeds=SEEDS,
            task1=model_spec("task1"),
            task2=model_spec("task2"),
            threshold="max tuning balanced accuracy; accuracy then lower threshold; grid0.10..0.90step0.01",
            bootstrap_attempts=2000,
            evaluation="fixed_image_id_TTA_v1",
        )
        path = Path(config["OUTPUT_ROOT"]) / "protocol.json"
        if path.is_file() and json.loads(path.read_text()) != expected_protocol:
            raise RuntimeError(
                "Saved protocol differs beyond the restored software versions. Existing summaries and "
                "models are preserved. Restore the original configuration; do not edit protocol.json."
            )
        return _v312_previous_run_project(config)


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError(
        "Mode changed mid-session. Restart the session and Run all with one selected mode."
    )
if MODE == "resume_analysis":
    _ANALYSIS_SETUP_READY = True
    _ANALYSIS_SUCCEEDED = False
    output_directory = run_project(CONFIG)
    _ANALYSIS_SUCCEEDED = True
    print("Analysis results:", output_directory)


# %% [markdown]
# ## 3. Plotting dependencies
# 
# Install any missing plotting libraries. This step leaves TensorFlow and Keras unchanged.
# In recovery mode, it runs after the analysis and its environment checks.


# %%
# Install missing plotting packages; leave the model libraries unchanged.
import importlib.util
import subprocess
import sys

needed = {
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "matplotlib": "matplotlib",
    "pymupdf": "PyMuPDF",
    "sklearn": "scikit-learn",
}
missing = [
    package
    for module, package in needed.items()
    if importlib.util.find_spec(module) is None
]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
print("Plotting libraries ready. CPU runtime is sufficient.")


# %% [markdown]
# ## 4. Figure and table preparation
# 
# The export checks prediction coverage, dataset and split fingerprints, recorded file hashes, repeated-run
# summaries and occlusion metadata. Sample radiographs come from the three recorded selections in the dataset.
# Source hashes are checked again after export.
# 
# | Figure | Content |
# |---|---|
# | Fig1 | Sample NORMAL, BACTERIAL PNEUMONIA and VIRAL PNEUMONIA radiographs |
# | Fig2 | Study workflow |
# | Fig3 | Task 1: primary seed-42 confusion matrix and ROC curve |
# | Fig4 | Task 2: filename-grouped seed-42 confusion matrix and ROC curve |
# | Fig5 | Task 2: controlled comparison across five matched seeds |
# | Fig6–Fig7 | Saved Task 1 and Task 2 occlusion examples |
# 
# Figures are exported as 300-DPI RGB TIFF, PNG and PDF, with captions in a separate file. The occlusion panels
# retain the resolution of their saved raster images. Figure 2 is the workflow; S1 is a separate reporting checklist.
# 
# [PLOS ONE figure guidance](https://journals.plos.org/plosone/s/figures).
# 
# ### 4.1. Storage and result readers


# %%
"""Connect Drive while preserving existing mount contents."""

import os
import json
from pathlib import Path, PurePosixPath


def _pneumonia_active_drive(path):
    path = Path(path)
    try:
        return (
            not path.is_symlink()
            and os.path.ismount(path)
            and ((path / "MyDrive").is_dir() or (path / "My Drive").is_dir())
        )
    except OSError as exc:
        raise RuntimeError(
            "A Drive mount is not responding. Preserve files and reconnect the runtime before continuing."
        ) from exc


def _pneumonia_empty_mount(path):
    path = Path(path)
    if path.is_symlink():
        return False
    if not path.exists():
        return True
    return path.is_dir() and next(path.iterdir(), None) is None


def _pneumonia_rebase_path(value, old_root, new_root):
    path, before, after = [
        PurePosixPath(str(item).replace("\\", "/"))
        for item in [value, old_root, new_root]
    ]
    try:
        relative = path.relative_to(before)
    except ValueError:
        return value
    if ".." in relative.parts:
        raise ValueError("Drive paths must not escape the configured mount directory.")
    return str(after / relative)


def ensure_pneumonia_drive(config, mount_callable=None, active_check=None):
    """Rebase only the mount prefix, preserving the exact MyDrive folder names."""
    requested = Path(config.get("DRIVE_MOUNTPOINT", "/content/drive"))
    if not requested.is_absolute():
        raise ValueError("DRIVE_MOUNTPOINT must be an absolute path.")
    active_check = active_check or _pneumonia_active_drive
    alternatives = [
        requested.parent / ("pneumonia_drive" + (f"_{i}" if i else ""))
        for i in range(16)
    ]
    candidates = list(dict.fromkeys([requested, *alternatives]))
    # Reuse active mounts: mounting again can stop an existing Drive process.
    # A new mount point must be an empty directory.
    target = next((path for path in candidates if active_check(path)), None)
    reused = target is not None
    if target is None:
        target = next(
            (path for path in candidates if _pneumonia_empty_mount(path)), None
        )
        if target is None:
            raise RuntimeError(
                "No empty Drive mount location was found. Existing files were left untouched."
            )
        if not target.parent.is_dir():
            raise RuntimeError("The parent of the selected Drive mount does not exist.")
        if mount_callable is None:
            try:
                from google.colab import drive
            except ImportError as exc:
                raise RuntimeError(
                    "This preset uses Colab. For local Python, disable Drive mounting and set local paths."
                ) from exc
            mount_callable = drive.mount
        print("Connecting Google Drive at:", target)
        if target != requested:
            print(
                "Keeping existing files at",
                requested,
                "untouched; using an empty mount location.",
            )
        mount_callable(str(target))
        if not active_check(target):
            raise RuntimeError(
                "Google Drive was not verified as mounted. No output folder or cache was created."
            )
    else:
        print("Reusing connected Google Drive at:", target)
    updated = dict(config)
    for key in ["DATASET_SOURCE", "OUTPUT_ROOT"]:
        updated[key] = _pneumonia_rebase_path(config[key], requested, target)
    updated["LEGACY_OUTPUT_ROOTS"] = [
        _pneumonia_rebase_path(path, requested, target)
        for path in config.get("LEGACY_OUTPUT_ROOTS", [])
    ]
    updated["DRIVE_MOUNTPOINT"] = str(target)
    # The analysis runner also calls mount(); authentication is complete here.
    # Skip the duplicate mount call.
    updated["MOUNT_GOOGLE_DRIVE"] = False
    updated["_DRIVE_MOUNT_GUARD"] = dict(
        version="V3.1.1",
        requested_mount=str(requested),
        active_mount=str(target),
        reused_active_mount=reused,
        local_conflicting_files_preserved=True,
    )
    return updated


"""Read saved prediction records and display existing results without modifying the analysis."""

import csv
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath

VIEWER_SEEDS = (17, 29, 42, 71, 101)
VIEWER_TASK1_REVISION = "fixed_image_tta_v2_padded_physical_batch"


def _viewer_runs():
    runs = [
        ("task2", design, seed)
        for seed in VIEWER_SEEDS
        for design in ("filename_grouped", "image_level")
    ]
    runs += [("task1", "filename_grouped", seed) for seed in VIEWER_SEEDS]
    runs += [("task2", "original_folders", 42)]
    return [
        dict(task=task, design=design, seed=seed, run_id=f"{task}_{design}_seed{seed}")
        for task, design, seed in runs
    ]


def _viewer_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {Path(path).name}")
    return value


def _viewer_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _viewer_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _viewer_object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _viewer_evidence(directory, run, task1_repair_required=False):
    """Check prediction coverage, signatures and file hashes without loading a model."""
    record = _viewer_json(directory / "evaluation_complete.json")
    training = _viewer_json(directory / "training_complete.json")
    config = _viewer_json(directory / "run_configuration.json")
    if (
        record.get("evaluation_complete") is not True
        or training.get("training_complete") is not True
    ):
        raise ValueError("Training/evaluation completion record is not true")
    digest = _viewer_object_sha(config)
    signature = record.get("signature", {})
    if training.get("signature_hash") != digest:
        raise ValueError("Recorded training configuration signature disagrees")
    inference_revision = (
        "single_pass_v1" if run["task"] == "task2" else "fixed_image_tta_v1"
    )
    if run["task"] == "task1":
        revised_digest = _viewer_object_sha(
            dict(config, evaluation_only_revision=VIEWER_TASK1_REVISION)
        )
        if signature.get("run_signature") == revised_digest:
            marker = _viewer_json(directory / "task1_inference_revision.json")
            if (
                marker.get("inference_revision") != VIEWER_TASK1_REVISION
                or marker.get("target_signature") != signature
                or marker.get("training_changed") is not False
            ):
                raise ValueError(
                    "Task 1 inference repair marker/signature is inconsistent"
                )
            inference_revision = VIEWER_TASK1_REVISION
        elif signature.get("run_signature") == digest:
            if task1_repair_required:
                raise ValueError(
                    "Old Task 1 predictions need V3.1 regeneration from saved trained weights"
                )
        else:
            raise ValueError("Unrecognized Task 1 evaluation configuration signature")
    elif signature.get("run_signature") != digest:
        raise ValueError("Recorded evaluation configuration signature disagrees")
    if signature.get("weights") != training.get(
        "selected_weights_sha256"
    ) or not signature.get("weights"):
        raise ValueError("Recorded selected-weight signatures disagree")
    if config.get("run") != run:
        raise ValueError("Run configuration does not match this scheduled run")
    assignments = _viewer_csv(directory / "split_assignments.csv")
    if not assignments or len({row["image_id"] for row in assignments}) != len(
        assignments
    ):
        raise ValueError("Missing or duplicate split assignments")
    positive = "PNEUMONIA" if run["task"] == "task1" else "VIRAL"
    threshold = float(record["threshold"])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid stored threshold")
    for partition in ("tuning", "test"):
        filename = f"{partition}_predictions.csv"
        if _viewer_sha(directory / filename) != record.get(filename):
            raise ValueError(f"Checksum mismatch: {filename}")
        predictions = _viewer_csv(directory / filename)
        expected = {
            row["image_id"]: row
            for row in assignments
            if row.get("partition") == partition
        }
        if (
            not expected
            or len(predictions) != len(expected)
            or {row["image_id"] for row in predictions} != set(expected)
        ):
            raise ValueError(f"Prediction coverage mismatch: {partition}")
        for row in predictions:
            original = expected[row["image_id"]]
            probability = float(row["probability"])
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability: {partition}")
            if row["positive_class"] != positive or int(row["y_true"]) != int(
                original["label"] == positive
            ):
                raise ValueError(f"Class/label mismatch: {partition}")
            if any(
                row[key] != original[key]
                for key in ("label", "qualified_key", "raw_key", "split_group")
            ):
                raise ValueError(f"Filename-group metadata mismatch: {partition}")
            if not math.isclose(float(row["threshold"]), threshold, abs_tol=1e-12):
                raise ValueError(f"Threshold mismatch: {partition}")
            if int(row["predicted_index"]) != int(probability >= threshold):
                raise ValueError(f"Prediction-label mismatch: {partition}")
    return record, inference_revision


def inventory_saved_results(output_root):
    """Inspect the 16 scheduled runs and their saved prediction records.

    Checks cover file hashes, image coverage and recorded signatures.
    Model files and figure contents are not validated by this reader.
    The returned analysis status is the status recorded by the runner."""
    root = Path(output_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Output folder not found: {root}. Mount Drive and check OUTPUT_ROOT."
        )
    result = dict(
        output_root=str(root),
        runs=[],
        metrics=[],
        diagnostics=[],
        saved_analysis_status=None,
    )
    repair_required = False
    patch_path = root / "reporting_patch.json"
    if patch_path.is_file():
        try:
            patch = _viewer_json(patch_path)
            repair_required = patch.get("revision") == "V3.1"
            if (
                repair_required
                and patch.get("task1_inference_revision") != VIEWER_TASK1_REVISION
            ):
                result["diagnostics"].append(
                    "V3.1 reporting marker has an unexpected Task 1 inference revision; inspect it before using results."
                )
        except (OSError, ValueError) as exc:
            result["diagnostics"].append(f"Cannot read reporting_patch.json: {exc}")
    status_path = root / "completion_status.json"
    if status_path.is_file():
        try:
            result["saved_analysis_status"] = _viewer_json(status_path)
        except (OSError, ValueError) as exc:
            result["diagnostics"].append(f"Cannot read completion_status.json: {exc}")
    for run in _viewer_runs():
        directory = root / "runs" / run["run_id"]
        primary = run["seed"] == 42 and run["design"] == "filename_grouped"
        figure_names = ["training_curves.png"]
        if primary or run["design"] == "original_folders":
            figure_names += ["roc_curve.png", "confusion_matrix.png"]
        if primary:
            figure_names += ["occlusion/occlusion_examples.csv"]
        row = dict(
            **run,
            training_record_present=(directory / "training_complete.json").is_file(),
            evaluation_record_present=(
                directory / "evaluation_complete.json"
            ).is_file(),
            prediction_evidence_verified=False,
            inference_revision="Not checked",
            missing_expected_figures="; ".join(
                name for name in figure_names if not (directory / name).is_file()
            ),
            note="No evaluation record",
        )
        if row["evaluation_record_present"]:
            try:
                record, inference_revision = _viewer_evidence(
                    directory, run, repair_required
                )
                row["prediction_evidence_verified"] = True
                row["inference_revision"] = inference_revision
                row["note"] = (
                    "Prediction files verified; model files not checked by viewer"
                )
                result["metrics"].append(
                    dict(
                        **run,
                        threshold=record["threshold"],
                        **record.get("metrics", {}),
                    )
                )
            except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
                row["note"] = f"Saved evidence needs inspection: {exc}"
        result["runs"].append(row)
    expected_dirs = {run["run_id"] for run in _viewer_runs()}
    if (root / "runs").is_dir():
        extras = sorted(
            path.name
            for path in (root / "runs").iterdir()
            if path.is_dir() and path.name not in expected_dirs
        )
        if extras:
            result["diagnostics"].append(
                "Unrecognized run folders (not counted): " + ", ".join(extras)
            )
    return result


def _viewer_sample_path(dataset_root, image_id):
    """Resolve a manifest image path within the dataset directory."""
    relative = PurePosixPath(str(image_id).replace("\\", "/"))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or ":" in relative.parts[0]
    ):
        raise ValueError("Unsafe image_id path")
    root = Path(dataset_root).expanduser().resolve()
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path is outside the dataset folder")
    if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError("Unsupported sample image format")
    return path


# %% [markdown]
# ### 4.2. Input verification


# %%
"""Prepare publication figures from saved analysis results."""

import csv
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pneumonia_figure_cache")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image as PILImage
import pymupdf

EXPECTED_IMPLEMENTATION = (
    "311fdd4442a623add59a8b23d14ff56b85fa278f03117ab70d36e15c955b0e36"
)
EXPECTED_BOOTSTRAP_ATTEMPTS = 2000
FIGURE_GUIDE = "https://journals.plos.org/plosone/s/figures"
IDENTITY_NOTE = (
    "Filename-grouped evaluation was used. Patient identities were not independently verified. "
    "Subtype-qualified key purity is partly imposed by construction; raw-prefix overlap alone "
    "does not establish that images represent the same patient."
)


def polish_connect(config):
    """Connect to Drive using the shared mount helper."""
    if config.get("MOUNT_GOOGLE_DRIVE", True):
        staged = ensure_pneumonia_drive(config)
        config.update(staged)
    if not Path(config["OUTPUT_ROOT"]).is_dir():
        raise FileNotFoundError(
            "The existing results folder is missing. Check the same Drive account and OUTPUT_ROOT."
        )
    print("Completed analysis folder:", config["OUTPUT_ROOT"])


def _polish_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _polish_hash(path):
    return _viewer_sha(path)


def verify_polish_inputs(config):
    root = Path(config["OUTPUT_ROOT"]).resolve()
    inventory = inventory_saved_results(root)
    status = inventory.get("saved_analysis_status") or {}
    failed = [
        r["run_id"] + ": " + r["note"]
        for r in inventory["runs"]
        if not r["prediction_evidence_verified"]
    ]
    if (
        status.get("analysis_complete") is not True
        or status.get("evaluated_runs_verified") != 16
        or failed
    ):
        raise ValueError(
            "Final polish requires all 16 completed analyses. It never repairs or trains missing models. "
            + "; ".join(failed)
        )
    protocol = _polish_json(root / "protocol.json")
    if protocol.get("implementation_sha256") != EXPECTED_IMPLEMENTATION:
        raise ValueError(
            "Unrecognized or synthetic experiment. Do not export test-fixture results as publication evidence."
        )
    ledger = {}

    def record(path):
        path = Path(path).resolve()
        ledger[str(path)] = _polish_hash(path)
        return path

    for name in (
        "completion_status.json",
        "protocol.json",
        "dataset_fingerprint.json",
        "audit/manifest.csv",
        "figures/dataset_examples.csv",
        "artifact_verification.csv",
        "all_run_results.csv",
        "repeated_split_summary.csv",
        "task2_paired_seed_differences.csv",
        "split_counts_all_runs.csv",
    ):
        record(root / name)
    fingerprint = _polish_json(root / "dataset_fingerprint.json")
    manifest = pd.read_csv(root / "audit/manifest.csv", keep_default_na=False)
    if manifest.image_id.duplicated().any() or len(manifest) != fingerprint["images"]:
        raise ValueError(
            "Saved manifest coverage differs from the completed dataset record."
        )
    identity_columns = [
        "image_id",
        "file_sha256",
        "pixel_sha256",
        "qualified_key",
        "split_group",
        "label_task1",
        "label_task2",
    ]
    manifest_digest = hashlib.sha256(
        manifest[identity_columns]
        .fillna("")
        .to_csv(index=False, lineterminator="\n")
        .encode()
    ).hexdigest()
    if manifest_digest != fingerprint["content_sha256"]:
        raise ValueError(
            "Saved manifest content fingerprint differs from the completed analysis."
        )
    artifact_rows = pd.read_csv(root / "artifact_verification.csv")
    artifact_hashes = dict(zip(artifact_rows.artifact, artifact_rows.sha256))
    results = pd.read_csv(root / "all_run_results.csv")
    expected = {r["run_id"] for r in _viewer_runs()}
    if (
        len(results) != 16
        or results.run_id.duplicated().any()
        or set(results.run_id) != expected
    ):
        raise ValueError(
            "All-run result table does not contain exactly the 16 scheduled runs."
        )
    verified_metrics = {r["run_id"]: r for r in inventory["metrics"]}
    for row in results.itertuples():
        saved = verified_metrics[row.run_id]
        for metric in (
            "accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "roc_auc",
            "threshold",
        ):
            if not np.isclose(
                float(getattr(row, metric)), float(saved[metric]), rtol=0, atol=1e-10
            ):
                raise ValueError(
                    "Summary disagrees with saved evaluation: "
                    + row.run_id
                    + "/"
                    + metric
                )
    for run in _viewer_runs():
        folder = root / "runs" / run["run_id"]
        configuration = _polish_json(record(folder / "run_configuration.json"))
        if (
            configuration["protocol"] != protocol
            or configuration["content_sha256"] != fingerprint["content_sha256"]
        ):
            raise ValueError("Run protocol/dataset differs: " + run["run_id"])
        assignments = pd.read_csv(
            folder / "split_assignments.csv", keep_default_na=False
        )
        split_digest = hashlib.sha256(
            assignments[["image_id", "label", "split_group", "partition"]]
            .fillna("")
            .to_csv(index=False, lineterminator="\n")
            .encode()
        ).hexdigest()
        if split_digest != configuration["split_sha256"]:
            raise ValueError("Saved split assignments differ: " + run["run_id"])
        for name in (
            "training_complete.json",
            "evaluation_complete.json",
            "split_assignments.csv",
            "tuning_predictions.csv",
            "test_predictions.csv",
            "threshold_sensitivity.csv",
            "training_history.csv",
        ):
            path = record(folder / name)
            relative = path.relative_to(root).as_posix()
            if artifact_hashes.get(relative) != ledger[str(path)]:
                raise ValueError("Final artifact checksum mismatch: " + relative)
        primary = run["seed"] == 42 and run["design"] == "filename_grouped"
        if primary or run["design"] == "original_folders":
            path = record(folder / "bootstrap_intervals.csv")
            if (
                artifact_hashes.get(path.relative_to(root).as_posix())
                != ledger[str(path)]
            ):
                raise ValueError("Stored interval table checksum mismatch.")
            intervals = pd.read_csv(path)
            if not intervals.attempted_replicates.eq(EXPECTED_BOOTSTRAP_ATTEMPTS).all():
                raise ValueError(
                    "Stored intervals do not match the 2,000-attempt publication protocol."
                )
    samples = pd.read_csv(root / "figures/dataset_examples.csv", keep_default_na=False)
    if (
        len(samples) != 3
        or samples.category.duplicated().any()
        or set(samples.category) != {"NORMAL", "BACTERIAL", "VIRAL"}
    ):
        raise ValueError("Expected the three originally selected radiographs.")
    originals = []
    by_id = manifest.set_index("image_id")
    for category in ("NORMAL", "BACTERIAL", "VIRAL"):
        row = samples[samples.category == category].iloc[0]
        image_id = row.image_id
        if image_id not in by_id.index:
            raise ValueError("Selected radiograph absent from original manifest.")
        metadata = by_id.loc[image_id]
        label = metadata.label_task1 if category == "NORMAL" else metadata.label_task2
        if label != category:
            raise ValueError("Selected radiograph category disagrees with manifest.")
        path = _viewer_sample_path(config["DATASET_SOURCE"], image_id)
        if not path.is_file() or _polish_hash(path) != metadata.file_sha256:
            raise ValueError("Original sample image is missing or changed: " + image_id)
        record(path)
        originals.append(
            dict(
                category=category,
                image_id=image_id,
                path=path,
                selection=row.selection,
                file_sha256=metadata.file_sha256,
            )
        )
    return dict(
        root=root,
        inventory=inventory,
        manifest=manifest,
        results=results,
        samples=originals,
        ledger=ledger,
        protocol=protocol,
    )


# %% [markdown]
# ### 4.3. Sample panel and workflow


# %%
def _polish_save(fig, directory, name, caption, manifest):
    """Save a Times-font PDF and render its TIFF and PNG copies at 300 DPI."""
    directory = Path(directory)
    pdf = directory / (name + ".pdf")
    with matplotlib.rc_context(
        {
            "pdf.use14corefonts": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    ):
        for text in fig.findobj(matplotlib.text.Text):
            text.set_fontfamily("serif")
            text.set_fontweight("roman")
        fig.savefig(pdf, format="pdf", facecolor="white", dpi=300)
    plt.close(fig)
    with pymupdf.open(pdf) as document:
        if document.page_count != 1:
            raise ValueError("Expected one page per figure.")
        page = document[0]
        fonts = sorted({font[3] for font in page.get_fonts()})
        if any(not font.startswith(("Times", "Symbol")) for font in fonts):
            raise ValueError("Unexpected figure font: " + str(fonts))
        pixel = page.get_pixmap(dpi=300, colorspace=pymupdf.csRGB, alpha=False)
        png = directory / (name + ".png")
        pixel.save(png)
        image = PILImage.frombytes("RGB", (pixel.width, pixel.height), pixel.samples)
        tif = directory / (name + ".tif")
        image.save(tif, compression="tiff_lzw", dpi=(300, 300))
    if (
        not 789 <= image.width <= 2250
        or image.height > 2625
        or tif.stat().st_size >= 10_000_000
    ):
        raise ValueError("Figure dimensions/file size failed export checks: " + name)
    with PILImage.open(tif) as check:
        if check.mode != "RGB" or any(abs(x - 300) > 0.01 for x in check.info["dpi"]):
            raise ValueError("TIFF color mode or resolution check failed.")
    manifest.append(
        dict(
            figure=name,
            width_pixels=image.width,
            height_pixels=image.height,
            dpi=300,
            fonts=fonts,
            caption=caption,
            files={
                p.name: dict(sha256=_polish_hash(p), bytes=p.stat().st_size)
                for p in (pdf, png, tif)
            },
        )
    )


def draw_sample_panel(samples):
    fig = plt.figure(figsize=(7.2, 3.15))
    titles = ("NORMAL", "BACTERIAL PNEUMONIA", "VIRAL PNEUMONIA")
    for index, (sample, title) in enumerate(zip(samples, titles)):
        ax = fig.add_axes([0.018 + index * 0.328, 0.055, 0.31, 0.84])
        with PILImage.open(sample["path"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        ax.imshow(pixels, interpolation="nearest", aspect="equal")
        ax.set_anchor("N")
        ax.set_axis_off()
        fig.text(
            0.173 + index * 0.328,
            0.94,
            chr(65 + index) + "  " + title,
            fontsize=10,
            ha="center",
            va="center",
        )
    return fig


def draw_workflow(image_count):
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.975)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, w, h, text, fill="#eef3f6", size=10):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.008,rounding_size=0.009",
                linewidth=0.9,
                edgecolor="#405467",
                facecolor=fill,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            linespacing=1.3,
        )

    def arrow(a, b):
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={"arrowstyle": "->", "color": "#405467", "lw": 1.0},
        )

    box(
        0.18,
        0.905,
        0.64,
        0.065,
        f"Chest radiograph dataset: {image_count:,} images\nSupplied NORMAL, BACTERIAL and VIRAL labels",
    )
    arrow((0.5, 0.897), (0.5, 0.86))
    box(
        0.12,
        0.765,
        0.76,
        0.085,
        "Dataset audit\nFilename parsing, exact duplicates and conflicting labels\nOriginal-folder overlaps and explicit denominators",
    )
    arrow((0.5, 0.758), (0.5, 0.718))
    box(
        0.12,
        0.625,
        0.76,
        0.085,
        "Grouping definition for filename-grouped evaluation\nQualified filename keys + exact-duplicate constraints\nPatient identities not independently verified",
    )
    for x in (0.165, 0.5, 0.835):
        arrow((0.5, 0.616), (x, 0.56))
    box(
        0.015,
        0.405,
        0.3,
        0.145,
        "Task 1\nNORMAL vs PNEUMONIA\n5 filename-grouped runs\nInceptionV3",
        size=9.5,
    )
    box(
        0.35,
        0.405,
        0.3,
        0.145,
        "Task 2 controlled comparison\nBACTERIAL vs VIRAL\nFilename-grouped vs image-level\n5 seeds; DenseNet121; 10 runs",
        size=9.5,
    )
    box(
        0.685,
        0.405,
        0.3,
        0.145,
        "Task 2 comparator\nOriginal TRAIN/TEST folders\nSeed 42: 1 run\nDenseNet121",
        size=9.5,
    )
    for x in (0.165, 0.5, 0.835):
        arrow((x, 0.397), (0.5, 0.352))
    box(
        0.12,
        0.255,
        0.76,
        0.09,
        "16 completed model runs\nTuning-based checkpoint and threshold selection\nGrouped seed-42 models reused for primary figures",
    )
    arrow((0.5, 0.247), (0.5, 0.208))
    box(
        0.12,
        0.115,
        0.76,
        0.085,
        "Held-out evaluation\nSaved predictions; default 0.50 and ±0.05 threshold sensitivity\n2,000-attempt filename-group bootstrap for primary/comparator results",
        size=9.5,
    )
    arrow((0.5, 0.107), (0.5, 0.073))
    box(
        0.12,
        0.012,
        0.76,
        0.054,
        "Descriptive repeated-run comparisons\nReproducible tables, figures and scientific limitations",
    )
    return fig


# %% [markdown]
# ### 4.4. Performance and occlusion figures


# %%
"""Plot the primary results, controlled comparison and saved occlusion examples."""

import io
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


def _final_record(context, path):
    """Record an input hash and reject changes to an input already read."""
    path = Path(path).resolve()
    digest = _polish_hash(path)
    previous = context["ledger"].get(str(path))
    if previous is not None and previous != digest:
        raise ValueError("Saved evidence changed before plotting: " + str(path))
    context["ledger"][str(path)] = digest
    return path


def _final_require(condition, message):
    if not condition:
        raise ValueError(message)


def _final_bool(value):
    """Parse explicit Boolean values, including CSV strings."""
    if value is True or isinstance(value, np.bool_) and bool(value):
        return True
    if value is False or isinstance(value, np.bool_) and not bool(value):
        return False
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError("Expected a saved Boolean value, got " + repr(value))


def _final_saved_predictions(context, task, classes):
    folder = context["root"] / "runs" / (task + "_filename_grouped_seed42")
    prediction_path = _final_record(context, folder / "test_predictions.csv")
    record = _polish_json(_final_record(context, folder / "evaluation_complete.json"))
    _final_require(
        record.get("evaluation_complete") is True,
        "Primary evaluation is incomplete: " + task,
    )
    _final_require(
        context["ledger"][str(prediction_path)] == record["test_predictions.csv"],
        "Saved primary prediction checksum differs: " + task,
    )
    predictions = pd.read_csv(prediction_path, keep_default_na=False)
    _final_require(
        len(predictions) > 0 and not predictions.image_id.duplicated().any(),
        "Primary test identifiers are empty or duplicated: " + task,
    )
    _final_require(
        predictions.y_true.isin([0, 1]).all()
        and predictions.predicted_index.isin([0, 1]).all(),
        "Primary predictions contain an invalid class index: " + task,
    )
    y = predictions.y_true.to_numpy(dtype=int)
    probabilities = predictions.probability.to_numpy(dtype=float)
    predicted = predictions.predicted_index.to_numpy(dtype=int)
    threshold = float(record["threshold"])
    _final_require(
        np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all(),
        "Primary probabilities are invalid: " + task,
    )
    _final_require(
        np.isfinite(threshold)
        and 0 <= threshold <= 1
        and np.array_equal(predicted, (probabilities >= threshold).astype(int)),
        "Primary predictions disagree with the saved threshold: " + task,
    )
    _final_require(
        predictions.threshold.nunique() == 1
        and np.isclose(
            float(predictions.threshold.iloc[0]), threshold, atol=1e-12, rtol=0
        ),
        "Primary per-image thresholds differ: " + task,
    )
    _final_require(
        predictions.positive_class.eq(classes[1]).all()
        and np.array_equal(predictions.label.to_numpy(), np.asarray(classes)[y]),
        "Saved dataset labels disagree with primary class order: " + task,
    )
    cm = confusion_matrix(y, predicted, labels=[0, 1])
    auc = float(roc_auc_score(y, probabilities))
    accuracy = float(accuracy_score(y, predicted))
    _final_require(
        np.isclose(auc, record["metrics"]["roc_auc"], atol=1e-12, rtol=0)
        and np.isclose(accuracy, record["metrics"]["accuracy"], atol=1e-12, rtol=0),
        "Reconstructed primary metrics disagree with saved evaluation: " + task,
    )
    return folder, predictions, record, cm, auc, accuracy


def _final_performance_figure(context, directory, figures, task, name, classes):
    folder, predictions, record, cm, auc, accuracy = _final_saved_predictions(
        context, task, classes
    )
    threshold = float(record["threshold"])
    fpr, tpr, _ = roc_curve(predictions.y_true, predictions.probability)
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1, 1.08]}
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.26, top=0.89, wspace=0.60)
    ax = axes[0]
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    for (i, j), count in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            str(count),
            ha="center",
            va="center",
            fontsize=12,
            color="white" if count > cm.max() / 2 else "#1e3044",
        )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted dataset label",
        ylabel="Supplied dataset label",
    )
    ax.tick_params(axis="both", length=0)
    fig.text(0.035, 0.94, "A", fontsize=12)
    ax.text(
        0.5,
        -0.31,
        f"Threshold = {threshold:.2f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    ax = axes[1]
    ax.plot(fpr, tpr, lw=1.4, color="#235b7f", label=f"AUROC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="#8a8a8a")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    ax.legend(loc="lower right", frameon=False)
    fig.text(0.565, 0.94, "B", fontsize=12)
    caption = (
        f'{task.replace("task", "Task ")} filename-grouped seed-42 held-out performance. '
        f"(A) Confusion matrix at the unchanged tuning-selected threshold of {threshold:.2f}. "
        f"(B) ROC curve reconstructed from the saved test probabilities. The positive dataset class is {classes[1]}. "
        f"The test set contains {len(predictions):,} images; accuracy is {accuracy * 100:.2f}% and AUROC is {auc:.4f}. "
        "Confusion-matrix rows are supplied labels and columns are predictions. No model inference or new uncertainty "
        "estimation was performed. Filename grouping does not establish independently verified patient identity."
    )
    _polish_save(fig, directory, name, caption, figures)
    numeric = dict(
        run_id=folder.name,
        n_test=len(predictions),
        threshold=threshold,
        positive_class=classes[1],
        class_order=classes,
        confusion_matrix=cm.tolist(),
        accuracy=accuracy,
        roc_auc=auc,
        test_predictions_sha256=context["ledger"][
            str((folder / "test_predictions.csv").resolve())
        ],
    )
    return numeric, (folder, predictions, record)


def _final_comparison_figure(context, directory, figures):
    results = pd.read_csv(
        _final_record(context, context["root"] / "all_run_results.csv")
    )
    controlled = results[
        (results.task == "task2")
        & results.design.isin(["filename_grouped", "image_level"])
    ].copy()
    seeds = [17, 29, 42, 71, 101]
    _final_require(
        len(controlled) == 10
        and set(controlled.seed) == set(seeds)
        and not controlled.duplicated(["seed", "design"]).any(),
        "The controlled comparison must contain exactly five saved seed pairs.",
    )
    paired_source = pd.read_csv(
        _final_record(context, context["root"] / "task2_paired_seed_differences.csv")
    )
    _final_require(
        len(paired_source) == 5
        and not paired_source.seed.duplicated().any()
        and set(paired_source.seed) == set(seeds),
        "Saved paired-difference seed coverage differs.",
    )
    paired_source = paired_source.set_index("seed").loc[seeds]
    colors = ["#235b7f", "#bf7140", "#518b77", "#8062a6", "#a44f67"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.24, top=0.91, wspace=0.36)
    comparison = {}
    for ax, metric, panel, label, limits in zip(
        axes,
        ["accuracy", "roc_auc"],
        ["A", "B"],
        ["Accuracy", "AUROC"],
        [(0.75, 0.805), (0.815, 0.87)],
    ):
        paired = controlled.pivot(index="seed", columns="design", values=metric).loc[
            seeds, ["filename_grouped", "image_level"]
        ]
        _final_require(
            np.isfinite(paired.to_numpy()).all(), "Controlled metrics are not finite."
        )
        _final_require(
            np.allclose(
                paired.filename_grouped - paired.image_level,
                paired_source["filename_group_minus_image_" + metric],
                rtol=0,
                atol=1e-12,
            ),
            "Controlled metrics disagree with saved paired differences: " + metric,
        )
        _final_require(
            (paired.to_numpy() >= limits[0]).all()
            and (paired.to_numpy() <= limits[1]).all(),
            "Controlled metrics fall outside the verified figure scale: " + metric,
        )
        for seed, color in zip(seeds, colors):
            ax.plot(
                [0, 1],
                paired.loc[seed].to_numpy(),
                marker="o",
                markersize=4,
                lw=1.1,
                color=color,
                label=str(seed),
            )
        means = paired.mean()
        ax.scatter(
            [0, 1], means.to_numpy(), marker="D", s=29, color="#171717", zorder=5
        )
        ax.set(
            xlim=(-0.18, 1.18),
            ylim=limits,
            xticks=[0, 1],
            xticklabels=["Filename-grouped", "Image-level"],
            ylabel=label,
        )
        ax.grid(axis="y", color="#d9dfe3", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.text(-0.20, 1.08, panel, transform=ax.transAxes, fontsize=12)
        comparison[metric] = dict(
            filename_grouped_mean=float(means.filename_grouped),
            image_level_mean=float(means.image_level),
            mean_image_minus_grouped=float(
                (paired.image_level - paired.filename_grouped).mean()
            ),
            per_seed={
                str(seed): {
                    "filename_grouped": float(paired.loc[seed, "filename_grouped"]),
                    "image_level": float(paired.loc[seed, "image_level"]),
                }
                for seed in seeds
            },
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=5,
        frameon=False,
        title="Seed",
    )
    caption = (
        "Controlled Task 2 comparison of (A) accuracy and (B) AUROC across five paired seed values. "
        "Each colored line joins filename-grouped and image-level evaluations for the same seed (17, 29, 42, 71 or 101). "
        "Black diamonds denote the arithmetic mean of the five saved run metrics. The two designs use different held-out "
        "partitions; these are paired-seed descriptive comparisons, not patient-matched samples or independent cross-validation "
        "folds. The original-folder comparator is excluded. No inferential significance or causal patient-leakage effect is claimed."
    )
    _polish_save(fig, directory, "Fig5", caption, figures)
    controlled.to_csv(directory / "Fig5_plotting_data.csv", index=False)
    return comparison


def _final_occlusion_figure(context, directory, figures, task, name, classes, primary):
    folder, predictions, evaluation = primary
    source = folder / "occlusion"
    selected = pd.read_csv(
        _final_record(context, source / "occlusion_examples.csv"), keep_default_na=False
    )
    _final_require(
        len(selected) == 4 and not selected.image_id.duplicated().any(),
        "Expected exactly four distinct saved occlusion examples: " + task,
    )
    truth = predictions.set_index("image_id")
    _final_require(
        selected.image_id.isin(truth.index).all(),
        "Occlusion example is absent from primary held-out predictions: " + task,
    )
    # Check the recorded selections against the seeded rule. The plotted images
    # still come from the four saved examples.
    expected = []
    pool_sizes = []
    for label in (0, 1):
        for correct in (True, False):
            pool = predictions[
                (predictions.y_true == label)
                & ((predictions.y_true == predictions.predicted_index) == correct)
            ].sort_values("image_id")
            _final_require(
                len(pool) > 0, "A recorded occlusion selection stratum is empty."
            )
            expected.append(pool.sample(1, random_state=42).iloc[0].image_id)
            pool_sizes.append(len(pool))
    _final_require(
        selected.image_id.tolist() == expected,
        "Occlusion examples do not match the originally recorded seed-42 selection.",
    )
    consistency = None
    if task == "task1":
        consistency = pd.read_csv(
            _final_record(context, source / "inference_consistency.csv")
        )
        _final_require(
            len(consistency) == 4
            and not consistency.image_id.duplicated().any()
            and set(consistency.image_id) == set(selected.image_id),
            "Task 1 saved inference-consistency coverage differs.",
        )
        _final_require(
            consistency.probability_matches.map(_final_bool).all()
            and consistency.classification_matches.map(_final_bool).all(),
            "Task 1 saved inference-consistency checks did not all pass.",
        )
        _final_require(
            consistency.physical_batch_size.eq(32).all()
            and consistency.inference_revision.eq(
                "fixed_image_tta_v2_padded_physical_batch"
            ).all(),
            "Task 1 saved inference procedure differs from the verified export.",
        )
        consistency = consistency.set_index("image_id")
    assets = directory / "extracted_saved_pdf_assets"
    assets.mkdir(exist_ok=True)
    provenance = []
    fig = plt.figure(figsize=(7.2, 5.4))
    for i, row in enumerate(selected.to_dict("records")):
        metadata_path = _final_record(context, source / f"example_{i + 1}.json")
        metadata = _polish_json(metadata_path)
        for field in (
            "cache_key",
            "image_id",
            "inference",
            "patch",
            "positive_class",
            "selection_pool_size",
            "selection_seed",
            "stride",
            "target_label",
        ):
            _final_require(
                metadata[field] == row[field],
                "Saved occlusion metadata differs: " + task + "/" + field,
            )
        _final_require(
            _final_bool(metadata["correct"]) == _final_bool(row["correct"])
            and np.isclose(
                float(metadata["probability"]),
                float(row["probability"]),
                atol=2e-5,
                rtol=0,
            )
            and np.isclose(
                float(metadata["threshold"]),
                float(row["threshold"]),
                atol=1e-12,
                rtol=0,
            ),
            "Saved occlusion numeric metadata differs: " + task,
        )
        recorded = truth.loc[row["image_id"]]
        correct = bool(recorded.y_true == recorded.predicted_index)
        _final_require(
            correct == _final_bool(row["correct"])
            and np.isclose(
                float(row["probability"]),
                float(recorded.probability),
                atol=2e-5,
                rtol=0,
            )
            and np.isclose(
                float(row["threshold"]), float(recorded.threshold), atol=1e-12, rtol=0
            ),
            "Occlusion example disagrees with saved primary predictions: "
            + row["image_id"],
        )
        _final_require(
            row["target_label"] == classes[int(recorded.predicted_index)]
            and row["positive_class"] == classes[1]
            and row["patch"] == 48
            and row["stride"] == 48
            and row["selection_seed"] == 42
            and row["selection_pool_size"] == pool_sizes[i],
            "Occlusion class, perturbation or selection metadata differs.",
        )
        _final_require(
            row["inference"]
            == ("fixed TTA, three passes" if task == "task1" else "single pass"),
            "Occlusion inference description differs from the saved primary procedure.",
        )
        if task == "task1":
            _final_require(
                row.get("selected_weights_sha256") == evaluation["signature"]["weights"]
                and row.get("inference_revision")
                == "fixed_image_tta_v2_padded_physical_batch",
                "Task 1 saved occlusion weights/procedure provenance differs.",
            )
            check = consistency.loc[row["image_id"]]
            _final_require(
                np.isclose(
                    float(check.stored_probability),
                    float(recorded.probability),
                    atol=1e-12,
                    rtol=0,
                )
                and np.isclose(
                    float(check.reproduced_probability),
                    float(recorded.probability),
                    atol=2e-5,
                    rtol=0,
                )
                and np.isclose(
                    float(check.absolute_difference),
                    abs(
                        float(check.stored_probability)
                        - float(check.reproduced_probability)
                    ),
                    atol=1e-12,
                    rtol=0,
                )
                and int(
                    float(check.reproduced_probability) >= float(recorded.threshold)
                )
                == int(recorded.predicted_index),
                "Task 1 saved inference-consistency values disagree with predictions.",
            )
        pdf = _final_record(context, source / f"example_{i + 1}.pdf")
        with pymupdf.open(pdf) as document:
            _final_require(
                document.page_count == 1, "Expected a single saved occlusion PDF page."
            )
            objects = sorted(
                document[0].get_image_info(xrefs=True), key=lambda item: item["bbox"][0]
            )
            _final_require(
                len(objects) == 2 and objects[0]["bbox"][2] < objects[1]["bbox"][0],
                "Saved occlusion PDF must contain exactly the original/overlay raster pair.",
            )
            original_images = []
            for k, obj in enumerate(objects):
                _final_require(
                    obj["xref"] > 0,
                    "Saved occlusion raster has no extractable PDF image object.",
                )
                item = document.extract_image(obj["xref"])
                out = assets / f'{name}_{i + 1}_{k + 1}.{item["ext"]}'
                out.write_bytes(item["image"])
                with PILImage.open(io.BytesIO(item["image"])) as pixels:
                    _final_require(
                        pixels.size == (obj["width"], obj["height"]),
                        "Extracted raster dimensions differ from the saved PDF.",
                    )
                    original_images.append(np.asarray(pixels.convert("RGB")).copy())
                provenance.append(
                    dict(
                        figure=name,
                        panel="ABCD"[i],
                        kind=["original", "saved_overlay"][k],
                        image_id=row["image_id"],
                        source_pdf=pdf.relative_to(context["root"]).as_posix(),
                        source_pdf_sha256=context["ledger"][str(pdf)],
                        source_xref=obj["xref"],
                        source_metadata=metadata_path.relative_to(
                            context["root"]
                        ).as_posix(),
                        source_metadata_sha256=context["ledger"][str(metadata_path)],
                        width_pixels=obj["width"],
                        height_pixels=obj["height"],
                        extracted_file=out.relative_to(directory).as_posix(),
                        extracted_sha256=_polish_hash(out),
                        processing="Lossless extraction of the full embedded raster; no cropping or heatmap recalculation",
                        true_label=classes[int(recorded.y_true)],
                        predicted_label=classes[int(recorded.predicted_index)],
                        positive_probability=float(row["probability"]),
                        threshold=float(row["threshold"]),
                        correct=correct,
                    )
                )
        x = 0.035 if i % 2 == 0 else 0.53
        y = 0.95 if i < 2 else 0.47
        status = "correct" if correct else "incorrect"
        fig.text(
            x,
            y,
            f'{"ABCD"[i]}  Supplied: {classes[int(recorded.y_true)]}; {status}',
            fontsize=9,
        )
        fig.text(
            x,
            y - 0.042,
            f"Prediction: {classes[int(recorded.predicted_index)]}; "
            f'P({classes[1]}) = {row["probability"]:.4f}',
            fontsize=8.5,
        )
        for k, pixels in enumerate(original_images):
            ax = fig.add_axes([x + k * 0.221, y - 0.406, 0.217, 0.31])
            ax.imshow(pixels, interpolation="nearest", aspect="equal")
            ax.axis("off")
            ax.set_title(["Original", "Saved occlusion overlay"][k], fontsize=8, pad=4)
    caption = (
        f'{task.replace("task", "Task ")} seed-42 filename-grouped held-out occlusion examples. '
        "Panels A–D retain the original one-per-stratum selections: supplied negative class/correct, negative/incorrect, "
        "positive/correct, and positive/incorrect. Each pair shows the original model input and its saved occlusion overlay. "
        f"The positive dataset class is {classes[1]}; the tuning-selected threshold is {selected.threshold.iloc[0]:.2f}. "
        "Selection used random seed 42 within each class-by-correctness pool after sorting image identifiers. "
        "A 48-pixel block with 48-pixel stride was filled with the image median. The overlay shows the nonnegative "
        "decrease in predicted-class probability, normalized by the maximum decrease within that image; warm colors "
        "indicate larger relative decreases. Colors therefore cannot be compared as absolute effect sizes across images. "
        + (
            "The same three fixed augmentation passes and physical prediction batch size used for Task 1 evaluation "
            "were used to generate these saved overlays. "
            if task == "task1"
            else "The same single-pass inference procedure used for Task 2 evaluation generated these saved overlays. "
        )
        + "The original raster image pairs were extracted losslessly from saved PDFs and recomposed; no new model "
        "inference, image selection or occlusion calculation occurred. The panels are model perturbation illustrations, "
        "not validated anatomical localization or evidence of confirmed etiology. Raster resolution is limited to the "
        "previously exported assets; a 300-DPI container does not restore source image detail."
    )
    _polish_save(fig, directory, name, caption, figures)
    return provenance


def draw_final_result_figures(context, directory, figures):
    """Create Fig3–Fig7 and return the numerical checks and source records.

    The context contains the evidence directory and input hashes. The
    figure list already holds the sample panel and workflow."""
    context["root"] = Path(context["root"]).resolve()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    _final_require(
        not any(
            item["figure"] in {"Fig3", "Fig4", "Fig5", "Fig6", "Fig7"}
            for item in figures
        ),
        "Fig3–Fig7 already exist in this export manifest.",
    )
    numeric = {}
    primary = {}
    provenance = []
    # Set the resolution of rasters embedded in the PDF.
    # TIFF and PNG exports are also rendered at 300 DPI.
    with matplotlib.rc_context(
        {
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for task, name, classes in [
            ("task1", "Fig3", ["NORMAL", "PNEUMONIA"]),
            ("task2", "Fig4", ["BACTERIAL", "VIRAL"]),
        ]:
            numeric[name], primary[task] = _final_performance_figure(
                context, directory, figures, task, name, classes
            )
        numeric["Fig5"] = _final_comparison_figure(context, directory, figures)
        for task, name, classes in [
            ("task1", "Fig6", ["NORMAL", "PNEUMONIA"]),
            ("task2", "Fig7", ["BACTERIAL", "VIRAL"]),
        ]:
            provenance.extend(
                _final_occlusion_figure(
                    context, directory, figures, task, name, classes, primary[task]
                )
            )
    pd.DataFrame(provenance).to_csv(
        directory / "Fig6_Fig7_image_provenance.csv", index=False
    )
    changed = [
        path
        for path, digest in context["ledger"].items()
        if _polish_hash(path) != digest
    ]
    _final_require(
        not changed,
        "Saved source evidence changed while drawing final figures: " + str(changed),
    )
    return dict(
        numeric_checks=numeric,
        occlusion_composite_provenance=provenance,
        occlusion_selection_and_correctness_verified_against_saved_predictions=True,
        task1_saved_inference_consistency_verified=True,
        occlusion_heatmaps_recalculated=False,
        occlusion_source_resolution_limited=True,
        source_evidence_unchanged=True,
        training_performed=False,
        inference_performed=False,
        bootstrap_recalculated=False,
        native_model_revalidation_performed=False,
        scientific_data_changed=False,
    )


# %% [markdown]
# ### 4.5. Saved tables and results overview


# %%
"""Copy saved tables and assemble the figure package."""


def _final_record(context, path):
    path = Path(path).resolve()
    digest = _polish_hash(path)
    old = context["ledger"].get(str(path))
    if old is not None and old != digest:
        raise RuntimeError("Saved evidence changed during export: " + str(path))
    context["ledger"][str(path)] = digest
    return path


def _final_copy(context, source, destination):
    source = _final_record(context, source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if _polish_hash(destination) != context["ledger"][str(source)]:
        raise RuntimeError("Copied evidence checksum mismatch: " + str(source))


def _final_verify_artifacts(context):
    """Validate the recorded reporting files without opening model checkpoints."""
    root = context["root"]
    artifacts = pd.read_csv(root / "artifact_verification.csv")
    for row in artifacts.itertuples():
        relative = PurePosixPath(str(row.artifact).replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in relative.parts[0]:
            raise ValueError("Unsafe artifact path in saved manifest.")
        source = root.joinpath(*relative.parts).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Saved artifact is outside the results folder.")
        if source.suffix.lower() not in {
            ".csv",
            ".json",
            ".md",
            ".txt",
            ".png",
            ".pdf",
        }:
            raise ValueError(
                "Unexpected model/binary artifact in formatting evidence: "
                + str(relative)
            )
        _final_record(context, source)
        if context["ledger"][str(source)] != row.sha256 or source.stat().st_size != int(
            row.bytes
        ):
            raise ValueError(
                "Saved artifact no longer matches its record: " + str(relative)
            )
    print(
        f"All 16 prediction records and {len(artifacts)} recorded evidence files verified."
    )
    return len(artifacts)


def _final_check_descriptive_tables(context):
    """Check descriptive summaries against the stored run metrics."""
    root = context["root"]
    results = context["results"]
    summary = pd.read_csv(_final_record(context, root / "repeated_split_summary.csv"))
    expected = {
        ("task1", "filename_grouped"),
        ("task2", "filename_grouped"),
        ("task2", "image_level"),
    }
    if len(summary) != 3 or set(zip(summary.task, summary.design)) != expected:
        raise ValueError(
            "Repeated-run table does not contain the three expected five-run summaries."
        )
    metrics = ("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc")
    for row in summary.to_dict("records"):
        selected = results[
            (results.task == row["task"]) & (results.design == row["design"])
        ]
        if len(selected) != 5 or set(selected.seed) != {17, 29, 42, 71, 101}:
            raise ValueError(
                "Repeated-run seed coverage differs from the recorded schedule."
            )
        for metric in metrics:
            actual = selected[metric].agg(["count", "mean", "std", "min", "max"])
            for statistic, value in actual.items():
                if not np.isclose(
                    float(value),
                    float(row[f"{metric}_{statistic}"]),
                    atol=1e-10,
                    rtol=0,
                ):
                    raise ValueError(
                        "Saved descriptive summary disagrees with run metrics."
                    )
    paired = pd.read_csv(
        _final_record(context, root / "task2_paired_seed_differences.csv")
    )
    if len(paired) != 5 or set(paired.seed) != {17, 29, 42, 71, 101}:
        raise ValueError("Paired comparison seed coverage is incomplete.")
    controlled = results[
        (results.task == "task2")
        & results.design.isin(["filename_grouped", "image_level"])
    ]
    for metric in metrics:
        pivot = controlled.pivot(index="seed", columns="design", values=metric)
        observed = paired.set_index("seed")[
            f"filename_group_minus_image_{metric}"
        ].sort_index()
        expected_values = (pivot.filename_grouped - pivot.image_level).sort_index()
        if not np.allclose(
            observed.to_numpy(), expected_values.to_numpy(), atol=1e-10, rtol=0
        ):
            raise ValueError("Saved paired differences disagree with run metrics.")
    return summary


def _final_result_overview(context, directory, summary):
    """Format the saved estimates and intervals for the results overview."""
    root = context["root"]
    primary_rows = []
    names = {
        "task1_filename_grouped_seed42": "Task 1: filename-grouped, seed 42",
        "task2_filename_grouped_seed42": "Task 2: filename-grouped, seed 42",
        "task2_original_folders_seed42": "Task 2: original-folder comparator, seed 42",
    }
    for run_id, title in names.items():
        run = context["results"].set_index("run_id").loc[run_id]
        intervals = pd.read_csv(
            _final_record(context, root / "runs" / run_id / "bootstrap_intervals.csv")
        ).set_index("metric")
        for metric in intervals.index:
            if not np.isclose(
                float(intervals.loc[metric, "estimate"]),
                float(run[metric]),
                atol=1e-10,
                rtol=0,
            ):
                raise ValueError(
                    "Stored interval estimate differs from the saved run metric: "
                    + run_id
                )
        a, r = intervals.loc["accuracy"], intervals.loc["roc_auc"]
        primary_rows.append(
            {
                "Evaluation": title,
                "Threshold": f"{run.threshold:.2f}",
                "Accuracy, % (95% interval)": f"{100*a.estimate:.2f} ({100*a.lower_95:.2f}–{100*a.upper_95:.2f})",
                "AUROC (95% interval)": f"{r.estimate:.4f} ({r.lower_95:.4f}–{r.upper_95:.4f})",
            }
        )
    repeated_rows = []
    for row in summary.itertuples():
        repeated_rows.append(
            {
                "Evaluation": row.task.replace("task", "Task ")
                + ": "
                + row.design.replace("_", "-"),
                "Runs": int(row.accuracy_count),
                "Accuracy, % (mean ± SD)": f"{100*row.accuracy_mean:.2f} ± {100*row.accuracy_std:.2f}",
                "AUROC (mean ± SD)": f"{row.roc_auc_mean:.4f} ± {row.roc_auc_std:.4f}",
            }
        )
    primary = pd.DataFrame(primary_rows)
    repeated = pd.DataFrame(repeated_rows)
    primary.to_csv(directory / "primary_results_formatted.csv", index=False)
    repeated.to_csv(directory / "repeated_results_formatted.csv", index=False)
    page = """<!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Completed pneumonia evaluation</title>
      <style>
        /* Keep the report readable when embedded in a dark Colab output. */
        #pneumonia-results {
          box-sizing: border-box;
          color-scheme: light;
          background: #ffffff !important;
          color: #18283b !important;
          font-family: Arial, Helvetica, sans-serif !important;
          font-size: 16px;
          line-height: 1.6;
          max-width: 1200px;
          width: 100%;
          margin: 16px auto;
          padding: 24px;
          border: 1px solid #d3dce6;
          border-radius: 8px;
          text-align: left;
        }
        #pneumonia-results h1,
        #pneumonia-results h2,
        #pneumonia-results p {
          color: #18283b !important;
          background: transparent !important;
          font-family: Arial, Helvetica, sans-serif !important;
          text-align: left !important;
          opacity: 1 !important;
        }
        #pneumonia-results h1 {
          font-size: 26px !important;
          line-height: 1.3;
          margin: 0 0 16px;
        }
        #pneumonia-results h2 {
          font-size: 21px !important;
          line-height: 1.4;
          margin: 28px 0 14px;
        }
        #pneumonia-results p {
          font-size: 16px !important;
          margin: 12px 0;
        }
        #pneumonia-results .results-table-scroll {
          max-width: 100%;
          overflow-x: auto;
          background: #ffffff !important;
          margin: 0 0 24px;
        }
        #pneumonia-results table {
          width: 100%;
          min-width: 760px;
          border-collapse: collapse;
          border-spacing: 0;
          margin: 0 !important;
          border: 0;
          background: #ffffff !important;
          color: #18283b !important;
          font-variant-numeric: tabular-nums;
        }
        #pneumonia-results table th,
        #pneumonia-results table td {
          font-family: Arial, Helvetica, sans-serif !important;
          font-size: 15px !important;
          line-height: 1.5;
          color: #18283b !important;
          background: #ffffff !important;
          padding: 12px 14px;
          border: 0;
          border-bottom: 1px solid #d3dce6;
          text-align: right !important;
          vertical-align: top;
          opacity: 1 !important;
          white-space: nowrap;
        }
        #pneumonia-results table thead th {
          color: #ffffff !important;
          background: #22364d !important;
          font-weight: 700;
          white-space: normal;
        }
        #pneumonia-results table th:first-child,
        #pneumonia-results table td:first-child {
          text-align: left !important;
          white-space: normal;
          min-width: 235px;
          width: 36%;
        }
        #pneumonia-results table tbody tr:nth-child(even) td {
          color: #18283b !important;
          background: #f1f5f9 !important;
        }
        #pneumonia-results p.results-note {
          color: #18283b !important;
          background: #edf3f8 !important;
          border-left: 4px solid #426786;
          padding: 16px;
        }
        @media (max-width: 600px) {
          #pneumonia-results { padding: 16px; }
          #pneumonia-results h1 { font-size: 23px !important; }
          #pneumonia-results h2 { font-size: 19px !important; }
        }
      </style>
    </head>
    <body>
    <section id="pneumonia-results" aria-labelledby="pneumonia-results-title">
      <h1 id="pneumonia-results-title">Completed filename-grouped evaluation</h1>
      <p>16 saved model runs across five seed values. This export reads existing predictions, thresholds and intervals.
         It performs no training, inference or bootstrap calculation.</p>
      <h2>Primary seed-42 results and separate original-folder comparator</h2>
      <div class="results-table-scroll" role="region" aria-label="Primary results" tabindex="0">
        {{PRIMARY_TABLE}}
      </div>
      <h2>Descriptive summaries of five repeated holdouts</h2>
      <div class="results-table-scroll" role="region" aria-label="Repeated holdout results" tabindex="0">
        {{REPEATED_TABLE}}
      </div>
      <p>Intervals are the saved filename-group bootstrap intervals (2,000 attempted replicates).
         The five-run SD describes variation across repeated holdouts; it is not a confidence interval.
         Task 1 and Task 2 use different pipelines. The original-folder comparator is separate from the controlled comparison.</p>
      <p class="results-note">{{IDENTITY_NOTE}} BACTERIAL and VIRAL are supplied dataset labels;
         they do not establish confirmed etiology. Differences cannot establish a causal patient-leakage effect.</p>
    </section>
    </body>
    </html>
    """
    page = page.replace(
        "{{PRIMARY_TABLE}}", primary.to_html(index=False, border=0, escape=True)
    )
    page = page.replace(
        "{{REPEATED_TABLE}}", repeated.to_html(index=False, border=0, escape=True)
    )
    page = page.replace("{{IDENTITY_NOTE}}", html.escape(IDENTITY_NOTE))
    (directory / "RESULTS_OVERVIEW.html").write_text(page, encoding="utf-8")


# %% [markdown]
# ### 4.6. Export assembly


# %%
def run_publication_polish(config):
    start = time.monotonic()
    print(
        "Reading completed evidence from Drive. No model loading or GPU work is enabled."
    )
    context = verify_polish_inputs(config)
    artifact_count = _final_verify_artifacts(context)
    summary = _final_check_descriptive_tables(context)
    root = context["root"]
    directory = (
        root
        / "publication_polish"
        / (time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:6])
    )
    directory.mkdir(parents=True)
    figures = []
    samples = []
    for row in context["samples"]:
        with PILImage.open(row["path"]) as image:
            width, height = image.size
        samples.append(
            {k: str(v) for k, v in row.items() if k != "path"}
            | {
                "width_pixels": width,
                "height_pixels": height,
                "processing": "Full original image; original aspect ratio; no enhancement or cropping",
            }
        )
    print("Formatting Fig1–Fig2: selected original radiographs and workflow.")
    sample_caption = (
        "Example chest radiographs with supplied dataset labels: (A) normal, (B) bacterial pneumonia, "
        "and (C) viral pneumonia. The original recorded selections are retained. These images illustrate dataset "
        "categories, not microbiologically confirmed etiology or representative diagnostic appearance. Full fields "
        "of view and original aspect ratios are preserved without enhancement or cropping. Original image bytes "
        "match the saved dataset-manifest checksums; source dimensions are recorded in Fig1_image_provenance.csv."
    )
    sample_figure = draw_sample_panel(context["samples"])
    sample_figure.set_size_inches(7.2, 2.25)
    _polish_save(sample_figure, directory, "Fig1", sample_caption, figures)
    workflow_caption = (
        f'Analysis workflow for {len(context["manifest"]):,} chest radiographs. Task 1 uses five '
        "filename-grouped runs. The controlled Task 2 comparison uses five matched seed values across "
        "filename-grouped and image-level designs, plus one separate original-folder comparator. Tuning data "
        "determined checkpoint and threshold selection. The grouped seed-42 models provide the main figures "
        "and intervals. Repeated-run comparisons are descriptive. Saved primary/comparator intervals use "
        "2,000 attempted filename-group bootstrap replicates. "
        + IDENTITY_NOTE
        + " Figure 2 is separate from the S1 checklist."
    )
    _polish_save(
        draw_workflow(len(context["manifest"])),
        directory,
        "Fig2",
        workflow_caption,
        figures,
    )
    pd.DataFrame(samples).to_csv(directory / "Fig1_image_provenance.csv", index=False)
    print(
        "Formatting Fig3–Fig7 from saved probabilities, metrics and occlusion assets."
    )
    figure_metadata = draw_final_result_figures(context, directory, figures)
    if [item["figure"] for item in figures] != [f"Fig{i}" for i in range(1, 8)]:
        raise RuntimeError("Seven-figure export is incomplete.")
    print(
        "Copying saved tables, audit counts, split assignments, predictions and training histories."
    )
    tables = directory / "tables"
    for name in (
        "all_run_results.csv",
        "repeated_split_summary.csv",
        "task2_paired_seed_differences.csv",
        "split_counts_all_runs.csv",
        "reviewer_requirement_checklist.csv",
        "artifact_verification.csv",
    ):
        _final_copy(context, root / name, tables / name)
    for path in sorted((root / "audit").iterdir()):
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
            _final_copy(context, path, tables / "audit" / path.name)
    for run in _viewer_runs():
        folder = root / "runs" / run["run_id"]
        for name in (
            "run_configuration.json",
            "training_specification.json",
            "training_history.csv",
            "training_complete.json",
            "evaluation_complete.json",
            "split_assignments.csv",
            "split_counts.csv",
            "partition_overlaps.csv",
            "tuning_predictions.csv",
            "test_predictions.csv",
            "test_metrics.csv",
            "threshold_sensitivity.csv",
        ):
            _final_copy(context, folder / name, tables / "runs" / run["run_id"] / name)
        if (folder / "bootstrap_intervals.csv").is_file():
            _final_copy(
                context,
                folder / "bootstrap_intervals.csv",
                tables / "runs" / run["run_id"] / "bootstrap_intervals.csv",
            )
    for name in (
        "protocol.json",
        "dataset_fingerprint.json",
        "completion_status.json",
        "reporting_patch.json",
        "native_recovery_self_test.json",
        "METHODS_AND_LIMITATIONS.md",
    ):
        _final_copy(context, root / name, directory / "recorded_analysis" / name)
    _final_result_overview(context, directory, summary)
    (directory / "FIGURE_CAPTIONS.md").write_text(
        "\n\n".join(
            "## " + item["figure"] + "\n\n" + item["caption"] for item in figures
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "SCIENTIFIC_WORDING.md").write_text(
        "# Scientific interpretation\n\n" + IDENTITY_NOTE + "\n\n"
        "The grouping rule assigns person123_bacteria and person123_virus to separate subtype-qualified keys. "
        "Neither their constructed purity nor overlap of raw person123 prefixes independently establishes patient identity. "
        "Exact same-label decoded-image copies are linked for generated grouped partitions. Original-folder assignments "
        "remain separately audited. Raw-prefix overlap can remain even when qualified-key and duplicate overlap is zero.\n\n"
        "BACTERIAL and VIRAL are supplied filename-derived labels, not a uniform microbiological reference standard. "
        "Five repeated holdouts are not conventional five-fold cross-validation. Tuning data alone selected thresholds. "
        "Replication oversampling was disabled; class weights were used. Task 1 and Task 2 comparisons are descriptive "
        "because the pipelines differ. The original-folder comparator cannot isolate a causal patient-leakage effect.\n\n"
        "This formatting pass preserves all saved scientific evidence. Public repository publication, its real URL, "
        "the S1 checklist and final submission checks remain author actions.\n",
        encoding="utf-8",
    )
    changed = [
        path
        for path, digest in context["ledger"].items()
        if _polish_hash(path) != digest
    ]
    if changed:
        raise RuntimeError(
            "Saved input changed while exporting; do not use this export: "
            + str(changed)
        )
    report = dict(
        formatting_complete=True,
        analysis_complete_recorded=True,
        prediction_evidence_verified=16,
        recorded_artifacts_verified=artifact_count,
        native_model_revalidation_performed=False,
        training_performed=False,
        inference_performed=False,
        bootstrap_recalculated=False,
        patient_identity_verified=False,
        submission_ready=False,
        source_evidence_unchanged=True,
        figures=figures,
        figure_details=figure_metadata,
        source_evidence_sha256=context["ledger"],
        plotting_software={
            name: importlib.metadata.version(name)
            for name in (
                "matplotlib",
                "numpy",
                "pandas",
                "Pillow",
                "PyMuPDF",
                "scikit-learn",
            )
        },
        figure_guidance=FIGURE_GUIDE,
        elapsed_seconds=round(time.monotonic() - start, 1),
    )
    (directory / "polish_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (directory / "README.md").write_text(
        "# Final figures and saved evidence\n\n"
        "Fig1: sample radiographs. Fig2: workflow. Fig3: primary Task 1 confusion matrix and ROC. "
        "Fig4: primary Task 2 confusion matrix and ROC. Fig5: controlled repeated Task 2 comparison. "
        "Fig6: Task 1 saved occlusion examples. Fig7: Task 2 saved occlusion examples.\n\n"
        "Use TIFF exports with the separate captions. All seven exports are RGB at 300 DPI; PNG previews and "
        "PDFs are also supplied. Occlusion resolution is limited by its saved raster assets; exporting at 300 DPI "
        "does not create missing image detail. The existing image selections and overlays are unchanged.\n\n"
        "RESULTS_OVERVIEW.html separates primary seed-42 results, the original-folder comparator and descriptive "
        "five-run summaries. Raw saved tables, audit counts, all 16 split assignments, tuning/test predictions, "
        "threshold sensitivity and training histories are copied without changes. The recorded analysis checklist "
        "is copied as historical evidence; its manuscript-action column is not an audit of later document revisions.\n\n"
        "No model loading, training, inference or bootstrap calculation ran. Original evidence remains in the "
        "parent folder. Patient identities were not independently verified. Inspect figures alongside the final "
        "manuscript; repository publication, S1 completion and journal submission are separate author actions.\n",
        encoding="utf-8",
    )
    archive = directory / "FINAL_FIGURES_AND_TABLES.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != archive:
                bundle.write(path, path.relative_to(directory).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("Export ZIP verification failed.")
    print(
        "COMPLETE: saved evidence verified for 16/16 runs; all seven figures exported."
    )
    print("Download from Google Drive:", archive)
    print(
        "No models were trained or loaded; no predictions or confidence intervals were regenerated."
    )
    return directory


# %% [markdown]
# ## 5. Export figures and tables
# 
# Write the figure package to a new folder under `OUTPUT_ROOT/publication_polish`. The package includes the
# saved audit tables, split assignments, predictions, thresholds and training histories, plus a formatted results
# overview. Reading the individual files from Drive may take a few minutes.


# %%
if MODE != _MODE_AT_START:
    raise RuntimeError("Mode changed mid-session. Restart the session and Run all.")
if MODE == "resume_analysis" and not _ANALYSIS_SUCCEEDED:
    raise RuntimeError(
        "Full analysis did not finish successfully. Final export was not started."
    )
polish_connect(CONFIG)
export_directory = run_publication_polish(CONFIG)


# %% [markdown]
# ## 6. Preview and download
# 
# Display the results overview and the seven manuscript figures. Download `FINAL_FIGURES_AND_TABLES.zip`
# from the printed Drive location. Use the TIFF files and the separate captions for the manuscript figures.


# %%
# Display this export rather than the full saved-image gallery.
if CONFIG.get("SHOW_PREVIEW", True):
    from IPython.display import HTML, Image, display

    display(
        HTML((export_directory / "RESULTS_OVERVIEW.html").read_text(encoding="utf-8"))
    )
    for item in json.loads((export_directory / "polish_manifest.json").read_text())[
        "figures"
    ]:
        print(item["figure"])
        display(
            Image(filename=str(export_directory / (item["figure"] + ".png")), width=900)
        )
print("Open this folder in Google Drive:")
print(export_directory)
print("Download:", export_directory / "FINAL_FIGURES_AND_TABLES.zip")
print(
    "All 16 saved runs were reused. No GPU training or inference ran in final_polish mode."
)


# %% [markdown]
# ## 7. Reporting notes
# 
# Use **filename-grouped evaluation** throughout the manuscript. The keys `person123_bacteria` and
# `person123_virus` belong to separate subtype-qualified groups. Their purity is partly a consequence of the key
# definition; raw-prefix overlap alone also does not establish shared patient identity. Patient identities remain
# unverified. BACTERIAL and VIRAL are supplied dataset labels, not a uniform microbiological reference standard.
# 
# Keep the primary seed-42 estimates, five-run descriptive summaries and original-folder comparator separate.
# Five repeated holdouts are not conventional five-fold cross-validation. Differences between evaluation designs
# do not establish a causal patient-leakage effect.
# 
# Before submission, reconcile the figures and captions with the clean and tracked manuscripts and reviewer
# response. Complete the S1 checklist and provide a real public repository URL. `polish_manifest.json` records
# the export checks; it does not establish patient identity or certify journal acceptance. A partial export without
# a successful manifest and ZIP verification should not be used.
