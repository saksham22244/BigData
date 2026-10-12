#!/usr/bin/env python3
"""Run the supplied notebook with explicit paths and guarded configuration.

Default final_polish copies the evidence into an empty output directory, then
exports figures using the original dataset. Training requires resume_analysis;
starting a new experiment additionally requires --new-experiment.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil

PACKAGE = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonoverlapping(first, second):
    return not first.is_relative_to(second) and not second.is_relative_to(first)


def configured_cells(notebook, mode, updates):
    """Change only MODE and CONFIG literals in the unique configuration cell."""
    document = json.loads(notebook.read_text(encoding="utf-8-sig"))
    require(document.get("nbformat") == 4, "Expected the supplied version-4 notebook")
    cells, candidates = [], []
    for index, cell in enumerate(document["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        tree = ast.parse(source, filename=f"{notebook}:cell-{index + 1}")
        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in {"MODE", "CONFIG"}:
                    require(target.id not in assignments, "Duplicate configuration assignment")
                    assignments[target.id] = node
        if assignments:
            require(set(assignments) == {"MODE", "CONFIG"},
                    "Unexpected MODE/CONFIG assignment outside the configuration cell")
            candidates.append(len(cells))
            config = ast.literal_eval(assignments["CONFIG"].value)
            require(isinstance(config, dict), "Expected a literal CONFIG dictionary")
            require(ast.literal_eval(assignments["MODE"].value) == "final_polish",
                    "Notebook default mode differs; review the supplied notebook")
            config.update(updates)
            assignments["MODE"].value = ast.Constant(mode)
            assignments["CONFIG"].value = ast.parse(repr(config), mode="eval").body
            tree = ast.fix_missing_locations(tree)
        cells.append((index + 1, tree))
    require(len(candidates) == 1 and candidates[0] == 0,
            "Expected one configuration cell before all other executable cells")
    return cells, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path,
                        help="Original chest_xray directory containing train/test; val is optional")
    parser.add_argument("--output", required=True, type=Path,
                        help="Empty destination for final_polish/new training; existing full experiment for resume")
    parser.add_argument("--mode", choices=("final_polish", "resume_analysis"), default="final_polish")
    parser.add_argument("--new-experiment", action="store_true",
                        help="Explicitly allow new training in resume_analysis mode with an empty output")
    parser.add_argument("--evidence", type=Path, default=PACKAGE / "evidence",
                        help="Saved evidence to copy for final_polish")
    parser.add_argument("--notebook", type=Path, default=PACKAGE / "notebooks/pneumonia_analysis.ipynb")
    parser.add_argument("--local-dataset", type=Path,
                        help="Optional dataset staging/cache directory; defaults inside --output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check paths and configuration; do not write, execute cells, install, or train")
    args = parser.parse_args(argv)
    try:
        for name in ("dataset", "output", "evidence", "notebook"):
            setattr(args, name, getattr(args, name).expanduser().resolve())
        local_dataset = (args.local_dataset.expanduser().resolve() if args.local_dataset
                         else args.output / "dataset_cache")
        require(args.dataset.is_dir(), f"Dataset directory not found: {args.dataset}")
        require(all((args.dataset / name).is_dir() for name in ("train", "test")),
                "--dataset must point to the chest_xray root containing train and test (val is optional)")
        require(nonoverlapping(args.output, args.dataset),
                "Output must not overlap the input dataset")
        require(nonoverlapping(args.output, args.evidence),
                "Output must not overlap the packaged/source evidence")
        require(nonoverlapping(local_dataset, args.dataset)
                and nonoverlapping(local_dataset, args.evidence),
                "Dataset cache must not overlap source dataset or evidence")
        require(local_dataset != args.output and not args.output.is_relative_to(local_dataset),
                "Dataset cache cannot contain or equal the experiment output")
        require(not args.new_experiment or args.mode == "resume_analysis",
                "--new-experiment requires --mode resume_analysis")
        empty = not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir()))
        if args.mode == "final_polish":
            require(empty, "final_polish output must be new or empty; existing files will not be replaced")
            require((args.evidence / "protocol.json").is_file()
                    and (args.evidence / "runs").is_dir(), "Saved evidence is missing")
        elif args.new_experiment:
            require(empty, "New training requires a new or empty --output directory")
        else:
            require((args.output / "protocol.json").is_file()
                    and (args.output / "runs").is_dir(),
                    "Resume requires the original full experiment with protocol/checkpoints. "
                    "Use --new-experiment and empty output for a fresh experiment.")
            require(any(args.output.glob("runs/**/*.weights.h5"))
                    or any(args.output.glob("runs/**/*.keras"))
                    or any(args.output.glob("runs/**/pointer.json")),
                    "No saved model weights/checkpoint pointers found. The evidence-only release cannot resume training.")
        cells, config = configured_cells(args.notebook, args.mode, dict(
            MOUNT_GOOGLE_DRIVE=False, DATASET_SOURCE=str(args.dataset),
            OUTPUT_ROOT=str(args.output), LOCAL_DATASET=str(local_dataset),
            LEGACY_OUTPUT_ROOTS=[], SHOW_PREVIEW=False,
            RESUME_EXISTING_EXPERIMENT=not args.new_experiment,
        ))
        print(json.dumps(dict(mode=args.mode, config=config, dry_run=args.dry_run), indent=2))
        if args.dry_run:
            print("Configuration verified. No files changed and no notebook cells executed.")
            return
        if args.mode == "final_polish":
            shutil.copytree(args.evidence, args.output, dirs_exist_ok=True)
        else:
            print("Running the original GPU analysis/recovery workflow. It may restore packages and request restart.")
        namespace = {"__name__": "__main__", "__file__": str(args.notebook)}
        for number, tree in cells:
            exec(compile(tree, f"{args.notebook}:cell-{number}", "exec"), namespace)
    except (OSError, ValueError, SyntaxError) as exc:
        parser.exit(1, f"Pipeline stopped: {exc}\n")


if __name__ == "__main__":
    main()
