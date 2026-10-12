"""Check a local image tree against the study manifest without changing files."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_COLUMNS = [
    "image_id", "file_sha256", "pixel_sha256", "qualified_key", "split_group",
    "label_task1", "label_task2",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_fingerprint(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=FINGERPRINT_COLUMNS, extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def safe_image_path(root, image_id):
    relative = PurePosixPath(image_id)
    if (
        relative.is_absolute() or ".." in relative.parts
        or "\\" in image_id or ":" in image_id or not relative.parts
    ):
        raise ValueError(f"Unsafe image_id: {image_id!r}")
    path = root.joinpath(*relative.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Image resolves outside the dataset: {image_id}")
    return path


def verify(dataset, evidence, decoded=False):
    dataset, evidence = dataset.resolve(), evidence.resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    with (evidence / "audit" / "manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        if not set(FINGERPRINT_COLUMNS).issubset(reader.fieldnames or []):
            raise ValueError("The manifest is missing fingerprint columns.")
        rows = list(reader)
    if not rows or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("The manifest is empty or contains duplicate image IDs.")
    expected = json.loads((evidence / "dataset_fingerprint.json").read_text())
    actual_fingerprint = manifest_fingerprint(rows)
    if actual_fingerprint != expected["content_sha256"] or len(rows) != expected["images"]:
        raise ValueError("Manifest contents do not match the recorded fingerprint.")

    if decoded:
        from PIL import Image

    issues = []
    checked = 0
    for row in rows:
        image_id = row["image_id"]
        path = safe_image_path(dataset, image_id)
        if not path.is_file():
            issues.append({"image_id": image_id, "reason": "missing"})
            continue
        if sha256_file(path) != row["file_sha256"]:
            issues.append({"image_id": image_id, "reason": "file_sha256 mismatch"})
            continue
        if decoded:
            try:
                with Image.open(path) as opened:
                    rgb = opened.convert("RGB")
                    digest = hashlib.sha256(
                        str(rgb.size).encode() + rgb.tobytes()
                    ).hexdigest()
            except Exception as exc:
                issues.append({"image_id": image_id, "reason": f"decode failure: {exc}"})
                continue
            if digest != row["pixel_sha256"]:
                issues.append({"image_id": image_id, "reason": "pixel_sha256 mismatch"})
                continue
        checked += 1

    expected_ids = {row["image_id"] for row in rows}
    for path in sorted(dataset.rglob("*")):
        relative = path.relative_to(dataset)
        if (
            not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES
            or "__MACOSX" in relative.parts or path.name.startswith("._")
        ):
            continue
        if relative.as_posix() not in expected_ids:
            issues.append({"image_id": relative.as_posix(), "reason": "extra image"})
    return {
        "matches_study_snapshot": not issues,
        "expected_images": len(rows),
        "verified_images": checked,
        "decoded_hashes_checked": decoded,
        "content_sha256": actual_fingerprint,
        "issue_count": len(issues),
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence")
    parser.add_argument("--decoded", action="store_true", help="Also check decoded RGB hashes.")
    args = parser.parse_args()
    try:
        result = verify(args.dataset, args.evidence, args.decoded)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.exit(1, f"Dataset verification failed: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0 if result["matches_study_snapshot"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
