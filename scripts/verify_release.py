"""Verify the files recorded in RELEASE_MANIFEST.json using only Python's standard library."""

import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for name, record in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            errors.append(f"Unsafe manifest path: {name}")
            continue
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root) or not path.is_file():
            errors.append(f"Missing or unsafe path: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if path.stat().st_size != record["bytes"] or digest.hexdigest() != record["sha256"]:
            errors.append(f"File changed: {name}")
    if errors:
        print("Release verification failed:\n" + "\n".join(errors))
        return 1
    print(f"Verified all {len(manifest['files'])} recorded release files.")
    print("This checks file integrity; it does not establish patient identities or clinical validity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
