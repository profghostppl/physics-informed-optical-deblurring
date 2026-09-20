"""
Cumulative real-photo dataset archive.

Every time train_on_real_photos.py runs, it first calls sync_from_source() here to
pull any new photos from the user's drop folder (SOURCE_DIR below) into
dataset/sharp/, this project's PERMANENT archive -- so successive training runs
build on every photo the user has ever provided, not just whatever happens to be
sitting in that folder at the moment (which the user has already replaced once).

sync_files() does the same thing for an explicit, curated list of individual files
from anywhere on disk -- used e.g. to pull a hand-picked subset of genuinely dark
photos out of a much larger folder, rather than archiving that whole folder.

Two things this guarantees, for either entry point:
  * Deduplication by content hash, not filename -- camera frame counters can repeat
    (e.g. after a card format, or across different camera bodies), so the same
    filename appearing twice is NOT assumed to be the same photo; only an identical
    SHA-256 is.
  * A photo's train/validation split assignment is decided once, deterministically,
    from its content hash, and never changes afterward -- so "held-out validation"
    stays a stable, meaningful comparison point across successive training runs as
    the archive grows, instead of silently reshuffling every time.

Each manifest entry also carries an optional `tag` (e.g. "night_subset") recording
why a photo was added, so later analysis can filter validation results by subset
(e.g. "did the retrained model actually improve on the night-tagged photos?").
"""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE_DIR = ROOT / "dataset" / "sharp"
MANIFEST_PATH = ROOT / "dataset" / "manifest.json"
SOURCE_DIR = Path(r"C:\Users\Manish Kumar\Desktop\New folder (2)")
VAL_FRACTION = 0.12


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _split_for_hash(content_hash: str) -> str:
    bucket = int(content_hash[:8], 16) / 0xFFFFFFFF
    return "val" if bucket < VAL_FRACTION else "train"


def _archive_one(path: Path, manifest: dict, known_hashes: set, tag: str | None) -> bool:
    """Hash, copy if new, and record one file in the manifest. Returns True if added."""
    content_hash = _hash_file(path)
    if content_hash in known_hashes:
        return False

    dest_name = path.name
    dest_path = ARCHIVE_DIR / dest_name
    if dest_path.exists():
        # Same filename, different content (camera counter reuse) -- disambiguate.
        dest_name = f"{path.stem}_{content_hash[:8]}{path.suffix}"
        dest_path = ARCHIVE_DIR / dest_name
    shutil.copy2(path, dest_path)

    manifest[content_hash] = {
        "archive_filename": dest_name,
        "original_name": path.name,
        "source_path": str(path),
        "added_at": datetime.now(timezone.utc).isoformat(),
        "split": _split_for_hash(content_hash),
        "tag": tag,
    }
    known_hashes.add(content_hash)
    return True


def sync_from_source(source_dir: Path = SOURCE_DIR, tag: str | None = None) -> dict:
    """Copy any new (by content) photos from source_dir into the permanent archive.

    Safe to call every run: already-archived photos (matched by content hash) are
    skipped, never re-copied or reassigned to a different split.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    known_hashes = set(manifest.keys())

    if not source_dir.exists():
        print(f"Source folder not found ({source_dir}); using archive as-is.")
        return manifest

    candidates = sorted(set(source_dir.glob("*.JPG")) | set(source_dir.glob("*.jpg")))
    added = sum(_archive_one(p, manifest, known_hashes, tag) for p in candidates)

    _save_manifest(manifest)
    print(f"Dataset sync: {added} new photo(s) archived, {len(manifest)} total in archive.")
    return manifest


def sync_files(file_paths: list, tag: str | None = None) -> dict:
    """Archive an explicit, curated list of individual files (e.g. a hand-picked
    subset of dark/night photos pulled out of a much larger folder)."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    known_hashes = set(manifest.keys())

    added = 0
    for p in file_paths:
        p = Path(p)
        if not p.exists():
            print(f"Skipping missing file: {p}")
            continue
        added += _archive_one(p, manifest, known_hashes, tag)

    _save_manifest(manifest)
    print(f"Dataset sync (curated list): {added} new photo(s) archived, {len(manifest)} total in archive.")
    return manifest


if __name__ == "__main__":
    m = sync_from_source()
    n_train = sum(1 for v in m.values() if v["split"] == "train")
    n_val = sum(1 for v in m.values() if v["split"] == "val")
    print(f"Archive: {len(m)} total | {n_train} train | {n_val} val")
