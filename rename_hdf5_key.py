"""Rename ee_pos -> ee_pose in HDF5 files (in-place)."""

import argparse
import fnmatch
import h5py


def rename_keys(filepath: str, dry_run: bool = False) -> None:
    mode = "r" if dry_run else "r+"
    with h5py.File(filepath, mode) as f:
        if "data" not in f:
            print(f"[SKIP] {filepath}: no 'data' group found")
            return

        count = 0
        for name in sorted(f["data"].keys()):
            if not fnmatch.fnmatch(name, "demo_*"):
                continue

            obs_group = f["data"][name].get("obs")
            if obs_group is None:
                continue

            if "ee_pos" not in obs_group:
                continue

            if "ee_pose" in obs_group:
                print(f"  [SKIP] data/{name}/obs/ee_pose already exists, skipping")
                continue

            src = f"data/{name}/obs/ee_pos"
            dst = f"data/{name}/obs/ee_pose"

            if dry_run:
                print(f"  [DRY-RUN] would rename: {src} -> {dst}")
            else:
                f.move(src, dst)
                print(f"  [RENAMED] {src} -> {dst}")
            count += 1

        if count == 0:
            print(f"[INFO] {filepath}: no matching ee_pos keys found")
        else:
            print(f"[DONE] {filepath}: {count} key(s) {'would be ' if dry_run else ''}renamed")


def main():
    parser = argparse.ArgumentParser(
        description="Rename data/dmeo_*/obs/ee_pos -> ee_pose in HDF5 files (in-place)"
    )
    parser.add_argument("files", nargs="+", help="HDF5 file(s) to process")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview changes without modifying files"
    )
    args = parser.parse_args()

    for fp in args.files:
        print(f"Processing: {fp}")
        rename_keys(fp, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
