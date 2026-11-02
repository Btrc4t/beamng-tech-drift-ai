"""Throwaway: restore the FFB of data/human in place, no game replay.

data/human was regenerated from data/human-outdated-etki with
``regenerate --lidar``: the live FFB read is always 0 (the replayed vehicle
is a clone the game's hydros module doesn't drive), so every episode's FFB
was zeroed out. This re-runs the same FFB reuse as the regenerate fix
(align by airspeed, copy from the source episode) over the existing files,
in place.

    python -m data.fix_ffb
"""
from pathlib import Path

from data.dataset import episode_paths
from data.regenerate import _reuse_ffb

SRC = Path("data/human-outdated-etki")
DST = Path("data/human")


def main() -> None:
    fixed = skipped = 0
    for out_path in episode_paths(DST):
        src_path = SRC / out_path.name
        if not src_path.is_file():
            print(f"[fix_ffb] {out_path.name}: no source episode; skipping")
            skipped += 1
            continue
        _reuse_ffb(out_path, src_path)
        fixed += 1
    print(f"[fix_ffb] {fixed} episodes fixed in place, {skipped} skipped")


if __name__ == "__main__":
    main()
