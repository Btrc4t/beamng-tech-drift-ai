"""Curate a collected drift dataset: clean it up and make it usable.

A dataset is what :mod:`data.collect` produces (one ``episode_*.npz`` per
episode; each file is self-contained, with the observation/action spaces
embedded). This tool operates on that directory:

- **collect-replays** copies the game's native ``.rpl`` replays (which live in
  the game's user-data folder, separate from the dataset) into the dataset so
  it is self-contained.
- **truncate-damage** cuts each episode at its first damage step (optionally a
  few steps before it, ``--margin``), so a crash doesn't pollute the clean
  driving that came before it.
- **truncate-short** cuts the last ``--margin`` steps off every episode
  shorter than ``--max-steps`` (default 1190), no damage check: those usually
  ended early due to damage, and the margin removes the crash aftermath.
  ``--margin`` is required, since without it the cut is a no-op.
- **truncate-start** drops the leading idle so each episode starts ``--lead``
  steps (default 1) before its first gear shift: the driver's 0-9 s of sitting
  in neutral at spawn is removed, so the shift is the episode's first action.
  Episodes whose gear never changes are skipped.
- **delete-damage** removes episodes that took any damage.
- **delete-short** removes episodes shorter than a given number of seconds.
- **split** splits the dataset into ``train/`` and ``validation/``
  subdirectories (the Hugging Face Hub split layout: the Hub auto-detects
  splits from the directory names, and the written ``README.md`` carries the
  manual ``configs:`` configuration). The assignment is a deterministic hash
  of each episode's name and ``--seed``: re-running is a no-op, and new
  episodes don't reshuffle the existing ones.
- **video** renders an episode as an ``.mp4`` (MJPEG) showing the LiDAR BEV
  (with the 20 ultrasonic beams overlaid: ray per beam, colored by distance,
  endpoint at the first obstacle) alongside the telemetry (drift angle,
  airspeed, force feedback, gear, reward).
- **play** launches the game (if it isn't running) and plays an episode's
  ``.rpl`` replay: the replay drives the car in real time, no menu navigation,
  and the command exits (closing the game) when the replay ends.
- **list** prints per-episode stats: length, total damage, the spawn waypoint
  name, and the paths of the episode file and its ``.rpl`` replay, sorted from
  highest to lowest reward per step. ``--min-seconds`` (one or more values)
  lists only the episodes at least that long; with several values it lists
  the episodes above each threshold in turn.
- **synthetic** reads the newest TensorBoard run's spawn observations
  (``episode_start/bev`` image + ``episode_start/*`` scalars) and writes
  length-1 launch episodes to a ``synthetic/`` folder at the data root (a
  sibling of the datasets, e.g. ``data/synthetic``): each pairs a real spawn
  observation with an action that engages a gear (``--first-gear-prob`` chance
  of 1st gear, else reverse) plus ``--throttle``. When ``--repeat`` > 1 the
  launch RPM is swept across the duplicates (rep 1 = idle, each later rep adds
  ``--rpm-range``/repeat) with random noise (``--rpm-noise``) so no two share
  an exact RPM. This reinforces "shift out of neutral on the first
  observation", where the recorded data is sparse and weak. BC training picks
  the ``synthetic/`` folder up automatically.

Damage is detected from the per-step ``dmg_delta`` stored in each step's
``info["reward_components"]`` (the increase in the car's damage that step); real
hits are large while an undamaged step is ~0.

Run it from ``src/`` (so the ``data`` package is importable)::

    # pull this dataset's replays out of the game folder
    python -m data.curate collect-replays --data data

    # preview the damage cleanup, then run it for real
    python -m data.curate truncate-damage --data data --dry-run
    python -m data.curate truncate-damage --data data

    # cut the last 20 steps off the short episodes (usually damage-ended)
    python -m data.curate truncate-short --data data --margin 20

    # drop the leading idle so each episode starts 1 step before the first shift
    python -m data.curate truncate-start --data data --lead 1

    # drop episodes shorter than 10 s
    python -m data.curate delete-short --data data --min-seconds 10

    # split the dataset into train/ and validation/ (the Hub split layout)
    python -m data.curate split --data data

    # render episode 6 as a video (BEV + telemetry)
    python -m data.curate video --data data --episode 6

    # launch the game and play episode 6's replay (no menus)
    python -m data.curate play --data data --episode 6

    # list episode lengths, damage, spawn, and file paths
    python -m data.curate list --data data
    python -m data.curate list --data data --min-seconds 10 30

    # build synthetic launch data from the newest TensorBoard run
    python -m data.curate synthetic --data data --runs 2 --repeat 10
"""
import argparse
import hashlib
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from data.dataset import episode_path, episode_paths, resolve_dataset
from beamng_tech_env.core.constants import AIRSPEED_MAX_KPH, gear_to_action
from beamng_tech_env.core.rpl import game_save_dir

# physics tick rate (matches the env); one step = 1/FPS seconds of sim time
FPS = 20
# per-step damage delta above this counts as damage (real hits are large, ~0 noise)
DEFAULT_DAMAGE_THRESHOLD = 0.1
# episodes shorter than this (steps) usually ended early due to damage; a full
# episode runs EPISODE_TIMEOUT * FPS = 1200 steps
DEFAULT_SHORT_MAX_STEPS = 1190
# the game's user-data "current" folder (contains replays/); override with --game-folder
DEFAULT_GAME_FOLDER = Path.home() / ".local/share/BeamNG/BeamNG.tech/current"
# observation normalisation constant (mirrors the env) so the gear channel can
# be decoded: gear obs = (gear + 1) / (MAX_GEAR + 1)
MAX_GEAR = 6


# -- episode io --------------------------------------------------------------
def _episode_files(data_dir):
    return episode_paths(data_dir)


def _episode_by_index(data_dir, index):
    """One episode by index (top level or a split subdirectory); for a
    missing index the top-level path is still built so callers can report
    it."""
    return (episode_path(data_dir, index)
            or data_dir / f"episode_{index:06d}.npz")


def _load_episode(path):
    """Load one ``episode_*.npz`` into a dict (obs/action/... + meta)."""
    with np.load(path, allow_pickle=True) as d:
        obs = {k[len("obs_"):]: d[k] for k in d.files if k.startswith("obs_")}
        meta_keys = [k for k in d.files
                     if k.startswith("ep_") or k in ("seed", "replay", "timestamp", "spawn",
                                                    "space_obs", "space_act", "env_meta")]
        return {
            "obs": obs,
            "action": d["action"],
            "reward": d["reward"],
            "terminated": d["terminated"],
            "truncated": d["truncated"],
            "info": [dict(x) for x in d["info"]] if "info" in d.files
                    else [{} for _ in d["reward"]],
            "meta": {k: d[k] for k in meta_keys},
        }


def _save_episode(path, ep):
    """Write an episode dict back to ``path`` in the collector's layout."""
    payload = {f"obs_{k}": v for k, v in ep["obs"].items()}
    payload.update({
        "action": ep["action"],
        "reward": ep["reward"],
        "terminated": ep["terminated"],
        "truncated": ep["truncated"],
        "info": np.array(ep["info"], dtype=object),
    })
    for k, v in ep["meta"].items():
        payload[k] = v if isinstance(v, np.ndarray) else np.array(v)
    np.savez_compressed(path, **payload)


def _damage_series(ep):
    """Per-step damage delta (from info reward_components), length T."""
    return np.array([float(x.get("reward_components", {})
                          .get("dmg_delta", 0.0)) for x in ep["info"]])


def _gear_series(ep):
    """Per-step integer gear (decoded from the obs gear channel), length T."""
    return np.round(ep["obs"]["gear"].reshape(-1) * (MAX_GEAR + 1) - 1).astype(int)


def _episode_replay(path):
    with np.load(path, allow_pickle=True) as d:
        return str(d["replay"].item()) if "replay" in d.files else ""


# -- subcommands -------------------------------------------------------------
def cmd_collect_replays(args):
    data_dir = resolve_dataset(args.data)
    game_folder = Path(args.game_folder).expanduser()
    dest = data_dir / "replays"
    dest.mkdir(parents=True, exist_ok=True)
    if not game_folder.is_dir():
        print(f"[curate] game folder not found: {game_folder} (pass --game-folder)")
        return
    copied = missing = 0
    for path in _episode_files(data_dir):
        replay = _episode_replay(path)
        if not replay:
            continue
        src = game_folder / replay
        if src.exists():
            shutil.copy2(src, dest / src.name)
            copied += 1
        else:
            missing += 1
    print(f"[curate] copied {copied} replays -> {dest} ({missing} missing)")


def _truncate_at(ep, cut):
    """Cut an episode at step ``cut`` and mark the new end as truncated."""
    for k in ep["obs"]:
        ep["obs"][k] = ep["obs"][k][:cut]
    ep["action"] = ep["action"][:cut]
    ep["reward"] = ep["reward"][:cut]
    ep["info"] = ep["info"][:cut]
    ep["terminated"] = ep["terminated"][:cut].copy()
    ep["truncated"] = ep["truncated"][:cut].copy()
    ep["terminated"][-1] = False   # cut before any termination
    ep["truncated"][-1] = True     # ...and mark the new end as truncated
    ep["meta"]["ep_length"] = np.int32(cut)
    ep["meta"]["ep_reward"] = np.float32(float(np.sum(ep["reward"])))
    ep["meta"]["ep_terminated"] = np.bool_(False)
    ep["meta"]["ep_truncated"] = np.bool_(True)


def _truncate_start_at(ep, start):
    """Drop the first ``start`` steps of an episode (the leading idle).

    The end of the episode is untouched, so its terminated/truncated flags are
    left as-is; only the length and reward meta are refreshed."""
    for k in ep["obs"]:
        ep["obs"][k] = ep["obs"][k][start:]
    ep["action"] = ep["action"][start:]
    ep["reward"] = ep["reward"][start:]
    ep["info"] = ep["info"][start:]
    ep["terminated"] = ep["terminated"][start:].copy()
    ep["truncated"] = ep["truncated"][start:].copy()
    ep["meta"]["ep_length"] = np.int32(len(ep["reward"]))
    ep["meta"]["ep_reward"] = np.float32(float(np.sum(ep["reward"])))


def _truncate_episodes(data_dir, args, max_steps=None, at_damage=True):
    """Cut episodes and mark the new end as truncated; episodes whose cut
    falls below ``args.min_keep`` are deleted instead.

    With ``at_damage`` the cut is the first damage step minus ``args.margin``
    (episodes without damage are skipped); otherwise it is the episode end
    minus ``args.margin``, with no damage check. If ``max_steps`` is given,
    only episodes shorter than that are considered."""
    truncated = deleted = steps_lost = total_steps = 0
    for path in _episode_files(data_dir):
        ep = _load_episode(path)
        T = len(ep["reward"])
        total_steps += T
        if max_steps is not None and T >= max_steps:
            continue
        if at_damage:
            hits = np.where(_damage_series(ep) > args.threshold)[0]
            if len(hits) == 0:
                continue
            cut = max(0, int(hits[0]) - args.margin)
            why = f"first damage at {int(hits[0])}, margin {args.margin}"
        else:
            cut = max(0, T - args.margin)
            why = f"margin {args.margin}"
        if cut < args.min_keep:
            if args.dry_run:
                print(f"[curate] would delete {path.name} ({T} steps): "
                      f"cut {cut} < min_keep {args.min_keep}")
            else:
                path.unlink()
            deleted += 1
            steps_lost += T
            continue
        if args.dry_run:
            print(f"[curate] would truncate {path.name}: {T} -> {cut} steps ({why})")
            truncated += 1
            steps_lost += T - cut
            continue
        _truncate_at(ep, cut)
        _save_episode(path, ep)
        truncated += 1
        steps_lost += T - cut
    if args.dry_run:
        pct = 100.0 * steps_lost / total_steps if total_steps else 0.0
        print(f"[curate] would truncate {truncated}, would delete {deleted} "
              f"episode(s), losing {steps_lost} of {total_steps} steps ({pct:.1f}%)")
    else:
        print(f"[curate] truncated {truncated}, deleted {deleted} episode(s)")


def cmd_truncate_damage(args):
    _truncate_episodes(resolve_dataset(args.data), args, at_damage=True)


def cmd_truncate_short(args):
    _truncate_episodes(resolve_dataset(args.data), args, max_steps=args.max_steps,
                       at_damage=False)


def cmd_truncate_start(args):
    """Drop the leading idle so each episode starts ``--lead`` steps before its
    first gear shift (the driver's idle time at spawn is removed). Episodes
    whose gear never changes, or that already start within the lead, are
    skipped."""
    truncated = skipped = steps_lost = total_steps = 0
    for path in _episode_files(resolve_dataset(args.data)):
        ep = _load_episode(path)
        T = len(ep["reward"])
        total_steps += T
        gear = _gear_series(ep)
        changes = np.where(gear[1:] != gear[:-1])[0]
        if len(changes) == 0:
            skipped += 1
            continue
        t_change = int(changes[0]) + 1   # first step in the new gear
        start = max(0, t_change - args.lead)
        if start == 0:
            skipped += 1
            continue
        if args.dry_run:
            print(f"[curate] would truncate {path.name}: drop first {start} "
                  f"steps (first gear change at {t_change}), {T} -> {T - start}")
            truncated += 1
            steps_lost += start
            continue
        _truncate_start_at(ep, start)
        _save_episode(path, ep)
        truncated += 1
        steps_lost += start
    if args.dry_run:
        pct = 100.0 * steps_lost / total_steps if total_steps else 0.0
        print(f"[curate] would truncate {truncated}, skip {skipped} episode(s), "
              f"losing {steps_lost} of {total_steps} steps ({pct:.1f}%)")
    else:
        print(f"[curate] truncated {truncated}, skipped {skipped} episode(s)")


def cmd_delete_damage(args):
    deleted = 0
    for path in _episode_files(resolve_dataset(args.data)):
        dmg = _damage_series(_load_episode(path))
        if not (dmg > args.threshold).any():
            continue
        if args.dry_run:
            print(f"[curate] would delete {path.name} (max dmg delta {dmg.max():.3f})")
        else:
            path.unlink()
        deleted += 1
    if args.dry_run:
        print(f"[curate] would delete {deleted} episode(s) with damage")
    else:
        print(f"[curate] deleted {deleted} episode(s) with damage")


def cmd_delete_short(args):
    deleted = 0
    for path in _episode_files(resolve_dataset(args.data)):
        T = len(_load_episode(path)["reward"])
        secs = T / FPS
        if secs >= args.min_seconds:
            continue
        if args.dry_run:
            print(f"[curate] would delete {path.name} ({secs:.1f}s < {args.min_seconds}s)")
        else:
            path.unlink()
        deleted += 1
    if args.dry_run:
        print(f"[curate] would delete {deleted} short episode(s)")
    else:
        print(f"[curate] deleted {deleted} short episode(s)")


# -- split -------------------------------------------------------------------
def _split_fraction(seed, name):
    """A deterministic per-episode value in [0, 1): the same episode always
    lands in the same split for a given seed, so new episodes don't reshuffle
    the existing ones."""
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _split_readme(data_dir):
    """The dataset card with the Hub's manual split configuration (the
    configs: YAML block); written once, left alone afterwards."""
    readme = Path(data_dir) / "README.md"
    if readme.exists():
        return
    readme.write_text(
        "---\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        '    path: "train/*.npz"\n'
        "  - split: validation\n"
        '    path: "validation/*.npz"\n'
        "---\n"
        "\n"
        f"# {Path(data_dir).name} drift episodes\n"
        "\n"
        "Recorded driving episodes from BeamNG.drive. Each ``episode_*.npz`` "
        "is self-contained: per-step observations, actions, rewards, and the "
        "embedded observation/action spaces.\n"
        "\n"
        "The dataset is split into `train` and `validation` (the Hugging "
        "Face Hub split layout; see the [data files "
        "configuration](https://huggingface.co/docs/hub/"
        "datasets-data-files-configuration) doc). Recreate it with:\n"
        "\n"
        "    python -m data.curate split --data <this directory> "
        "--val-ratio 0.1 --seed 42\n"
    )


def cmd_split(args):
    """Split a dataset into train/ and validation/ subdirectories (the Hub
    split layout, auto-detected by name). The assignment is a deterministic
    hash of each episode's name and --seed: re-running is a no-op, and new
    episodes don't reshuffle the existing ones."""
    data_dir = resolve_dataset(args.data)
    paths = episode_paths(data_dir)
    if not paths:
        raise SystemExit(f"[curate] no episode_*.npz in {data_dir}")
    val = {p for p in paths
           if _split_fraction(args.seed, p.name) < args.val_ratio}
    train_dir, val_dir = data_dir / "train", data_dir / "validation"
    moved = 0
    for p in paths:
        dest_dir = val_dir if p in val else train_dir
        if p.parent != dest_dir:
            if not args.dry_run:
                dest_dir.mkdir(exist_ok=True)
                shutil.move(str(p), str(dest_dir / p.name))
            moved += 1
    if not args.dry_run:
        for d in (train_dir, val_dir):
            if d.is_dir() and not list(d.glob("episode_*.npz")):
                d.rmdir()
        _split_readme(data_dir)
    verb = "would move" if args.dry_run else "moved"
    print(f"[curate] {verb} {moved} episode(s): {len(paths) - len(val)} "
          f"train, {len(val)} validation (val-ratio {args.val_ratio}, "
          f"seed {args.seed})")


def cmd_list(args):
    data_dir = resolve_dataset(args.data)
    paths = ([_episode_by_index(data_dir, args.episode)]
             if args.episode is not None else _episode_files(data_dir))
    episodes = []
    for path in paths:
        if not path.exists():
            print(f"[curate] no such episode: {path.name}")
            continue
        ep = _load_episode(path)
        T = len(ep["reward"])
        episodes.append((T / FPS, float(ep["reward"].sum()) / T if T else 0.0, path, ep))
    for secs in sorted(args.min_seconds) if args.min_seconds else [0.0]:
        kept = sorted((e for e in episodes if e[0] >= secs),
                      key=lambda item: item[1], reverse=True)
        if args.min_seconds:
            print(f"[curate] {len(kept)} of {len(episodes)} episode(s) >= {secs:.1f}s")
        for _, _, path, ep in kept:
            T = len(ep["reward"])
            reward = float(ep["reward"].sum())
            damage = float(_damage_series(ep).sum())
            meta = ep["meta"]
            spawn = str(np.asarray(meta["spawn"]).item()) if "spawn" in meta else ""
            replay = str(np.asarray(meta["replay"]).item()) if "replay" in meta else ""
            print(f"[{path.stem.split('_')[1]}] {path.name}")
            print(f"    length: {T} steps ({T / FPS:.1f} s)")
            print(f"    reward: {reward:.3f}")
            print(f"    damage: {damage:.3f}")
            print(f"    spawn:  {spawn or '-'}")
            print(f"    npz:    {path}")
            print(f"    rpl:    {replay or '-'}")


def cmd_video(args):
    data_dir = resolve_dataset(args.data)
    paths = ([_episode_by_index(data_dir, args.episode)]
             if args.episode is not None else _episode_files(data_dir))
    for path in paths:
        if not path.exists():
            print(f"[curate] no such episode: {path.name}")
            continue
        out = (Path(args.out) if (args.out and len(paths) == 1)
               else path.with_suffix(".mp4"))
        _render_video(_load_episode(path), out, fps=args.fps, width=args.width,
                      workers=args.workers)
        print(f"[curate] wrote {out}")


# -- synthetic launch data ---------------------------------------------------
def _load_spaces(data_dir):
    """The observation/action spaces embedded in the dataset's first episode.

    The synthetic episodes must be self-contained (they embed these spaces), so
    they are read from the dataset the synthetic data augments."""
    paths = episode_paths(data_dir)
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz in {data_dir}")
    with np.load(paths[0], allow_pickle=True) as d:
        space_obs = pickle.loads(d["space_obs"].tobytes())
        space_act = pickle.loads(d["space_act"].tobytes())
        env_meta = pickle.loads(d["env_meta"].tobytes())
    return space_obs, space_act, env_meta


# Canonical value for a spawn channel missing from the logs (a run that
# predates the numeric ``episode_start/*`` scalars): the car the env hands over
# in, i.e. neutral gear and at rest.
_SPAWN_DEFAULTS = {
    "rpm": 0.0,
    "imu_acc": (0.0, 0.0, 0.0),
    "imu_angvel": (0.0, 0.0, 0.0),
    "drift_angle": 0.0,
    "airspeed": 0.0,
    "is_shifting": 0.0,
    "gear": 1.0 / (MAX_GEAR + 1),  # neutral
    "ffb": 0.0,
}


def _tb_spawn_obs(bev_event, scalars, space_obs):
    """Rebuild one spawn observation from a TB image event + its step scalars.

    The BEV comes from the ``episode_start/bev`` image; each numeric channel
    comes from the ``episode_start/*`` scalar logged at the same step, or a
    canonical neutral-spawn default when that scalar is absent. The route
    channels are included only when the dataset's space carries them (newer
    envs): from their ``episode_start/*`` scalars, or the neutral (no-route)
    values when absent."""
    import cv2
    arr = cv2.imdecode(np.frombuffer(bev_event.encoded_image_string, np.uint8),
                       cv2.IMREAD_GRAYSCALE)
    bev = arr.astype(np.uint8).reshape(1, arr.shape[0], arr.shape[1])

    def val(tag, default):
        return float(scalars[tag]) if tag in scalars else float(default)

    d = _SPAWN_DEFAULTS
    obs = {
        "rpm":         np.array([val("episode_start/rpm", d["rpm"])], np.float32),
        "imu_acc":     np.array([val("episode_start/imu_acc[0]", d["imu_acc"][0]),
                                 val("episode_start/imu_acc[1]", d["imu_acc"][1]),
                                 val("episode_start/imu_acc[2]", d["imu_acc"][2])], np.float32),
        "imu_angvel":  np.array([val("episode_start/imu_angvel[0]", d["imu_angvel"][0]),
                                 val("episode_start/imu_angvel[1]", d["imu_angvel"][1]),
                                 val("episode_start/imu_angvel[2]", d["imu_angvel"][2])], np.float32),
        "drift_angle": np.array([val("episode_start/drift_angle", d["drift_angle"])], np.float32),
        "airspeed":    np.array([val("episode_start/airspeed", d["airspeed"])], np.float32),
        "is_shifting": np.array([val("episode_start/is_shifting", d["is_shifting"])], np.float32),
        "gear":        np.array([val("episode_start/gear", d["gear"])], np.float32),
        "ffb":         np.array([val("episode_start/ffb", d["ffb"])], np.float32),
        "bev":         bev,
    }
    for key, size in (("next", 3), ("attack", 1), ("turns", 3)):
        if key not in space_obs.spaces:
            continue
        if size == 1:
            obs[key] = np.array([val(f"episode_start/{key}", 0.0)], np.float32)
        else:
            obs[key] = np.array(
                [val(f"episode_start/{key}[{j}]", 0.0) for j in range(size)],
                np.float32)
    return obs


def _read_tb_spawns(log_dir, n_runs, space_obs):
    """Spawn observations from the newest ``n_runs`` TensorBoard runs.

    Each ``episode_start/bev`` image is one spawn; the ``episode_start/*``
    scalars logged at the same step supply the numeric channels. Returns
    ``(spawns, run_names)``; both are empty when the log dir is missing or no
    run has spawn images. ``space_obs`` (the dataset's observation space)
    selects which channels the rebuilt obs carries (the route channels are
    included only when the space has them)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    log_root = Path(log_dir)
    if not log_root.is_dir():
        return [], []
    runs = sorted((d for d in log_root.iterdir() if d.is_dir()), reverse=True)[:n_runs]
    spawns = []
    run_names = []
    for run in runs:
        event_files = sorted(run.rglob("events.out.tfevents.*"))
        if not event_files:
            continue
        # 0 = store everything: the default size guidance keeps only 4 images
        # per tag in memory, which would silently cap the spawn count
        ea = EventAccumulator(str(event_files[-1].parent),
                             size_guidance={"images": 0, "scalars": 0})
        ea.Reload()
        tags = ea.Tags()
        if "episode_start/bev" not in tags.get("images", []):
            continue
        run_names.append(run.name)
        scalars_by_step = {}
        for tag in tags.get("scalars", []):
            if not tag.startswith("episode_start/"):
                continue
            for ev in ea.Scalars(tag):
                scalars_by_step.setdefault(ev.step, {})[tag] = ev.value
        for ev in ea.Images("episode_start/bev"):
            spawns.append(_tb_spawn_obs(ev, scalars_by_step.get(ev.step, {}),
                                        space_obs))
    return spawns, run_names


def cmd_synthetic(args):
    """Create synthetic launch transitions from the newest TB spawn observations.

    Each transition pairs a real spawn observation with a launch action that
    engages a gear (``--first-gear-prob`` chance of 1st gear, else reverse) and
    applies ``--throttle``. This teaches the policy to shift out of neutral on
    the first observation, where the recorded data is sparse and weak."""
    data_dir = resolve_dataset(args.data)
    # the synthetic launch folder lives at the data root, as a sibling of the
    # datasets (e.g. data/synthetic, not data/human/synthetic)
    out_dir = Path(args.out) if args.out else data_dir.parent / "synthetic"
    space_obs, space_act, env_meta = _load_spaces(data_dir)

    # the initial observations come solely from the TensorBoard logs
    spawns, run_names = _read_tb_spawns(args.log_dir, args.runs, space_obs)
    if not spawns:
        raise SystemExit(
            f"[curate] no spawn observations in {args.log_dir}; run online "
            f"training first so the logs contain episode_start data")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n = 0
    for obs in spawns:
        base_rpm = float(np.asarray(obs["rpm"]).ravel()[0])
        for rep in range(1, args.repeat + 1):
            gear = (gear_to_action(1) if rng.random() < args.first_gear_prob
                    else gear_to_action(-1))
            action = np.concatenate([[0.0, args.throttle, 0.0], gear]).astype(np.float32)
            # sweep the launch RPM across the repetitions (rep 1 is the spawn's
            # idle RPM, each later rep adds rpm_range/repeat) and add random
            # noise so no two episodes share an exact RPM
            rpm = base_rpm + (rep - 1) * (args.rpm_range / args.repeat)
            rpm = float(np.clip(rpm + rng.normal(0.0, args.rpm_noise), 0.0, 1.0))
            obs_rep = dict(obs)
            obs_rep["rpm"] = np.array([rpm], dtype=np.float32)
            if not args.dry_run:
                ep = {
                    "obs": {k: v.reshape(1, *v.shape) for k, v in obs_rep.items()},
                    "action": action.reshape(1, -1),
                    "reward": np.zeros(1, dtype=np.float32),
                    "terminated": np.zeros(1, dtype=bool),
                    "truncated": np.ones(1, dtype=bool),
                    "info": [{}],
                    "meta": {
                        "ep_reward": np.float32(0.0),
                        "ep_length": np.int32(1),
                        "ep_terminated": np.bool_(False),
                        "ep_truncated": np.bool_(True),
                        "seed": np.int64(args.seed + n),
                        "replay": np.array(""),
                        "timestamp": np.array(datetime.now().isoformat()),
                        "space_obs": np.frombuffer(pickle.dumps(space_obs), np.uint8),
                        "space_act": np.frombuffer(pickle.dumps(space_act), np.uint8),
                        "env_meta": np.frombuffer(pickle.dumps(env_meta), np.uint8),
                    },
                }
                _save_episode(out_dir / f"episode_{n:06d}.npz", ep)
            n += 1
    verb = "would write" if args.dry_run else "wrote"
    print(f"[curate] {len(spawns)} spawn observation(s) from {', '.join(run_names)}, "
          f"repeat={args.repeat}")
    print(f"[curate] {verb} {n} synthetic launch episode(s) -> {out_dir}")


def _render_chunk(args):
    """Worker: render frames [start, end) of one episode into memory.

    Runs in its own process. ``args`` is (start, end, data, width,
    trace_vals) where ``data`` is the (obs, action, reward) tuple (obs a
    dict of per-step arrays) and ``trace_vals`` the per-step trace values
    (debug._TRACE_SPECS order): each chunk seeds its rolling window with the
    samples before the chunk, so the video shows the same sliding trace the
    live window does (no per-chunk reset). Each frame is rendered through
    the live debug window's code path (beamng_tech_env.debug), so the video
    and the window stay in sync. Returns ``(W, H, frames)`` where ``frames``
    is a uint8 ``(n, H, W, 3)`` BGR array.
    """
    start, end, data, width, trace_vals = args
    import cv2

    from beamng_tech_env import debug

    obs, action, reward = data
    layout = debug.make_layout(width)
    traces = debug.make_traces()
    for (label, *_), values in zip(debug._TRACE_SPECS, trace_vals):
        traces[label].extend(values[:start])
    frames = np.empty((end - start, layout.height, layout.width, 3),
                      dtype=np.uint8)
    for i, t in enumerate(range(start, end)):
        obs_t = {k: v[t] for k, v in obs.items()}
        payload = debug._payload_from(obs_t, action=action[t],
                                      reward=float(reward[t]), bev=None)
        frames[i] = debug._render_frame(cv2, payload, traces, layout)
    return (layout.width, layout.height, frames)


def _render_video(ep, out_path, fps, width, workers=None):
    """Render an episode to an MJPEG ``.mp4`` (BEV + telemetry) through the
    live debug window's render path (beamng_tech_env.debug), so the video and
    the window stay in sync. Frames are rendered in parallel across worker
    processes (kept in memory) and written in order. ``width`` is the window
    width in pixels; ``workers`` caps the number of parallel renderers
    (default: min(cpu count, 8))."""
    import os

    import cv2

    T = len(ep["reward"])
    # the BEV the debug window shows: the (road-corrected) LiDAR BEV, or the
    # road BEV (scaled to uint8) for datasets that predate it; the road_bev
    # channel itself is dropped (the window renders only the BEV)
    obs = {k: v for k, v in ep["obs"].items() if k != "road_bev"}
    if "bev" not in obs:
        obs["bev"] = (np.clip(ep["obs"]["road_bev"], 0.0, 1.0)
                      * 255.0).astype(np.uint8)
    data = (obs, ep["action"], ep["reward"])
    # the trace values per step (the _payload_from extraction, vectorised),
    # so each parallel chunk can seed its rolling window with the samples
    # before the chunk: the video shows the same sliding trace the live
    # window does, with no per-chunk reset
    def _hist(key, scale):
        arr = obs.get(key)
        return (np.asarray(arr).reshape(-1) * scale
                if arr is not None else np.zeros(T))
    trace_vals = (_hist("drift_angle", 180.0),
                  _hist("airspeed", AIRSPEED_MAX_KPH),
                  _hist("ffb", 1.0))

    if workers is None:
        workers = min(os.cpu_count() or 1, 8)
    workers = max(1, min(workers, T))
    chunk = -(-T // workers)                    # ceil(T / workers)
    chunk_args = [(i, min(i + chunk, T), data, width, trace_vals)
                  for i in range(0, T, chunk)]

    if len(chunk_args) == 1:
        results = [_render_chunk(chunk_args[0])]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=len(chunk_args)) as pool:
            results = list(pool.map(_render_chunk, chunk_args))

    W, H = results[0][0], results[0][1]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (W, H))
    for _, _, frames in results:
        for frame in frames:
            writer.write(frame)
    writer.release()


# -- cli ---------------------------------------------------------------------
def _resolve_play_replay(args):
    """The .rpl to play: --replay directly, or the dataset episode's recorded
    replay (resolved from the dataset's replays/ folder, then the game's)."""
    if args.replay:
        path = Path(args.replay).expanduser()
        if not path.is_file():
            raise SystemExit(f"[curate] no such replay: {path}")
        return path
    data_dir = resolve_dataset(args.data)
    ep = _episode_by_index(data_dir, args.episode)
    if not ep.is_file():
        raise SystemExit(f"[curate] no such episode: {ep.name} in {data_dir}")
    replay = _episode_replay(ep)
    if not replay:
        raise SystemExit(f"[curate] {ep.name} has no recorded replay")
    src = data_dir / "replays" / Path(replay).name
    if not src.is_file():
        src = game_save_dir() / replay
    if not src.is_file():
        raise SystemExit(f"[curate] replay {replay!r} not found (looked in "
                         f"{data_dir / 'replays'} and {game_save_dir()})")
    return src


def cmd_play(args):
    """Launch the game (if it isn't running) and play the replay: the replay
    drives the car in real time, and the command exits (closing the game) when
    the replay ends."""
    from beamng_tech_env.envs.replay_env import BeamNGReplayEnv
    from data.regenerate import _game_relative

    rpl = _resolve_play_replay(args)
    rel = _game_relative(rpl)
    print(f"[curate] playing {rpl.name} (launches the game if needed)",
          flush=True)
    env = BeamNGReplayEnv(replay_path=rel, level=args.level)
    try:
        env.reset(options={"replay": rel})
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        steps = 0
        terminated = False
        while True:
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        print(f"[curate] {'terminated' if terminated else 'replay ended'} "
              f"after {steps} steps; closing the game", flush=True)
    except KeyboardInterrupt:
        print("\n[curate] interrupted; closing the game", flush=True)
    finally:
        env.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Curate a collected drift dataset (clean + visualise)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_data(p):
        p.add_argument("--data", required=True,
                       help="dataset directory, or the data root (e.g. data/) "
                            "to auto-resolve the dataset (prefers 'human')")

    p = sub.add_parser("collect-replays",
                       help="copy this dataset's .rpl replays from the game folder")
    add_data(p)
    p.add_argument("--game-folder", default=str(DEFAULT_GAME_FOLDER),
                   help="the game's user-data 'current' folder (contains replays/)")
    p.set_defaults(func=cmd_collect_replays)

    p = sub.add_parser("truncate-damage",
                       help="cut each episode at its first damage step")
    add_data(p)
    p.add_argument("--threshold", type=float, default=DEFAULT_DAMAGE_THRESHOLD,
                   help="per-step damage delta above this counts as damage")
    p.add_argument("--margin", type=int, default=0,
                   help="extra steps to cut before the first damage step")
    p.add_argument("--min-keep", type=int, default=20,
                   help="if truncation leaves fewer steps than this, delete instead")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_truncate_damage)

    p = sub.add_parser("truncate-short",
                       help="cut the last --margin steps off short episodes "
                            "(usually damage-ended)")
    add_data(p)
    p.add_argument("--max-steps", type=int, default=DEFAULT_SHORT_MAX_STEPS,
                   help="only consider episodes shorter than this "
                        f"(default: {DEFAULT_SHORT_MAX_STEPS})")
    p.add_argument("--margin", type=int, required=True,
                   help="steps to cut off the end of each episode (required: "
                        "without it the cut is a no-op)")
    p.add_argument("--min-keep", type=int, default=20,
                   help="if truncation leaves fewer steps than this, delete instead")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_truncate_short)

    p = sub.add_parser("truncate-start",
                       help="drop the leading idle so each episode starts "
                            "--lead steps before its first gear shift")
    add_data(p)
    p.add_argument("--lead", type=int, default=1,
                   help="steps before the first gear shift to keep (default: 1)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_truncate_start)

    p = sub.add_parser("delete-damage", help="delete episodes that took damage")
    add_data(p)
    p.add_argument("--threshold", type=float, default=DEFAULT_DAMAGE_THRESHOLD,
                   help="per-step damage delta above this counts as damage")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_delete_damage)

    p = sub.add_parser("delete-short", help="delete episodes shorter than N seconds")
    add_data(p)
    p.add_argument("--min-seconds", type=float, default=10.0,
                   help="delete episodes shorter than this many seconds")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_delete_short)

    p = sub.add_parser("split",
                       help="split the dataset into train/ and validation/ "
                            "subdirectories (the Hub split layout)")
    add_data(p)
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="fraction of episodes held out for validation "
                        "(default: 0.1)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the deterministic per-episode assignment "
                        "(default: 42)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("list",
                       help="list episodes: length, damage, spawn, file paths")
    add_data(p)
    p.add_argument("--episode", type=int, default=None,
                   help="episode index to list (default: all episodes)")
    p.add_argument("--min-seconds", type=float, nargs="+", default=None,
                   help="list only episodes at least this long; with several "
                        "values, list the episodes above each threshold")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("video", help="render an episode as an mp4 (BEV + telemetry)")
    add_data(p)
    p.add_argument("--episode", type=int, default=None,
                   help="episode index to render (default: all episodes)")
    p.add_argument("--out", default=None,
                   help="output mp4 path (default: <episode>.mp4 next to the data)")
    p.add_argument("--fps", type=int, default=FPS, help="output frame rate")
    p.add_argument("--width", type=int, default=960,
                   help="window width in pixels (default: 960, as the live "
                        "debug window)")
    p.add_argument("--workers", type=int, default=None,
                   help="parallel renderers (default: min(cpu count, 8))")
    p.set_defaults(func=cmd_video)

    p = sub.add_parser("synthetic",
                       help="create synthetic launch transitions from the newest "
                            "TensorBoard spawn observations")
    p.add_argument("--data", required=True,
                   help="dataset directory, or the data root (e.g. data/) to "
                        "auto-resolve the dataset (prefers 'human')")
    p.add_argument("--log-dir", default="logs/PPO",
                   help="TensorBoard log root (default: logs/PPO)")
    p.add_argument("--out", default=None,
                   help="output directory (default: <data root>/synthetic, a "
                        "sibling of the datasets, e.g. data/synthetic)")
    p.add_argument("--runs", type=int, default=1,
                   help="how many newest runs to read spawn observations from "
                        "(default: 1, i.e. all spawns in the newest run)")
    p.add_argument("--repeat", type=int, default=1,
                   help="duplicate each spawn observation this many times "
                        "(default: 1)")
    p.add_argument("--first-gear-prob", type=float, default=0.8,
                   help="probability the launch action selects 1st gear; the "
                        "rest select reverse (default: 0.8)")
    p.add_argument("--throttle", type=float, default=0.5,
                   help="throttle of the launch action (default: 0.5)")
    p.add_argument("--rpm-range", type=float, default=0.5,
                   help="total launch-RPM range swept across the repetitions, "
                        "above the spawn's idle RPM: rep 1 is idle, rep N is "
                        "idle + rpm_range (default: 0.5)")
    p.add_argument("--rpm-noise", type=float, default=0.01,
                   help="std of the random noise added to every launch RPM so "
                        "no two episodes share an exact value (default: 0.01)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_synthetic)

    p = sub.add_parser("play",
                       help="launch the game and play an episode's replay")
    p.add_argument("--data", default=None,
                   help="dataset directory, or the data root (e.g. data/) to "
                        "auto-resolve the dataset (with --episode)")
    p.add_argument("--episode", type=int, default=None,
                   help="episode index to play (with --data)")
    p.add_argument("--replay", default=None,
                   help="a .rpl file to play directly (instead of "
                        "--data/--episode)")
    p.add_argument("--level", default="west_coast_usa",
                   help="the level the replay was recorded on "
                        "(default: west_coast_usa)")
    p.set_defaults(func=cmd_play)

    args = parser.parse_args(argv)
    if args.cmd == "play" and not (args.replay
                                   or (args.data and args.episode is not None)):
        parser.error("play needs --replay <file.rpl> or --data <dataset> "
                     "--episode <N>")
    args.func(args)


if __name__ == "__main__":
    main()
