"""Regenerate dataset episodes from BeamNG ``.rpl`` replays.

Two modes:

- **offline** (the default): no game. Each episode is rebuilt from its
  ``.rpl`` (the recorded pose/electrics are the ground truth for the
  driving): the pose/electrics-derived channels (drift, airspeed, gear, rpm,
  IMU, action), the road BEV (rasterized from the cached road network) and
  the reward (the current formula) are recomputed, while the channels the
  ``.rpl`` doesn't carry (the force feedback, the per-step damage) are
  reused from the original episode. The ``bev`` channel is the corrected
  LiDAR: the original episode's LiDAR BEV, corrected against the road
  network (a bare ``.rpl`` has no LiDAR returns, so there the road BEV is
  the LiDAR: road surface = road level, off-road = obstacle), and the beams
  are computed on it. The
  episode's step grid (20 Hz) is kept; the ``.rpl`` is aligned to it first,
  since a curated episode may start part-way into its ``.rpl`` (the leading
  idle is truncated away). The game is used once, only to fetch the road
  network into the cache (``--roads``) when it is missing.
- **live** (``--lidar``): plays each ``.rpl`` in the game with the
  :class:`~beamng_tech_env.envs.replay_env.BeamNGReplayEnv` (which sources
  the pose/electrics from the ``.rpl`` and reads the LiDAR BEV live) and
  records the regenerated observation/action/reward with the
  :class:`~data.collector.TrajectoryCollector`. The FFB is reused from the
  source episode, as in offline mode: the replayed vehicle is a clone the
  game's hydros module doesn't drive, so the live read is always 0.

The ``.rpl`` files must be playable in-game (live mode), so each is copied
into the game's ``replays/`` folder (if not already there) and referenced by
its game-relative path. In live mode the game must be launchable (BNG_HOME
set); in offline mode only the road-network fetch needs it (and only when
the cache is missing).

``--data`` is the dataset directory (e.g. ``data/human``): the replays to
regenerate are taken from the dataset itself, each episode's recorded
``replay`` path, in episode order, so the regenerated dataset corresponds
1:1 to the curated one (episodes without a usable ``.rpl`` are skipped).
A folder of ``.rpl`` files (or a single ``.rpl``) is also accepted, but
pointing it at the game's ``replays/`` folder would pick up replays from
other sessions.

    python -m data.regenerate --data data/human --out data/human-regen
    python -m data.regenerate --data data/human --out data/human-regen --lidar
    python -m data.regenerate --data <one.rpl> --out data/regen --limit 1

The output directory is appended to (episode indices continue past any
existing ``episode_*.npz``), so re-running after an interruption resumes.
"""
import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

from beamng_tech_env.core.array import to_cpu
from beamng_tech_env.core.constants import (
    AIRSPEED_MAX_KPH,
    ANGLE_MAX,
    BEV_N,
    DEFAULT_STEERING_WHEEL_LOCK,
    MAX_GEAR,
    ON_ROAD_FAST_TRACK_DIST,
    TICK_DT,
    gear_to_action,
)
from beamng_tech_env.core.obs import (
    correct_lidar_bev,
    derive_imu,
    drift_from_state,
    state_snapshot,
)
from beamng_tech_env.core.reward import compute_reward
from beamng_tech_env.core.rpl import RplTrajectory, game_save_dir, quat_to_axes
from beamng_tech_env.core.roads import (
    Navigraph,
    car_grid_axes,
    road_centerline_segments,
)
from data.collector import TrajectoryCollector
from data.curate import _load_episode
from data.dataset import episode_paths

ENV_ID = "beamng_tech_env/BeamNGReplay-v0.39"
LEVEL = "west_coast_usa"
# a .rpl is only trusted for an episode when the alignment is this good
ALIGN_MIN_CORR = 0.9


def _dataset_pairs(data_dir: Path) -> list:
    """The ``(episode_path, rpl_path)`` of a dataset's episodes, in episode
    order.

    Each ``episode_*.npz`` stores the game-relative path of the ``.rpl`` that
    recorded it (``meta["replay"]``); each is resolved against the dataset's
    ``replays/`` subfolder first, then the game's replays folder. Episodes
    without a usable ``.rpl`` are skipped with a note."""
    pairs = []
    for path in episode_paths(data_dir):
        with np.load(path, allow_pickle=True) as d:
            replay = str(d["replay"].item()) if "replay" in d.files else ""
        if not replay:
            print(f"[regenerate] {path.name}: no replay recorded; skipping")
            continue
        src = data_dir / "replays" / Path(replay).name
        if not src.is_file():
            src = game_save_dir() / replay
        if not src.is_file():
            print(f"[regenerate] {path.name}: replay {replay!r} not found; skipping")
            continue
        pairs.append((path, src))
    return pairs


def _find_pairs(folder) -> list:
    """The ``(episode_path, rpl_path)`` pairs to regenerate: a dataset
    directory (each episode's recorded ``.rpl``, in episode order), a folder
    of ``.rpl`` (or a single ``.rpl``, no episode), or a directory with a
    ``replays/`` subfolder."""
    folder = Path(folder)
    if folder.suffix == ".rpl":
        return [(None, folder)]
    if episode_paths(folder):
        return _dataset_pairs(folder)
    candidates = sorted(folder.glob("*.rpl"))
    if not candidates:
        candidates = sorted(folder.glob("replays/*.rpl"))
    return [(None, c) for c in candidates if c.is_file()]


def _game_relative(replay: Path) -> str:
    """Ensure ``replay`` is in the game's ``replays/`` folder (the ``.rpl``
    must be there for both the in-game playback and the offline parse); return
    its game-relative path (``replays/<name>.rpl``)."""
    game_replays = game_save_dir() / "replays"
    game_replays.mkdir(parents=True, exist_ok=True)
    dest = game_replays / replay.name
    if dest.resolve() != replay.resolve():
        shutil.copy2(replay, dest)
    return f"replays/{replay.name}"


def _episode_index(path) -> int | None:
    """The index encoded in an ``episode_NNNNNN.npz`` filename, or None when
    the name doesn't match (e.g. a bare ``.rpl`` has no source episode)."""
    stem = Path(path).stem
    prefix = "episode_"
    if stem.startswith(prefix) and stem[len(prefix):].isdigit():
        return int(stem[len(prefix):])
    return None


def _out_index(ep_path, next_index: int) -> tuple:
    """The output index for a regenerated episode: the source episode's index
    (so a dataset keeps its numbering, gaps and all) or the next free index
    for a bare ``.rpl``. Returns ``(index, new_next_index)``."""
    idx = _episode_index(ep_path)
    if idx is None:
        idx = next_index
    return idx, max(next_index, idx + 1)


# -- offline regeneration ----------------------------------------------------
def _align_episode(air_ep, air_rpl) -> tuple:
    """The offset (s) at which the episode's airspeed series matches the
    ``.rpl``'s (resampled to the episode's 20 Hz grid), by sliding-window
    correlation. Returns ``(offset_s, corr)``; ``(None, 0.0)`` when the ``.rpl``
    is shorter than the episode.

    The ``.rpl`` recording starts at the episode's start in the game, but a
    curated episode may begin later (the leading idle is truncated away), so
    the episode sits at some offset inside the ``.rpl``. A (nearly)
    stationary episode has no signal to align on: offset 0 is assumed."""
    T = len(air_ep)
    if len(air_rpl) < T:
        return None, 0.0
    a = air_ep - air_ep.mean()
    an = float(np.linalg.norm(a))
    if an < 1e-9:
        return 0.0, 1.0
    a /= an
    win = np.lib.stride_tricks.sliding_window_view(air_rpl, T)
    w = win - win.mean(axis=1, keepdims=True)
    wn = np.linalg.norm(w, axis=1)
    corr = np.zeros(len(win))
    ok = wn > 1e-9
    corr[ok] = (w[ok] @ a) / wn[ok]
    i = int(np.argmax(corr))
    return i * TICK_DT, float(corr[i])


def _resample_rpl(rpl, times):
    """The ``.rpl``'s state/electrics on an explicit time grid (seconds).

    pos/vel are linearly interpolated; dir/up (from the quaternion) and the
    electrics come from the nearest frame. Returns ``(states, electrics)``
    where ``states`` is a list of State-sensor-shaped dicts (pos/vel/dir/up/
    time) and ``electrics`` a list of the raw recorded dicts (or None)."""
    pos = np.stack([np.interp(times, rpl.t, rpl.pos[:, k]) for k in range(3)],
                   axis=1)
    vel = np.stack([np.interp(times, rpl.t, rpl.vel[:, k]) for k in range(3)],
                   axis=1)
    idx = np.clip(np.searchsorted(rpl.t, times), 0, len(rpl.t) - 1)
    states = []
    for i in range(len(times)):
        fwd, _left, up = quat_to_axes(rpl.quat[idx[i]])
        states.append({"pos": pos[i], "vel": vel[i], "dir": fwd, "up": up,
                       "time": float(times[i])})
    return states, [rpl.electrics[i] for i in idx]


def _regenerate_offline(ep, rpl, navigraph):
    """Rebuild one episode offline from its ``.rpl``. ``ep`` is a loaded
    episode (the :func:`data.curate._load_episode` layout) or None (a bare
    ``.rpl`` with no original episode to reuse); ``rpl`` its RplTrajectory;
    ``navigraph`` the cached road network. Returns the rebuilt episode dict
    (same layout), or None when the ``.rpl`` can't be aligned to the episode.

    The ``.rpl`` is the ground truth for the driving: the
    pose/electrics-derived channels, the road BEV and the reward are
    recomputed from it. The channels the ``.rpl`` doesn't carry (the force
    feedback, the per-step damage) are reused from ``ep``. The ``bev``
    channel is the corrected LiDAR: the original episode's LiDAR BEV,
    corrected against the road network (its step i is the same physics tick
    as this one), or - for a bare ``.rpl`` with no LiDAR - synthesized from
    the road BEV (road surface = road level, off-road = obstacle), the same
    convention the env's corrected LiDAR produces."""
    obs_in = ep["obs"] if ep is not None else {}
    bev_in = obs_in.get("bev")   # the original LiDAR BEV (T, 1, n, n) or None
    if ep is not None:
        T = len(ep["reward"])
        air_ep = np.asarray(obs_in.get("airspeed", np.zeros(T))).reshape(-1) \
            * AIRSPEED_MAX_KPH
    else:
        T = int(round(rpl.t[-1] / TICK_DT)) + 1
        air_ep = None
    # the .rpl's airspeed on the episode's 20 Hz grid, for the alignment
    air_rpl = np.interp(np.arange(0, rpl.t[-1] + TICK_DT, TICK_DT),
                        rpl.t, np.linalg.norm(rpl.vel, axis=1) * 3.6)
    if air_ep is None:
        offset, corr = 0.0, 1.0
    else:
        offset, corr = _align_episode(air_ep, air_rpl)
        if offset is None or corr < ALIGN_MIN_CORR:
            print(f"[regenerate] no reliable .rpl alignment (corr {corr:.2f}); "
                  f"skipping")
            return None
    times = np.minimum(offset + np.arange(T) * TICK_DT, rpl.t[-1])
    states, electrics = _resample_rpl(rpl, times)

    obs = {k: [] for k in ("rpm", "imu_acc", "imu_angvel", "drift_angle",
                           "airspeed", "is_shifting", "gear")}
    ffb, bev, road_bev = [], [], []
    action_hist, reward_hist, info_hist = [], [], []
    prev_state = None
    prev_pos = None
    for i in range(T):
        state = states[i]
        e = electrics[i] or {}
        drift = drift_from_state(state)
        acc, angvel = derive_imu(prev_state, state)
        if drift["airSpeed"] <= 1.0:   # mirrors rig.get_obs
            acc = np.zeros(3, dtype=np.float32)
            angvel = np.zeros(3, dtype=np.float32)
        acc, angvel = to_cpu(acc), to_cpu(angvel)
        prev_state = state_snapshot(state)
        is_shifting = bool(e.get("is_shifting", e.get("isShifting", 0)))
        gear = float(e.get("gear", 0.0))
        pb = float(e.get("parkingbrake_input", 0.0))
        th = float(e.get("throttle_input", 0.0))
        obs["rpm"].append(np.clip(e.get("rpm", 0.0) / e["maxrpm"], 0.0, 1.0)
                          if e.get("maxrpm") else 0.0)
        obs["imu_acc"].append(acc)
        obs["imu_angvel"].append(angvel)
        obs["drift_angle"].append(np.clip(drift["angle"] / ANGLE_MAX, -1.0, 1.0))
        obs["airspeed"].append(min(AIRSPEED_MAX_KPH, drift["airSpeed"])
                               / AIRSPEED_MAX_KPH)
        obs["is_shifting"].append(1.0 if is_shifting else 0.0)
        obs["gear"].append((gear + 1.0) / (MAX_GEAR + 1.0))
        # the recorded action (the drive env's human-mode applied_action)
        steering = float(e.get("steering", 0.0)) / DEFAULT_STEERING_WHEEL_LOCK
        action = np.concatenate([
            np.array([
                float(np.clip(steering, -1.0, 1.0)),
                float(np.clip(th - float(e.get("brake_input", 0.0)), -1.0, 1.0)),
                float(np.clip(pb, -1.0, 1.0)),
            ], dtype=np.float32),
            gear_to_action(int(round(gear))),
        ])
        # the road BEV (1.0 = drivable) around the car, from the cached
        # network; an all-zero grid when the rasterization fails
        try:
            fwd, left = car_grid_axes(state)
            road = navigraph.to_bev(state["pos"], fwd=fwd, left=left)
            centerline_dist = road_centerline_segments(
                navigraph, state["pos"], fwd=fwd, left=left)[1]
        except Exception:
            road = np.zeros((BEV_N, BEV_N), float)
            centerline_dist = float("nan")
        road = to_cpu(road)
        # on-road determination, mirroring the live env (rig.on_road): the
        # fast-track reuses the centerline distance just computed (NaN
        # compares False and falls through to the strict check), the strict
        # navigraph check decides the rest; off-road scales the reward down
        try:
            on_road = (centerline_dist < ON_ROAD_FAST_TRACK_DIST
                       or navigraph.is_on_road(state["pos"]))
        except Exception:
            on_road = None
        # the reward, with the per-step damage reused from the original
        # episode (the .rpl doesn't record it)
        distance = (0.0 if prev_pos is None
                    else float(np.linalg.norm(state["pos"] - prev_pos)))
        prev_pos = state["pos"]
        dmg_delta = 0.0
        if ep is not None:
            dmg_delta = float(ep["info"][i].get("reward_components", {})
                              .get("dmg_delta", 0.0))
        reward, comps = compute_reward(
            distance, drift["scoreAdded"], dmg_delta, centerline_dist,
            offroad=on_road is False, reversing=drift["reversing"])
        if "ffb" in obs_in:
            ffb.append(obs_in["ffb"][i])
        else:
            ffb.append(np.array([0.0], np.float32))
        road_bev.append(road[None, ...].astype(np.float32))
        # the corrected LiDAR BEV: the original episode's LiDAR, corrected
        # against the road network (step i is the same physics tick), or -
        # for a bare .rpl with no LiDAR - the road network itself (road
        # surface = road level, off-road = obstacle)
        lidar = (bev_in[i] if bev_in is not None
                 else ((1.0 - road) * 255.0).astype(np.uint8)[None, ...])
        bev.append(to_cpu(correct_lidar_bev(lidar, road)))
        action_hist.append(action)
        reward_hist.append(reward)
        info_hist.append({"reward_components": comps, "applied_action": action})

    out_obs = {
        "rpm":         np.array(obs["rpm"], np.float32).reshape(T, 1),
        "imu_acc":     np.array(obs["imu_acc"], np.float32),
        "imu_angvel":  np.array(obs["imu_angvel"], np.float32),
        "drift_angle": np.array(obs["drift_angle"], np.float32).reshape(T, 1),
        "airspeed":    np.array(obs["airspeed"], np.float32).reshape(T, 1),
        "is_shifting": np.array(obs["is_shifting"], np.float32).reshape(T, 1),
        "gear":        np.array(obs["gear"], np.float32).reshape(T, 1),
        "ffb":         np.array(ffb, np.float32).reshape(T, 1),
        "bev":         np.array(bev, np.uint8),
        "road_bev":    np.array(road_bev, np.float32),
    }
    meta = {}
    if ep is not None:
        meta = {k: v for k, v in ep["meta"].items()
                if k not in ("space_obs", "space_act", "env_meta")}
    meta["ep_reward"] = float(np.sum(reward_hist))
    meta["ep_length"] = T
    return {
        "obs": out_obs,
        "action": np.array(action_hist, np.float32),
        "reward": np.array(reward_hist, np.float32),
        "terminated": (ep["terminated"] if ep is not None
                       else np.zeros(T, bool)),
        "truncated": (ep["truncated"] if ep is not None
                      else np.ones(T, bool)),
        "info": info_hist,
        "meta": meta,
    }


def _save_offline(path, ep, space_obs, space_act, env_meta):
    """Write a rebuilt episode in the collector's layout (self-contained: the
    current observation/action spaces are embedded)."""
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
    payload.update({
        "space_obs": np.frombuffer(pickle.dumps(space_obs), np.uint8),
        "space_act": np.frombuffer(pickle.dumps(space_act), np.uint8),
        "env_meta": np.frombuffer(pickle.dumps(env_meta), np.uint8),
    })
    np.savez_compressed(path, **payload)


def _load_navigraph(path, level):
    """The cached road network; fetched from the game (and cached) when the
    file is missing - the only game use in offline mode."""
    path = Path(path)
    if path.is_file():
        return Navigraph.load(path)
    print(f"[regenerate] no road cache at {path}; fetching from the game")
    from beamng_tech_env.core.game import GameDriver
    driver = GameDriver()
    try:
        driver.connect()
        driver.load_scenario(level, "Regenerate")
        driver.start_scenario()
        nav = Navigraph.from_beamng(driver.beamng)
        path.parent.mkdir(parents=True, exist_ok=True)
        nav.save(path)
        print(f"[regenerate] cached the road network ({nav.num_segments} "
              f"segments) -> {path}")
        return nav
    finally:
        driver.close()


def _run_offline(pairs, args) -> None:
    # the spaces are read from a game-free env construction (the constructor
    # only builds the spaces and the driver/rig objects)
    from beamng_tech_env.envs.beamng_tech import BeamNGTechEnv
    env = BeamNGTechEnv()
    space_obs, space_act = env.observation_space, env.action_space
    roads = (Path(args.roads) if args.roads
             else Path(args.data).parent / "roads" / f"{args.level}.npz")
    navigraph = _load_navigraph(roads, args.level)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    next_index = len(episode_paths(out_dir))
    done = skipped = 0
    try:
        for ep_path, rpl_path in pairs:
            print(f"[{done + skipped + 1}/{len(pairs)}] {rpl_path.name}",
                  flush=True)
            ep = _load_episode(ep_path) if ep_path is not None else None
            rpl = RplTrajectory(rpl_path)
            out = _regenerate_offline(ep, rpl, navigraph)
            if out is None:
                skipped += 1
                continue
            # keep the source episode's numbering when regenerating a dataset;
            # a bare .rpl has no source index, so it takes the next free one
            idx, next_index = _out_index(ep_path, next_index)
            path = out_dir / f"episode_{idx:06d}.npz"
            _save_offline(path, out, space_obs, space_act,
                          {"env_id": ENV_ID})
            done += 1
            print(f"    len={len(out['reward']):5d} "
                  f"reward={float(out['reward'].sum()):10.2f} -> {path.name}",
                  flush=True)
    except KeyboardInterrupt:
        print(f"\n[regenerate] interrupted; keeping the {done} completed "
              f"episode(s) (the in-progress one is discarded)")
    print(f"[regenerate] {done}/{len(pairs)} episodes -> {args.out} "
          f"({skipped} skipped)", flush=True)


def _reuse_ffb(out_path, ep_path):
    """Replace a regenerated episode's FFB with the source episode's,
    aligned by airspeed, in place.

    The live FFB read is always 0: the replayed vehicle is a clone the
    game's hydros module doesn't drive, so forceAtDriverNorm has no force to
    report. The source episode's FFB was recorded during the original
    driving the .rpl replays, so its step i is the force at the same
    physics state. A no-op when the source has no usable FFB or the
    alignment is unreliable (the live FFB is kept)."""
    if ep_path is None:
        return
    src = _load_episode(ep_path)
    ffb_src = np.asarray(src["obs"].get("ffb"), dtype=np.float32).reshape(-1)
    if ffb_src.size == 0 or not np.any(ffb_src):
        return
    with np.load(out_path, allow_pickle=True) as d:
        if "obs_ffb" not in d.files:
            return
        air_out = np.asarray(d["obs_airspeed"]).reshape(-1) * AIRSPEED_MAX_KPH
        payload = {k: d[k] for k in d.files}
    air_src = np.asarray(src["obs"]["airspeed"]).reshape(-1) * AIRSPEED_MAX_KPH
    # the offset (steps) of the shorter series inside the longer one: the
    # source may start part-way into the live episode (the leading idle is
    # truncated away) or precede it by a few ticks (the reset observation is
    # recorded before the first tick)
    if len(air_out) >= len(air_src):
        offset, corr = _align_episode(air_src, air_out)
        delta = -int(round(offset / TICK_DT)) if offset is not None else None
    else:
        offset, corr = _align_episode(air_out, air_src)
        delta = int(round(offset / TICK_DT)) if offset is not None else None
    if delta is None or corr < ALIGN_MIN_CORR:
        print(f"[regenerate] {Path(out_path).name}: no reliable FFB "
              f"alignment (corr {corr:.2f}); keeping the live FFB")
        return
    j = np.arange(len(air_out))
    i = j + delta
    ffb_out = np.zeros(len(air_out), dtype=np.float32)
    mask = (i >= 0) & (i < ffb_src.size)
    ffb_out[mask] = ffb_src[i[mask]]
    payload["obs_ffb"] = ffb_out.reshape(-1, 1)
    np.savez_compressed(out_path, **payload)


def _run_live(pairs, args) -> None:
    import beamng_tech_env  # noqa: F401  (registers the env ids)
    from beamng_tech_env.envs.replay_env import BeamNGReplayEnv

    env = BeamNGReplayEnv()
    collector = TrajectoryCollector(env, out_dir=args.out,
                                    env_meta={"env_id": ENV_ID})
    stats = []
    try:
        for ep_path, replay in pairs:
            rel = _game_relative(replay)
            print(f"[{len(stats) + 1}/{len(pairs)}] {replay.name}", flush=True)
            # keep the source episode's numbering when regenerating a dataset;
            # a bare .rpl has no source index, so the collector's own counter
            # is left to run on
            idx = _episode_index(ep_path)
            if idx is not None:
                collector._episode_index = idx
            # human=True: the replay drives the car, so the recorded action is
            # read back from info["applied_action"] (the electrics sensor)
            result = collector.collect_episode(
                None, max_steps=args.max_steps, human=True,
                options={"replay": rel}, replay=rel)
            stats.append(result)
            # the live FFB read is always 0 (the replayed vehicle is a clone
            # the hydros module doesn't drive): reuse the source episode's
            # FFB, aligned by airspeed, like offline mode does
            _reuse_ffb(result["path"], ep_path)
            tag = "term" if result["terminated"] else "trunc"
            print(f"    {tag} len={result['length']:5d} "
                  f"reward={result['reward']:10.2f} -> {Path(result['path']).name}",
                  flush=True)
    except KeyboardInterrupt:
        print(f"\n[regenerate] interrupted; keeping the {len(stats)} completed "
              f"episode(s) (the in-progress replay is discarded)")
    finally:
        env.close()
    print(f"[regenerate] {len(stats)}/{len(pairs)} episodes -> {args.out}",
          flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate dataset episodes from .rpl replays: offline "
                    "by default (the .rpl drives the rebuild, the FFB is "
                    "reused from the original episode, the beams come from "
                    "the road network sensor, and the road BEV and reward "
                    "are recomputed); --lidar plays each .rpl in the game "
                    "and reads the LiDAR live")
    parser.add_argument("--data", required=True,
                        help="the dataset directory (each episode's npz "
                             "records its .rpl; resolved in episode order); "
                             "a folder of .rpl files (or a single .rpl) is "
                             "also accepted")
    parser.add_argument("--out", required=True,
                        help="output directory for the regenerated episodes "
                             "(appended to: indices continue past existing files)")
    parser.add_argument("--lidar", action="store_true",
                        help="play each .rpl in the game and read the LiDAR "
                             "BEV live (the FFB is reused from the original "
                             "episode in both modes: the replayed vehicle is "
                             "a clone the hydros module doesn't drive, so "
                             "the live read is always 0; the default is "
                             "offline: the pose/electrics come from the "
                             ".rpl, the beams come from the road network "
                             "sensor, and only the road network comes from "
                             "the game)")
    parser.add_argument("--roads", default=None,
                        help="the cached road network .npz (default: "
                             "<data root>/roads/<level>.npz); fetched from "
                             "the game and cached when missing (offline mode)")
    parser.add_argument("--level", default=LEVEL,
                        help=f"the level the road network is for (default: {LEVEL})")
    parser.add_argument("--max-steps", type=int, default=10_000,
                        help="max steps per replay, --lidar mode only (default: 10000)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only regenerate the first N replays (default: all)")
    parser.add_argument("--episode", type=int, default=None,
                        help="only regenerate the episode with this index "
                             "(the source episode_NNNNNN.npz number; "
                             "default: all)")
    args = parser.parse_args(argv)

    # make the sibling packages importable when run from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    pairs = _find_pairs(args.data)
    if not pairs:
        parser.error(f"no episodes or .rpl files found in {args.data}")
    if args.episode is not None:
        pairs = [p for p in pairs if _episode_index(p[0]) == args.episode]
        if not pairs:
            parser.error(f"episode {args.episode} not found in {args.data}")
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[regenerate] {len(pairs)} episodes -> {args.out}"
          f"{' (lidar)' if args.lidar else ''}", flush=True)

    if args.lidar:
        _run_live(pairs, args)
    else:
        _run_offline(pairs, args)


if __name__ == "__main__":
    main()
