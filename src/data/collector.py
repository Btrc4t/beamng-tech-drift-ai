"""Record environment trajectories (transitions) to disk for offline training.

A *trajectory* here is a sequence of gymnasium transitions
``(obs, action, reward, terminated, truncated, info)``. Each completed
episode is written to its own ``.npz`` file so a partially-collected dataset
is never lost. Every file is self-contained: the observation/action spaces
and env metadata (needed to rebuild a policy or replay buffer later) are
embedded in each episode, so a dataset is just a directory of ``.npz`` files.

The collector is deliberately environment-agnostic: it works with any
gymnasium env and any policy. A policy is anything that can turn an
observation into an action:

- ``None``            -> sample uniformly from the action space
- a Stable-Baselines3 model (has ``.predict``)
- a plain ``callable(obs) -> action``

Usage (standalone, the main path)::

    from data.collector import TrajectoryCollector

    collector = TrajectoryCollector(env, out_dir="data/run1")
    collector.collect_episodes(n_episodes=20, policy=model, max_steps=2000)

Usage (as a gym wrapper, e.g. to record while training online)::

    env = TrajectoryRecorder(env, out_dir="data/online")
    # ... run env.reset()/env.step() normally; episodes are saved on done
"""
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np


def _obs_to_arrays(obs: Any) -> dict:
    """Flatten an observation (dict or array) into ``{name: ndarray}``.

    A Dict observation keeps its keys (``"bev"``, ``"rpm"``, ...); a plain
    array is stored under ``"obs"``. Each value is copied so later in-place
    mutation by the env cannot corrupt what we recorded.
    """
    if isinstance(obs, dict):
        return {k: np.array(v, copy=True) for k, v in obs.items()}
    return {"obs": np.array(obs, copy=True)}


def _arrays_to_obs(arrays: dict) -> Any:
    """Inverse of :func:`_obs_to_arrays`: rebuild the env observation."""
    if set(arrays.keys()) == {"obs"}:
        return arrays["obs"]
    return {k: v for k, v in arrays.items()}


def _policy_fn(policy: Optional[Callable], action_space: gym.spaces.Space) -> Callable:
    """Wrap the many ways a policy can be given into ``fn(obs) -> action``."""
    if policy is None:
        return lambda obs: action_space.sample()
    if hasattr(policy, "predict"):
        # Stable-Baselines3 (and any object exposing predict(obs, deterministic))
        return lambda obs: policy.predict(obs, deterministic=True)[0]
    return policy


def _space_payload(observation_space, action_space, env_meta) -> dict:
    """Dataset-level fields embedded in every episode npz.

    The spaces are stored as pickled bytes (a plain ``np.array(space)`` would
    iterate a Dict space's keys), so each episode file is self-contained and
    no separate meta file is needed.
    """
    return {
        "space_obs": np.frombuffer(pickle.dumps(observation_space), dtype=np.uint8),
        "space_act": np.frombuffer(pickle.dumps(action_space), dtype=np.uint8),
        "env_meta": np.frombuffer(pickle.dumps(env_meta), dtype=np.uint8),
    }


def _stack_poses(pose_hist, t):
    """Stack a list of per-step ``(pos, fwd, left)`` tuples (or ``None``) into
    three ``(t, 3)`` float32 arrays (NaN where a pose is None), for the
    offline video's route render (see collect_episode / TrajectoryRecorder)."""
    bev_pos = np.full((t, 3), np.nan, dtype=np.float32)
    bev_fwd = np.full((t, 3), np.nan, dtype=np.float32)
    bev_left = np.full((t, 3), np.nan, dtype=np.float32)
    for i, p in enumerate(pose_hist):
        if p is not None:
            bev_pos[i], bev_fwd[i], bev_left[i] = p
    return bev_pos, bev_fwd, bev_left


def _route_payload(episode):
    """The route geometry + pose arrays for an episode's npz payload: the
    world-coordinate route geometry (``route_centerline`` / ``route_waypoints``
    / ``route_gate_s``, omitted when the env has no route) and the per-step BEV
    pose / route progress (``bev_pos`` / ``bev_fwd`` / ``bev_left`` /
    ``route_progress``), recorded for the offline video's route render."""
    payload = {}
    geometry = episode.get("route_geometry")
    if geometry:
        for key in ("centerline", "waypoints", "gate_s"):
            if geometry.get(key) is not None:
                payload[f"route_{key}"] = np.asarray(
                    geometry[key], np.float32)
    for key in ("bev_pos", "bev_fwd", "bev_left", "route_progress"):
        if key in episode:
            payload[key] = np.asarray(episode[key], np.float32)
    return payload


class TrajectoryCollector:
    """Collect episodes from an env and save each one as a ``.npz`` file.

    Parameters
    ----------
    env:
        The gymnasium environment to collect from (wrapped or raw).
    out_dir:
        Directory that will hold one ``episode_*.npz`` per episode. Created if
        missing.
    env_meta:
        Optional dict of global metadata (e.g. ``{"env_id": ...}``) embedded in
        each episode alongside the spaces.
    """

    def __init__(self, env: gym.Env, out_dir, env_meta: Optional[dict] = None,
                 replay_recorder=None):
        self.env = env
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env_meta = dict(env_meta or {})
        # optional BeamNGReplayRecorder (or anything with start/stop) that
        # captures the game's native .rpl replay in lock-step with each episode
        self._replay_recorder = replay_recorder
        self._episode_index = self._next_episode_index()

    # -- persistence -------------------------------------------------------
    def _next_episode_index(self) -> int:
        # dataset.dataset imports this module, so import it lazily; the
        # numbering spans a split dataset's train/ and validation/ folders
        from data.dataset import episode_paths

        return len(episode_paths(self.out_dir))

    def _save_episode(self, episode: dict, meta: dict) -> Path:
        path = self.out_dir / f"episode_{self._episode_index:06d}.npz"
        self._episode_index += 1
        payload = {}
        for key, arr in episode["obs"].items():
            payload[f"obs_{key}"] = arr
        payload["action"] = episode["action"]
        payload["reward"] = episode["reward"]
        payload["terminated"] = episode["terminated"]
        payload["truncated"] = episode["truncated"]
        # info is a list of dicts; store as one object array
        payload["info"] = np.array(episode["info"], dtype=object)
        # the route geometry + per-step BEV pose / progress, recorded for the
        # offline video's route render (see collect_episode)
        payload.update(_route_payload(episode))
        for key, value in meta.items():
            payload[key] = np.array(value)
        payload.update(_space_payload(self.observation_space, self.action_space,
                                      self.env_meta))
        np.savez_compressed(path, **payload)
        return path

    # -- collection --------------------------------------------------------
    def collect_episode(self, policy_fn: Callable, max_steps: int = 10_000,
                        render: bool = False, seed: Optional[int] = None,
                        human: bool = False, options: Optional[dict] = None,
                        replay: Optional[str] = None) -> dict:
        """Run one episode, recording every transition. Returns episode stats.

        ``human=True``: an external driver (a person, or the game's built-in
        AI in synthetic mode) drives the car. The env is stepped with a dummy
        action and the applied action is taken from
        ``info["applied_action"]`` (the env reads it back from the game).
        ``options`` is passed through to ``env.reset`` (e.g. the replay env's
        per-episode ``{"replay": path}``). ``replay`` names the source ``.rpl``
        for the episode meta when no replay recorder is attached (the replay
        env plays an existing replay rather than recording a new one).
        """
        episode_index = self._episode_index
        # reset first so the game (and its beamng connection) is ready; the
        # replay then starts from the freshly spawned car, not the respawn
        obs, info = self.env.reset(seed=seed, options=options)
        replay_name = None
        if self._replay_recorder is not None:
            replay_name = self._replay_recorder.start(episode_index)
        if render:
            self.env.render()

        # The route's world-coordinate geometry (centerline + waypoints),
        # recorded once per episode: the route is fixed for the episode, so the
        # offline video can place it in the BEV from the per-step pose (see
        # _save_episode). The rig is reached through the (possibly wrapped)
        # env; None for envs without a route (the replay env, a route-less run).
        rig = getattr(getattr(self.env, "unwrapped", self.env), "rig", None)
        route_geometry = None
        if rig is not None:
            fn = getattr(rig, "route_geometry", None)
            if fn is not None:
                route_geometry = fn()

        obs_hist = []
        action_hist = []
        reward_hist = []
        terminated_hist = []
        truncated_hist = []
        info_hist = []
        pose_hist = []  # per-step (pos, fwd, left) for the offline video's route render
        progress_hist = []  # per-step route progress (m) for the gate-state replay
        total_reward = 0.0

        for _ in range(max_steps):
            # The BEV pose and route progress at the current obs's time
            # (before the step), so they align with obs_hist; None / NaN for
            # envs without a rig / route.
            pose = None
            progress = float("nan")
            if rig is not None:
                pose_fn = getattr(rig, "bev_pose", None)
                if pose_fn is not None:
                    pose = pose_fn()
                progress_fn = getattr(rig, "route_progress", None)
                if progress_fn is not None:
                    progress = progress_fn()
            if human:
                # an external driver (a person, or the game's built-in AI)
                # drives: step with a dummy action and take the applied
                # action the env read back from the game
                action = np.zeros(self.action_space.shape, dtype=np.float32)
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
                action = np.asarray(step_info.get("applied_action", action),
                                    dtype=np.float32)
            else:
                action = policy_fn(obs)
                next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            obs_hist.append(_obs_to_arrays(obs))
            pose_hist.append(pose)
            progress_hist.append(progress)
            action_hist.append(np.array(action))
            reward_hist.append(float(reward))
            terminated_hist.append(bool(terminated))
            truncated_hist.append(bool(truncated))
            info_hist.append(step_info)
            total_reward += float(reward)
            obs = next_obs
            if render:
                self.env.render()
            if terminated or truncated:
                break

        # stack the per-step observation dicts into one array per key
        keys = list(obs_hist[0].keys()) if obs_hist else []
        merged_obs = {k: np.stack([step[k] for step in obs_hist]) for k in keys}
        # the per-step BEV pose (pos, fwd, left) and route progress as (T, 3) /
        # (T,) float32 arrays (NaN where the env has no rig / route), so the
        # offline video can place the route geometry in the BEV with the same
        # transform the live window uses (see _save_episode / data.curate)
        bev_pos, bev_fwd, bev_left = _stack_poses(pose_hist, len(reward_hist))
        episode = {
            "obs": merged_obs,
            "action": (np.stack(action_hist) if action_hist
                       else np.empty((0, self.action_space.shape))),
            "reward": np.array(reward_hist, dtype=np.float32),
            "terminated": np.array(terminated_hist, dtype=bool),
            "truncated": np.array(truncated_hist, dtype=bool),
            "info": info_hist,
            "route_geometry": route_geometry,
            "bev_pos": bev_pos,
            "bev_fwd": bev_fwd,
            "bev_left": bev_left,
            "route_progress": np.array(progress_hist, dtype=np.float32),
        }

        # stop the replay before saving so the file is complete on disk
        if self._replay_recorder is not None:
            replay_name = self._replay_recorder.stop(episode_index) or replay_name

        meta = {
            "ep_reward": np.float32(total_reward),
            "ep_length": np.int32(len(reward_hist)),
            "ep_terminated": np.bool_(bool(terminated_hist[-1]) if terminated_hist else False),
            "ep_truncated": np.bool_(bool(truncated_hist[-1]) if truncated_hist else False),
            "seed": np.int64(seed if seed is not None else -1),
            "replay": np.array(replay_name or replay or ""),
            "spawn": np.array(info.get("spawn", "")),
            "timestamp": np.array(datetime.now(timezone.utc).isoformat()),
        }
        path = self._save_episode(episode, meta)
        return {"path": str(path), "reward": total_reward,
                "length": len(reward_hist), "terminated": meta["ep_terminated"],
                "truncated": meta["ep_truncated"], "replay": replay_name or replay}

    def collect_episodes(self, n_episodes: int, policy=None,
                         max_steps: int = 10_000, render: bool = False,
                         verbose: bool = True,
                         max_duration: Optional[float] = None) -> list:
        """Collect episodes and save each. Returns per-episode stats.

        ``policy`` may be ``None`` (random actions), a Stable-Baselines3
        model, a ``callable(obs) -> action``, the string ``"human"`` (a
        person drives; the env must support human mode and return
        ``info["applied_action"]``), or the string ``"ai"`` (the game's
        built-in AI drives, the env's synthetic mode; same read-back
        mechanism). ``n_episodes=0`` collects until
        interrupted (Ctrl+C); completed episodes are kept.
        ``max_duration`` stops collection after that many seconds of
        wall-clock time (the in-progress episode is discarded); when both
        limits are given, whichever is reached first wins.
        """
        human = (policy in ("human", "ai"))
        policy_fn = None if human else _policy_fn(policy, self.action_space)
        stats = []
        i = 0
        t_start = time.time()
        try:
            while True:
                i += 1
                result = self.collect_episode(policy_fn, max_steps=max_steps,
                                              render=render, human=human)
                stats.append(result)
                if verbose:
                    tag = "term" if result["terminated"] else "trunc"
                    replay = f"  replay={Path(result['replay']).name}" if result.get("replay") else ""
                    count = f"[{i}/{n_episodes}]" if n_episodes else f"[{i}]"
                    print(f"{count} {tag} len={result['length']:5d} "
                          f"reward={result['reward']:10.2f} -> {Path(result['path']).name}{replay}",
                          flush=True)
                if n_episodes and i >= n_episodes:
                    break
                if (max_duration is not None
                        and time.time() - t_start >= max_duration):
                    if verbose:
                        print(f"\n[collect] reached the "
                              f"{max_duration / 60:.0f} min limit; keeping the "
                              f"{len(stats)} completed episode(s)")
                    break
        except KeyboardInterrupt:
            if verbose:
                print(f"\n[collect] interrupted; keeping the {len(stats)} completed "
                      f"episode(s) (the in-progress episode is discarded)")
        return stats


class TrajectoryRecorder(gym.ObservationWrapper):
    """Gym wrapper that records episodes as you step, saving on episode end.

    Drop it around any env to capture data while you train or roll out
    online. Each finished episode (terminated or truncated) is written to
    ``out_dir`` immediately, so an interrupted run keeps everything up to that
    point.
    """

    def __init__(self, env: gym.Env, out_dir, env_meta: Optional[dict] = None,
                 replay_recorder=None):
        super().__init__(env)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env_meta = dict(env_meta or {})
        self._replay_recorder = replay_recorder
        self._replay_name = None
        self._spawn = ""
        self._episode_index = len(list(self.out_dir.glob("episode_*.npz")))
        self._last_obs = None
        self._route_geometry = None
        self._buf = self._new_buffer()

    def _new_buffer(self) -> dict:
        return {"obs": [], "action": [], "reward": [], "terminated": [],
                "truncated": [], "info": [], "pose": [], "progress": []}

    def _rig(self):
        # the env's rig, reached through the (possibly wrapped) env; None for
        # envs without one (the replay env, a route-less run)
        return getattr(getattr(self.env, "unwrapped", self.env), "rig", None)

    def observation(self, observation):
        # pass-through: we record but do not transform the observation
        return observation

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._last_obs = obs
        self._buf = self._new_buffer()
        self._replay_name = None
        self._spawn = info.get("spawn", "")
        # the route's world-coordinate geometry, recorded once per episode
        # (see TrajectoryCollector.collect_episode)
        self._route_geometry = None
        rig = self._rig()
        if rig is not None:
            fn = getattr(rig, "route_geometry", None)
            if fn is not None:
                self._route_geometry = fn()
        if self._replay_recorder is not None:
            self._replay_name = self._replay_recorder.start(self._episode_index)
        return obs, info

    def step(self, action):
        # the BEV pose / route progress at the current (pre-step) state, so
        # they align with self._last_obs (the pre-step observation)
        rig = self._rig()
        pose = None
        progress = float("nan")
        if rig is not None:
            pose_fn = getattr(rig, "bev_pose", None)
            if pose_fn is not None:
                pose = pose_fn()
            progress_fn = getattr(rig, "route_progress", None)
            if progress_fn is not None:
                progress = progress_fn()
        obs, reward, terminated, truncated, info = super().step(action)
        self._buf["obs"].append(_obs_to_arrays(self._last_obs))
        self._buf["pose"].append(pose)
        self._buf["progress"].append(progress)
        self._buf["action"].append(np.array(action))
        self._buf["reward"].append(float(reward))
        self._buf["terminated"].append(bool(terminated))
        self._buf["truncated"].append(bool(truncated))
        self._buf["info"].append(info)
        self._last_obs = obs
        if terminated or truncated:
            self._flush()
        return obs, reward, terminated, truncated, info

    def _flush(self) -> None:
        if not self._buf["reward"]:
            return
        if self._replay_recorder is not None:
            self._replay_name = self._replay_recorder.stop(self._episode_index) or self._replay_name
        keys = self._buf["obs"][0].keys()
        bev_pos, bev_fwd, bev_left = _stack_poses(
            self._buf["pose"], len(self._buf["reward"]))
        episode = {
            "obs": {k: np.stack([s[k] for s in self._buf["obs"]]) for k in keys},
            "action": np.stack(self._buf["action"]),
            "reward": np.array(self._buf["reward"], dtype=np.float32),
            "terminated": np.array(self._buf["terminated"], dtype=bool),
            "truncated": np.array(self._buf["truncated"], dtype=bool),
            "info": self._buf["info"],
            "route_geometry": self._route_geometry,
            "bev_pos": bev_pos,
            "bev_fwd": bev_fwd,
            "bev_left": bev_left,
            "route_progress": np.array(self._buf["progress"], dtype=np.float32),
        }
        total = float(np.sum(self._buf["reward"]))
        meta = {
            "ep_reward": np.float32(total),
            "ep_length": np.int32(len(self._buf["reward"])),
            "ep_terminated": np.bool_(bool(self._buf["terminated"][-1])),
            "ep_truncated": np.bool_(bool(self._buf["truncated"][-1])),
            "seed": np.int64(-1),
            "replay": np.array(self._replay_name or ""),
            "spawn": np.array(self._spawn),
            "timestamp": np.array(datetime.now(timezone.utc).isoformat()),
        }
        path = self.out_dir / f"episode_{self._episode_index:06d}.npz"
        self._episode_index += 1
        payload = {f"obs_{k}": v for k, v in episode["obs"].items()}
        payload.update({
            "action": episode["action"],
            "reward": episode["reward"],
            "terminated": episode["terminated"],
            "truncated": episode["truncated"],
            "info": np.array(episode["info"], dtype=object),
        })
        # the route geometry + per-step BEV pose / progress (see reset/step)
        payload.update(_route_payload(episode))
        payload.update(meta)
        payload.update(_space_payload(self.observation_space, self.action_space,
                                      self.env_meta))
        np.savez_compressed(path, **payload)
        self._buf = self._new_buffer()

    def close(self):
        self._flush()
        super().close()
