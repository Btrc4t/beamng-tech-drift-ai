"""Load recorded trajectories back into memory for offline training.

A dataset directory is what :class:`data.collector.TrajectoryCollector`
produces::

    data_dir/
      episode_000000.npz
      episode_000001.npz
      ...

A dataset may also be *split* (the Hugging Face Hub split layout, created by
``data.curate split``): the episodes live in ``train/`` and ``validation/``
subdirectories, which the Hub auto-detects as the dataset's splits::

    data_dir/
      README.md
      train/
        episode_000000.npz
        ...
      validation/
        episode_000010.npz
        ...

Every episode is self-contained: the observation/action spaces and env
metadata are embedded in each file (pickled bytes under the ``space_obs``,
``space_act``, and ``env_meta`` keys).

:class:`TrajectoryDataset` is a :class:`torch.utils.data.Dataset` over the
individual transitions. It loads every split by default; pass ``split``
(e.g. ``"train"``) to load one split only. Each item is a dict::

    {"obs": {...}, "action": ..., "reward": ...,
     "next_obs": {...}, "terminated": bool, "truncated": bool, "info": {...}}

``next_obs`` is the observation of the following step within the same
episode; on the final step of an episode it is set to the step's own
observation (the bootstrap is masked by ``terminated``/``truncated`` anyway).

:func:`to_replay_buffer` turns the dataset into a Stable-Baselines3 replay
buffer (``DictReplayBuffer`` for Dict observations) so the offline training
loop can sample batches exactly like an on-policy/off-policy algorithm does.
"""
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from data.collector import _arrays_to_obs

# split directory names; the Hugging Face Hub auto-detects a dataset's splits
# from directories named like this (see the Hub's "data files configuration"
# doc), so the split layout is train/ + validation/ subdirectories
SPLIT_NAMES = ("train", "validation", "test")


def split_subdirs(data_dir):
    """The split subdirectories of a dataset (train/, validation/, ...), in
    canonical split order; empty for a flat dataset."""
    data_dir = Path(data_dir)
    return [data_dir / name for name in SPLIT_NAMES
            if list((data_dir / name).glob("episode_*.npz"))]


def episode_paths(data_dir):
    """All episode files of a dataset: the top-level ones plus those in its
    split subdirectories, sorted by name."""
    data_dir = Path(data_dir)
    paths = list(data_dir.glob("episode_*.npz"))
    for sub in split_subdirs(data_dir):
        paths.extend(sub.glob("episode_*.npz"))
    return sorted(paths, key=lambda p: p.name)


def episode_path(data_dir, index):
    """The path of one episode (top level or a split subdirectory), or None
    when the dataset has no such episode."""
    name = f"episode_{index:06d}.npz"
    for d in (Path(data_dir), *split_subdirs(data_dir)):
        path = d / name
        if path.is_file():
            return path
    return None


def _split_paths(data_dir, split):
    """The episode paths of one dataset directory under a split selection:
    the split's subdirectory for a split dataset (empty when it lacks the
    split), all top-level episodes for a flat dataset (which has no splits).
    """
    data_dir = Path(data_dir)
    if split is None:
        return episode_paths(data_dir)
    if split_subdirs(data_dir):
        return sorted((data_dir / split).glob("episode_*.npz"))
    return sorted(data_dir.glob("episode_*.npz"))


class TrajectoryDataset(torch.utils.data.Dataset):
    """A torch dataset over recorded trajectory transitions.

    ``data_dir`` is a dataset directory (flat, or split into train/ and
    validation/ subdirectories) or a list of them. ``split`` selects one
    split (e.g. ``"train"``); flat datasets have no splits and pass through
    in full.
    """

    def __init__(self, data_dir, split=None):
        # A single directory, or several (e.g. a dataset plus its synthetic/
        # launch folder); episodes are loaded from all of them and the
        # embedded spaces are read from the first file found.
        if isinstance(data_dir, (list, tuple)):
            self.data_dirs = [Path(d) for d in data_dir]
        else:
            self.data_dirs = [Path(data_dir)]
        self.data_dir = self.data_dirs[0]
        self.split = split
        paths = [p for d in self.data_dirs for p in _split_paths(d, split)]
        if not paths:
            raise FileNotFoundError(
                f"no episode_*.npz in {self.data_dirs}"
                + (f" (split {split!r})" if split else "")
                + "; is this a collected dataset? (run TrajectoryCollector "
                  "first)"
            )
        # the spaces are embedded in every episode (self-contained files)
        with np.load(paths[0], allow_pickle=True) as data:
            if "space_obs" not in data.files:
                raise ValueError(
                    f"episodes in {self.data_dir} carry no embedded spaces; "
                    f"re-collect the dataset"
                )
            self.observation_space: spaces.Space = pickle.loads(
                data["space_obs"].tobytes())
            self.action_space: spaces.Space = pickle.loads(
                data["space_act"].tobytes())
            self.env_meta: dict = pickle.loads(data["env_meta"].tobytes())

        self.episodes: list[dict] = [self._load_episode(path) for path in paths]

        # flat (episode_idx, step_idx) index over every transition
        self._flat: list[tuple[int, int]] = [
            (e, s) for e, ep in enumerate(self.episodes)
            for s in range(len(ep["reward"]))
        ]
        if not self._flat:
            raise ValueError(f"no transitions found in {self.data_dir}")

    @staticmethod
    def _load_episode(path: Path) -> dict:
        with np.load(path, allow_pickle=True) as data:
            obs = {k[len("obs_"):]: data[k] for k in data.files if k.startswith("obs_")}
            info = [dict(x) for x in data["info"]] if "info" in data.files else [{} for _ in data["reward"]]
            return {
                "obs": obs,
                "action": data["action"],
                "reward": data["reward"],
                "terminated": data["terminated"],
                "truncated": data["truncated"],
                "info": info,
            }

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> dict:
        e, s = self._flat[idx]
        ep = self.episodes[e]
        obs = _arrays_to_obs({k: ep["obs"][k][s] for k in ep["obs"]})
        if s + 1 < len(ep["reward"]):
            next_obs = _arrays_to_obs({k: ep["obs"][k][s + 1] for k in ep["obs"]})
        else:
            next_obs = obs
        return {
            "obs": obs,
            "action": ep["action"][s],
            "reward": float(ep["reward"][s]),
            "next_obs": next_obs,
            "terminated": bool(ep["terminated"][s]),
            "truncated": bool(ep["truncated"][s]),
            "info": ep["info"][s],
        }

    def summary(self) -> str:
        n_trans = len(self)
        n_ep = len(self.episodes)
        rewards = [float(np.sum(ep["reward"])) for ep in self.episodes]
        return (f"{n_ep} episodes, {n_trans} transitions; "
                f"episode reward mean={np.mean(rewards):.2f} "
                f"min={np.min(rewards):.2f} max={np.max(rewards):.2f}")


def resolve_dataset(data_arg, create=False) -> Path:
    """Resolve a data path to a concrete dataset folder.

    Accepts a dataset folder (holding ``episode_*.npz`` directly, or in
    ``train/``/``validation/`` split subfolders) or the data root (holding
    dataset subfolders); in the latter case the dataset name is filled in,
    preferring ``human``, else the only dataset present. A path that does not
    exist yet is a dataset folder to create (``create=True``); an existing
    root without datasets resolves to ``<root>/human`` when ``create=True``
    and raises otherwise."""
    data = Path(data_arg)
    if episode_paths(data):
        return data
    if not data.is_dir():
        if not create:
            raise FileNotFoundError(f"no such file or directory: {data}")
        data.mkdir(parents=True, exist_ok=True)
        return data
    human = data / "human"
    if episode_paths(human):
        return human
    datasets = [d for d in sorted(data.iterdir())
                if d.is_dir() and episode_paths(d)]
    if len(datasets) == 1:
        return datasets[0]
    if not datasets:
        if create:
            human.mkdir(parents=True, exist_ok=True)
            return human
        raise FileNotFoundError(f"no dataset (episode_*.npz) found in {data}")
    raise FileNotFoundError(
        f"multiple datasets in {data}: {', '.join(d.name for d in datasets)}; "
        f"pass one explicitly (e.g. --data {data / datasets[0].name})")


def load_dataset(data_dir, split=None) -> TrajectoryDataset:
    """Convenience constructor for :class:`TrajectoryDataset`."""
    return TrajectoryDataset(data_dir, split=split)


def to_replay_buffer(dataset: TrajectoryDataset,
                     buffer_size: Optional[int] = None,
                     device: str = "cpu"):
    """Build a Stable-Baselines3 replay buffer from a dataset.

    Uses ``DictReplayBuffer`` for Dict observation spaces, ``ReplayBuffer``
    otherwise. ``done`` is the ``terminated`` flag and each info carries
    ``TimeLimit.truncated`` so SB3 bootstraps across truncations (treating
    them as infinite-horizon) exactly as it does during online collection.
    """
    from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer

    buffer_cls = (DictReplayBuffer
                  if isinstance(dataset.observation_space, spaces.Dict)
                  else ReplayBuffer)
    n = len(dataset)
    buffer = buffer_cls(buffer_size or n, dataset.observation_space,
                        dataset.action_space, device=device)
    for i in range(n):
        item = dataset[i]
        buffer.add(
            obs=item["obs"],
            next_obs=item["next_obs"],
            action=np.asarray(item["action"]),
            reward=np.asarray(item["reward"], dtype=np.float32),
            done=np.asarray([item["terminated"]], dtype=np.float32),
            infos=[{"TimeLimit.truncated": item["truncated"]}],
        )
    # mark the buffer full so sample() draws from the whole dataset
    buffer.full = True
    return buffer
