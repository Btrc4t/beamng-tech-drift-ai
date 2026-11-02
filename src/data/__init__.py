"""Trajectory (transition) collection for offline training.

This package records (observation, action, reward, done) transitions from a
gymnasium environment so they can be replayed later without the game:

- :mod:`collector` records episodes and saves them to disk (one ``.npz`` per
  episode). It works with any policy: a Stable-Baselines3 model, a plain
  callable, or random actions.
- :mod:`dataset` loads those recordings back into memory as a
  :class:`torch.utils.data.Dataset` and can build a Stable-Baselines3
  ``ReplayBuffer`` for offline RL.

See ``src/train/train.py`` (the ``offline`` mode) for the offline training
loop that consumes the collected data.
"""
from data.collector import TrajectoryCollector, TrajectoryRecorder
from data.dataset import TrajectoryDataset, load_dataset, to_replay_buffer
from data.replay import BeamNGReplayRecorder

__all__ = [
    "TrajectoryCollector",
    "TrajectoryRecorder",
    "TrajectoryDataset",
    "load_dataset",
    "to_replay_buffer",
    "BeamNGReplayRecorder",
]
