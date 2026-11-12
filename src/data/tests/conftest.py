"""Pytest config for the data package tests (no game required)."""
import os
import sys

import pytest

# Make the ``data`` package (and ``src``) importable regardless of cwd.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402
from beamng_tech_env.core.constants import ALLOWED_GEARS  # noqa: E402


class TinyDriftEnv(gym.Env):
    """A tiny stand-in for the BeamNG env: Dict obs (small BEV + vectors) and a
    3-D continuous action. Episodes are short and deterministic-ish so the
    tests run fast without the game."""

    def __init__(self, horizon: int = 8):
        bev = (1, 36, 36)
        self.observation_space = spaces.Dict({
            "rpm": spaces.Box(0.0, 1.0, (1,), np.float32),
            "imu_acc": spaces.Box(-1.0, 1.0, (3,), np.float32),
            "drift_angle": spaces.Box(-1.0, 1.0, (1,), np.float32),
            "airspeed": spaces.Box(0.0, 1.0, (1,), np.float32),
            "gear": spaces.Box(0.0, 1.0, (1,), np.float32),
            "next": spaces.Box(-1.0, 1.0, (3,), np.float32),
            "attack": spaces.Box(-1.0, 1.0, (1,), np.float32),
            "turns": spaces.Box(-1.0, 1.0, (3,), np.float32),
            "bev": spaces.Box(0, 255, bev, np.uint8),
        })
        # match the real env's action layout: [steering, throttle/brake,
        # parkingbrake] in [-1, 1] plus one [0, 1] value per allowed gear
        n_gears = len(ALLOWED_GEARS)
        self.action_space = spaces.Box(
            low=np.concatenate([np.full(3, -1.0), np.zeros(n_gears)]),
            high=np.concatenate([np.full(3, 1.0), np.ones(n_gears)]),
            dtype=np.float32)
        self.horizon = horizon
        self._t = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def _obs(self):
        rng = self.np_random
        # deterministic gear: neutral for the first 3 steps, then 1st (the
        # shift lands on step 3), encoded like the real env: (gear + 1) / 7
        gear = 0 if self._t < 3 else 1
        return {
            "rpm": rng.random((1,)).astype(np.float32),
            "imu_acc": (rng.random((3,)) - 0.5).astype(np.float32),
            "drift_angle": (rng.random((1,)) - 0.5).astype(np.float32),
            "airspeed": rng.random((1,)).astype(np.float32),
            "gear": np.array([(gear + 1) / 7.0], dtype=np.float32),
            # the route channels: a time-varying "next" (the gate gets
            # closer), the rest neutral
            "next": np.array([0.0, min(1.0, self._t * 0.25), 0.0], np.float32),
            "attack": np.zeros(1, np.float32),
            "turns": np.zeros(3, np.float32),
            "bev": rng.integers(0, 256, (1, 36, 36), dtype=np.uint8),
        }

    def step(self, action):
        self._t += 1
        reward = float(np.sum(np.asarray(action) ** 2))
        terminated = self._t >= self.horizon
        return self._obs(), reward, terminated, False, {"step": self._t}


@pytest.fixture
def tiny_env():
    return TinyDriftEnv()
