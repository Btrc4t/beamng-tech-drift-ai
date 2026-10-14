"""Gymnasium API tests for BeamNGTechEnv.

The env launches BeamNG.drive itself (``beamng.open()`` in ``_first_time_setup``),
so no manual game start is needed. These tests share one game instance for the
whole session (see the ``env`` fixture in ``conftest.py``) and each call
``env.reset()`` for a clean state.

The ``render`` test is interactive (it runs until Ctrl+C) and is skipped by
default; run it on its own with ``pytest -m interactive``.
"""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import beamng_tech_env  # noqa: F401  (registers the env id)

pytestmark = pytest.mark.game


def validate_obs(env, obs):
    assert isinstance(obs, dict), f"observation must be a dict, got {type(obs)}"
    for key, space in env.observation_space.items():
        assert key in obs, f"missing observation key: {key}"
        val = obs[key]
        assert val.shape == space.shape, f"{key}: shape {val.shape} != expected {space.shape}"
        assert val.dtype == space.dtype, f"{key}: dtype {val.dtype} != expected {space.dtype}"
        assert space.contains(val), f"{key}: value outside declared space"
    extra = set(obs) - set(env.observation_space)
    assert not extra, f"unexpected observation keys: {extra}"


def test_conformance(env):
    """Official Gymnasium API conformance check."""
    check_env(env, skip_render_check=True)


def test_smoke(env, n_steps=30):
    """Reset + N steps, validating every observation against its space."""
    obs, info = env.reset(seed=0)
    validate_obs(env, obs)

    total_reward = 0.0
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        validate_obs(env, obs)
        assert isinstance(reward, (float, np.floating)), f"reward not float: {type(reward)}"
        assert isinstance(terminated, (bool, np.bool_)), f"terminated not bool: {type(terminated)}"
        assert isinstance(truncated, (bool, np.bool_)), f"truncated not bool: {type(truncated)}"
        total_reward += float(reward)
        if terminated or truncated:
            break
    # reaching this point without raising is the pass condition


def test_determinism(env, n_steps=10):
    """Same seed + same action sequence should reproduce observations.

    With the deterministic physics flag set this should be (near) exact; any
    residual difference is a timing/floating-point artifact of the game.
    """
    def run():
        env.reset(seed=123)
        obs_list = []
        for _ in range(n_steps):
            # fixed, reproducible action (gearbox is in arcade/auto mode)
            action = np.array([0.0, 0.5, 0.0], np.float32)
            obs, *_ = env.step(action)
            obs_list.append({k: v.copy() for k, v in obs.items()})
        return obs_list

    a = run()
    b = run()
    max_diff = 0.0
    for obs_a, obs_b in zip(a, b):
        for k in obs_a:
            if k == "bev":
                # The LiDAR BEV is read from a streaming shared-memory buffer,
                # so it is sensitive to frame timing; the kinematic channels
                # are the deterministic ones.
                continue
            max_diff = max(max_diff, float(np.max(np.abs(obs_a[k] - obs_b[k]))))
    assert max_diff < 0.05, (
        f"observations differ between identical runs: max |diff| = {max_diff:.6f}"
    )


def test_diagnose(env):
    """Every observation key must be present and inside its declared space."""
    obs, info = env.reset(seed=0)
    validate_obs(env, obs)


@pytest.mark.interactive
def test_render(env):
    """Drive with random actions until Ctrl+C (interactive, skipped by default)."""
    env.reset(seed=0)
    try:
        while True:
            env.step(env.action_space.sample())
    except KeyboardInterrupt:
        pass
