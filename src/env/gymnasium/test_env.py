"""Tests for BeamNGTechEnv.

The env launches BeamNG.drive itself (beamng.open() in _first_time_setup),
so no manual game start is needed. Run:

    python test_env.py

Each test is independent; run them all or pick one:

    python test_env.py conformance
    python test_env.py smoke
    python test_env.py determinism
    python test_env.py render
    python test_env.py all
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from beamng_tech_env.envs.beamng_tech import BeamNGTechEnv

def make_env(render_mode="human"):
    # gym.make sets env.spec, so the registered nondeterministic=True is
    # respected by check_env's determinism check (early return).
    # The env starts BeamNG.drive on the first reset() (see _ensure_started),
    # so no manual bootstrap is needed.
    env = gym.make("beamng_tech_env/BeamNGTech-v0.39", render_mode=render_mode)
    return env

def validate_obs(env, obs):
    assert isinstance(obs, dict), f"observation must be a dict, got {type(obs)}"
    for key, space in env.observation_space.items():
        assert key in obs, f"missing observation key: {key}"
        val = obs[key]
        assert val.shape == space.shape, f"{key}: shape {val.shape} != expected {space.shape}"
        assert val.dtype == space.dtype, f"{key}: dtype {val.dtype} != expected {space.dtype}"
        assert space.contains(val), f"{key}: value outside declared space"

def test_conformance():
    """Official Gymnasium API conformance check."""
    env = make_env()
    check_env(env, skip_render_check=True)
    env.close()
    print("[conformance] check_env passed")

def test_smoke(n_steps=30):
    """Reset + N steps, validating every observation against its space."""
    env = make_env()
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
            print(f"[smoke] episode ended early at step {i} (term={terminated}, trunc={truncated})")
            break
    env.close()
    print(f"[smoke] OK: {i + 1} steps, total_reward={total_reward:.3f}")

def test_determinism(n_steps=10):
    """Same seed + same action sequence should reproduce observations.

    Reported as a soft check: with deterministic physics it should be exact,
    but small timing/fp differences in the game can show up, so we print the
    max diff instead of hard-failing.
    """
    env = make_env()

    def run():
        env.reset(seed=123)
        obs_list = []
        for _ in range(n_steps):
            # fixed, reproducible action (gearbox is in arcade/auto mode)
            action = {
                "controls": np.array([0.0, 0.5, 0.0], np.float32),
            }
            obs, *_ = env.step(action)
            obs_list.append({k: v.copy() for k, v in obs.items()})
        return obs_list

    a = run()
    b = run()
    max_diff = 0.0
    for obs_a, obs_b in zip(a, b):
        for k in obs_a:
            max_diff = max(max_diff, float(np.max(np.abs(obs_a[k] - obs_b[k]))))
    env.close()
    print(f"[determinism] max |obs diff| across {len(a)} steps = {max_diff:.6f}")
    if max_diff > 1e-3:
        print("[determinism] WARNING: observations differ between identical runs")

def test_render():
    """Open the window and let the user watch the car. Ctrl+C to stop."""
    env = make_env(render_mode="human")
    env.reset(seed=0)
    print("[render] window open; pressing Ctrl+C closes the env")
    try:
        while True:
            action = env.action_space.sample()
            env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

def test_diagnose():
    """Print, per observation key, exactly why it is/isn't in its space."""
    env = make_env()
    obs, info = env.reset(seed=0)
    print(f"[diagnose] obs type={type(obs).__name__}, info type={type(info).__name__}")
    for key, space in env.observation_space.items():
        if key not in obs:
            print(f"  {key:12s} MISSING")
            continue
        val = obs[key]
        ok = space.contains(val)
        nan = bool(np.isnan(val).any()) if val.dtype.kind == "f" else False
        lo, hi = float(space.low.min()), float(space.high.max())
        print(
            f"  {key:12s} contains={str(ok):5s} shape={str(val.shape):8s}(want {space.shape}) "
            f"dtype={str(val.dtype):8s}(want {space.dtype}) nan={nan} "
            f"min={float(np.min(val)):8.4f} max={float(np.max(val)):8.4f} bounds=[{lo:.2f},{hi:.2f}]"
        )
    extra = set(obs) - set(env.observation_space)
    if extra:
        print(f"  EXTRA keys: {extra}")
    env.close()

TESTS = {
    "conformance": test_conformance,
    "smoke": test_smoke,
    "determinism": test_determinism,
    "diagnose": test_diagnose,
    "render": test_render,
}

if __name__ == "__main__":
    import subprocess

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        # Each test launches and closes its own BeamNG instance, so run them
        # in separate processes: a close() in one test would otherwise race
        # the open() of the next on the game socket. render is interactive,
        # so it is excluded from 'all'.
        for name in TESTS:
            if name == "render":
                continue
            print(f"\n=== {name} ===")
            rc = subprocess.call([sys.executable, os.path.abspath(__file__), name])
            if rc != 0:
                sys.exit(rc)
    elif which in TESTS:
        TESTS[which]()
    else:
        sys.exit(f"unknown test '{which}'. Choose from: {', '.join(TESTS)} | all")
