"""Drive the game with a saved SB3 model and report the reward per episode.

``train.py drive`` loads a saved PPO/TD3 model (.zip) and drives the game
with it, deterministically. The env is wrapped to match the model's
observation: RouteObsWrapper (the route channels next/attack/turns) when the
model consumes them (the default training observation), and BeamsWrapper
when it consumes the 20 beams instead of the BEV image. With ``--episodes N``
it drives N scenarios back to back, each a fresh reset at a different
(reproducible) waypoint.
"""
import json
import zipfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, TD3
# Importing the package registers the env id used by gym.make below.
import beamng_tech_env  # noqa: F401

from beamng_tech_env.wrappers import BeamsWrapper, RouteObsWrapper


def cmd_drive(args) -> None:
    from beamng_tech_env.core.constants import FPS

    model_path = Path(args.model)
    # SB3 stores no algo name in the zip. PPO and TD3 share the same policy
    # classes (module stable_baselines3.common.policies), so the policy
    # module can't tell them apart; use the algorithm-specific
    # hyperparameters instead (PPO: clip_range/n_epochs, TD3: tau/policy_delay)
    data = json.loads(zipfile.ZipFile(model_path).read("data"))
    module = data["policy_class"]["__module__"]
    algo = {"stable_baselines3.ppo.policies": PPO,
            "stable_baselines3.td3.policies": TD3}.get(module)
    if algo is None:
        if "clip_range" in data:
            algo = PPO
        elif "tau" in data or "policy_delay" in data:
            algo = TD3
        else:
            raise SystemExit(
                f"unknown SB3 model {model_path}: policy module {module!r} "
                "and no PPO/TD3 hyperparameters in the saved data")
    model = algo.load(model_path, verbose=0)
    use_beams = "beams" in model.observation_space.spaces
    use_route_obs = "turns" in model.observation_space.spaces

    env = gym.make("beamng_tech_env/BeamNGTech-v0.39", render_mode="human")
    if use_route_obs:
        env = RouteObsWrapper(env)
    if use_beams:
        env = BeamsWrapper(env)
    if args.debug_window:
        # the display thread gets every frame the agent sees; the agent loop
        # only pays for a non-blocking queue put (see debug.py)
        from beamng_tech_env.debug import DebugWindowWrapper
        env = DebugWindowWrapper(env)

    # eval=True spawns at a random waypoint (no checkpoint resume), so with a
    # fixed --seed every model starts from the same spot; each scenario uses a
    # derived seed (base + index) so the episodes differ but stay reproducible
    rewards = []
    try:
        for ep in range(args.episodes):
            seed = None if args.seed is None else args.seed + ep
            obs, info = env.reset(seed=seed, options={"eval": True})
            total, t = 0.0, 0
            while t < args.steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(
                    np.asarray(action, dtype=np.float32))
                total += float(r)
                t += 1
                if t % 200 == 0:
                    print(f"[drive] ep {ep + 1}/{args.episodes} "
                          f"step {t}/{args.steps}  reward {total:.1f}")
                if term or trunc:
                    break
            rewards.append(total)
            print(f"[drive] ep {ep + 1}/{args.episodes}  {t} steps "
                  f"({t / FPS:.1f} sim s)  reward {total:.2f}")
    except KeyboardInterrupt:
        print("[drive] interrupted")
    finally:
        env.close()
    if rewards:
        print(f"[drive] {len(rewards)}/{args.episodes} episodes  "
              f"mean reward {sum(rewards) / len(rewards):.2f}  "
              f"total {sum(rewards):.2f}")
