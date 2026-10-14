# https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/callbacks.html
# https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
# install env before with ../venv/bin/pip install -e src/env/gymnasium
from datetime import datetime
from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO as MODEL
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.ppo import MultiInputPolicy as POLICY
# Importing the package registers the env id used by gym.make below.
import beamng_tech_env  # noqa: F401


class ResumeFrom(str, Enum):
    """Which saved checkpoint to resume training from."""
    LAST = "last_model.zip"   # most recent state (saved on interrupt)
    BEST = "best_model.zip"   # highest eval reward


TIMESTEPS=250000
# Which checkpoint to resume from: the most recent state (LAST) or the
# highest-eval-reward model (BEST).
RESUME_FROM = ResumeFrom.LAST

def find_latest_model(models_root: Path, resume_from: ResumeFrom) -> Path | None:
    """Newest run dir (timestamps sort chronologically) holding a checkpoint.

    Resolution order: (1) ``resume_from``, (2) the other checkpoint type
    (LAST<->BEST), (3) ``None`` -> the caller trains from scratch. Training
    from scratch is only the last resort, reached when neither checkpoint
    exists in any run.
    """
    fallback = ResumeFrom.BEST if resume_from is ResumeFrom.LAST else ResumeFrom.LAST
    for choice in (resume_from, fallback):
        for run_dir in sorted(models_root.glob("*/"), reverse=True):
            candidate = run_dir / choice.value
            if candidate.is_file():
                if choice is not resume_from:
                    print(f"no {resume_from.value} found; falling back to {choice.value}")
                return candidate
    return None


class RewardComponentsCallback(BaseCallback):
    """Log the env's per-step reward components to TensorBoard.

    The env reports each reward term (drift/dmg/slow/fast) in ``info`` every
    step; this writes them as ``reward/*`` scalars so you can see *why* the
    total reward moves, not just that it moved.
    """

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        # gym.make wraps the env in OrderEnforcing/PassiveEnvChecker; the raw
        # env (which stores reward components in _last_info) sits at the bottom
        # of that .env chain, so walk down to it
        while hasattr(env, "env"):
            env = env.env
        self.env = env
        self._tb_writer = None
        # True until the first on_step; flags the first step of an episode
        self._prev_done = True

    def _on_step(self) -> bool:
        if self._tb_writer is None:
            self._tb_writer = next(
                (getattr(fmt, "writer", None) for fmt in self.logger.output_formats
                 if getattr(fmt, "writer", None) is not None),
                None,
            )
        if self._tb_writer is None:
            return True
        last_info = getattr(self.env, "_last_info", {})
        step = self.model.num_timesteps
        comps = last_info.get("reward_components")
        if comps:
            for key, value in comps.items():
                self._tb_writer.add_scalar(f"reward/{key}", float(value), step)
        dones = self.locals.get("dones")
        current_done = bool(np.any(dones)) if dones is not None else False
        # the env sets failure_reason only on a crash (damage / tire flat), so
        # this captures the bird's-eye view at the moment of impact
        if last_info.get("failure_reason"):
            bev = self.locals.get("new_obs", {}).get("bev")
            if bev is not None:
                self._tb_writer.add_image("crash/bev", np.asarray(bev)[0][None, ...], step)
        # the first step of a new episode (right after a done) logs the reset
        # bird's-eye view, so you can compare the start state against the crash
        if self._prev_done:
            reset_bev = getattr(self.env, "_reset_bev", None)
            if reset_bev is not None:
                self._tb_writer.add_image("episode_start/bev", np.asarray(reset_bev)[None, ...], step)
        self._prev_done = current_done
        return True


class SaveOnInterruptCallback(BaseCallback):
    """Persist the latest weights when training is interrupted (CTRL+C).

    train.py wraps ``learn()`` in ``try/except KeyboardInterrupt`` and calls
    :meth:`save_last`, so progress since the last ``best_model`` save is not
    lost. The model reference is set by SB3 during ``init_callback``.
    """

    def __init__(self, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def save_last(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        model.save(self.save_path)
        if self.verbose:
            print(f"[interrupt] saved latest model -> {self.save_path}")


run_id = datetime.now().strftime("%Y.%m.%d-%H.%M")
models_dir = Path(f"models/{MODEL.__name__}/{run_id}")
log_dir = Path(f"logs/{MODEL.__name__}/{run_id}")
models_dir.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

env = gym.make("beamng_tech_env/BeamNGTech-v0.39", render_mode="human")
inner_env = env  # gym.make's wrapper; the callback unwraps it to the raw env
env = Monitor(env, filename=str(log_dir / "monitor.csv"))

# deterministic=True uses the policy's mean action (no sampling noise) for eval;
# it is independent of the env's own reset-time randomness (random car/waypoint)
eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                             log_path=log_dir, eval_freq=1000,
                             deterministic=True, render=True)
reward_callback = RewardComponentsCallback(inner_env)

prev_model = find_latest_model(models_dir.parent, RESUME_FROM)
save_callback = SaveOnInterruptCallback(models_dir / "last_model.zip", verbose=1)
if prev_model is not None:
    print(f"Resuming from {prev_model}")
    model = MODEL.load(prev_model, env=env, verbose=1)
    # learn() configures its own tensorboard logger from tensorboard_log;
    # with reset_num_timesteps=False it continues into the newest PPO_* dir.
    model.tensorboard_log = log_dir
    try:
        model.learn(TIMESTEPS, reset_num_timesteps=False,
                    callback=[eval_callback, reward_callback, save_callback])
    except KeyboardInterrupt:
        save_callback.save_last()
else:
    print("No saved checkpoint found; training from scratch (last resort)")
    model = MODEL(POLICY, env, verbose=1, tensorboard_log=log_dir)
    try:
        model.learn(TIMESTEPS, callback=[eval_callback, reward_callback, save_callback])
    except KeyboardInterrupt:
        save_callback.save_last()
