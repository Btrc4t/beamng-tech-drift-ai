"""Online training against the live game (PPO or TD3).

The observation is the env's float channels plus 20 ultrasonic beams derived
from the road BEV (the default - distance to the road edge); ``--no-beams``
observes the raw BEV images instead.

``train.py online``: resumes automatically from the newest checkpoint under
``models/<ALGO>/``, and saves the latest weights on interrupt or when the run
completes normally. ``--algo td3`` switches to off-policy TD3 (replay buffer,
deterministic policy, exploration via ``--action-noise``); ``--init`` starts
from a saved model (e.g. a BC warm start).

``--endpoints host:port,...`` trains against N game hosts in parallel (a
vector env with one ZeroMQ client sub-env per host; each host runs
``host_game.py``, which owns the game and the tcom loop). Eval reuses the
training sub-envs (round-robin over the hosts, with the env's special eval
reset); replays and the debug window are single-game features and are skipped
in that mode.
"""
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnvWrapper
# Importing the package registers the env id used by gym.make below.
import beamng_tech_env  # noqa: F401

from beamng_tech_env.wrappers import BeamsWrapper
from train.common import (ALGOS, DEFAULT_EVAL_FREQ, REPLAY_PREFIX, ResumeFrom,
                          find_latest_model)


class RewardComponentsCallback(BaseCallback):
    """Log the env's per-step reward components to TensorBoard.

    The env reports each reward term (drift/dmg/slow/fast) in ``info`` every
    step; this writes them as ``reward/*`` scalars so you can see *why* the
    total reward moves, not just that it moved.

    With ``record_replays`` it also drives the game's native replay system:
    each training episode is captured to a ``.rpl`` file, and at the end of
    the episode the replay path and the spawn point are written to
    TensorBoard as text, so the dashboard links the two.
    """

    def __init__(self, env, verbose=0, record_replays=False, replay_prefix="drift"):
        super().__init__(verbose)
        self._is_vec = isinstance(env, gym.vector.VectorEnv)
        if self._is_vec:
            self.env = env
        else:
            # gym.make wraps the env in OrderEnforcing/PassiveEnvChecker; the
            # raw env (which stores reward components in _last_info) sits at
            # the bottom of that .env chain, so walk down to it
            while hasattr(env, "env"):
                env = env.env
            self.env = env
        self._tb_writer = None
        # True until the first on_step; flags the first step of an episode
        self._prev_done = True
        # native replay recording (optional): one .rpl per training episode;
        # single-env only, because the per-sub-env beamng clients are not
        # reachable through the vector env
        self._record_replays = record_replays and not self._is_vec
        self._replay_recorder = None
        self._replay_index = 0
        self._current_replay = None
        if self._record_replays:
            from data.replay import BeamNGReplayRecorder
            # beamng only exists after the first reset, so resolve it lazily
            self._replay_recorder = BeamNGReplayRecorder(
                lambda: getattr(self.env, "beamng", None), prefix=replay_prefix)

    def _log_episode_summary(self, replay_path, step) -> None:
        """Write the episode's spawn point (and replay path, if any) as text."""
        writer = self._tb_writer
        if writer is None:
            return
        reset_pos = getattr(self.env, "_reset_pos", None)
        reset_spawn = getattr(self.env, "_reset_spawn", "") or ""
        if reset_pos is not None:
            x, y, z = (float(v) for v in reset_pos)
            spawn_text = f"{reset_spawn} at ({x:.1f}, {y:.1f}, {z:.1f})"
        else:
            spawn_text = reset_spawn or "unknown"
        writer.add_text("episode/spawn", spawn_text, step)
        if replay_path:
            writer.add_text("episode/replay", replay_path, step)

    def _on_step(self) -> bool:
        if self._tb_writer is None:
            self._tb_writer = next(
                (getattr(fmt, "writer", None) for fmt in self.logger.output_formats
                 if getattr(fmt, "writer", None) is not None),
                None,
            )
        if self._tb_writer is None:
            return True
        step = self.model.num_timesteps
        if self._is_vec:
            # one _last_info per sub-env; log the mean of each reward
            # component across the games (BEV images and replays are
            # single-env only)
            infos = self.env.get_attr("_last_info")
            for key in {k for info in infos
                        for k in (info.get("reward_components") or {})}:
                values = [info["reward_components"][key] for info in infos
                          if key in (info.get("reward_components") or {})]
                self._tb_writer.add_scalar(f"reward/{key}",
                                           float(np.mean(values)), step)
            dones = self.locals.get("dones")
            self._prev_done = bool(np.any(dones)) if dones is not None else False
            return True
        last_info = getattr(self.env, "_last_info", {})
        comps = last_info.get("reward_components")
        if comps:
            for key, value in comps.items():
                self._tb_writer.add_scalar(f"reward/{key}", float(value), step)
        dones = self.locals.get("dones")
        current_done = bool(np.any(dones)) if dones is not None else False
        # the env sets failure_reason only on a crash (damage / tire flat), so
        # this captures the bird's-eye view at the moment of impact; when the
        # env is wrapped (e.g. BeamsWrapper), the obs carries no BEV, so fall
        # back to the raw env's last one
        if last_info.get("failure_reason"):
            bev = self.locals.get("new_obs", {}).get("bev")
            if bev is None:
                bev = getattr(self.env, "_last_bev", None)
            if bev is not None:
                arr = np.asarray(bev)
                if arr.ndim == 4:
                    arr = arr[0]
                self._tb_writer.add_image("crash/bev", arr, step)
        # the first step of a new episode (right after a done) logs the reset
        # bird's-eye view, so you can compare the start state against the crash
        if self._prev_done:
            reset_bev = getattr(self.env, "_reset_bev", None)
            if reset_bev is not None:
                self._tb_writer.add_image("episode_start/bev", np.asarray(reset_bev), step)
            # log the numeric spawn channels (rpm/gear/airspeed/...) so the real
            # start observation is inspectable, not just the BEV image
            reset_obs = getattr(self.env, "_reset_obs", None)
            if reset_obs is not None:
                for key, value in reset_obs.items():
                    if key == "bev":
                        continue
                    arr = np.asarray(value).ravel()
                    for j, v in enumerate(arr):
                        name = f"episode_start/{key}" if arr.size == 1 else f"episode_start/{key}[{j}]"
                        self._tb_writer.add_scalar(name, float(v), step)
            # start recording this episode's native replay from the spawn
            if self._replay_recorder is not None and getattr(self.env, "beamng", None) is not None:
                self._current_replay = self._replay_recorder.start(self._replay_index)
                self._replay_index += 1
        # the step that ends an episode: stop its replay and log where it
        # spawned plus the .rpl path, so the dashboard links the two
        if current_done:
            replay_path = None
            if self._replay_recorder is not None and self._current_replay is not None:
                replay_path = self._replay_recorder.stop(self._replay_index - 1) or self._current_replay
                self._current_replay = None
            self._log_episode_summary(replay_path, step)
        self._prev_done = current_done
        return True


class SB3VecEnvAdapter(VecEnvWrapper):
    """Adapt a gymnasium vector env to SB3's VecEnv API.

    SB3 2.9 only recognises its own ``VecEnv`` hierarchy: a gymnasium
    ``AsyncVectorEnv`` is neither a ``VecEnv`` nor a ``gym.Env``, so
    ``_wrap_env`` rejects it. It also expects ``step_wait`` to return a
    4-tuple (``dones = terminated | truncated``) with per-env infos as a
    list of dicts, while gymnasium returns a 5-tuple with a batched info
    dict. Spaces are exposed unbatched (``single_*``), the same convention
    SB3's own vec envs use, so a checkpoint trained with one env loads with
    any number of sub-envs.
    """

    def __init__(self, venv):
        super().__init__(venv, observation_space=venv.single_observation_space,
                         action_space=venv.single_action_space)

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.venv.step_wait()
        dones = terminated | truncated
        # gymnasium batches the per-env info dict; SB3's buffers extend it as
        # a list and index it per env, so unbatch it (list values are already
        # per-env, arrays/scalars are distributed)
        per_env = [{} for _ in range(self.num_envs)]
        for key, value in infos.items():
            if isinstance(value, (list, tuple)) and len(value) == self.num_envs:
                for i, item in enumerate(value):
                    per_env[i][key] = item
            elif (isinstance(value, np.ndarray) and value.ndim
                  and value.shape[0] == self.num_envs):
                for i in range(self.num_envs):
                    per_env[i][key] = value[i]
            else:
                for i in range(self.num_envs):
                    per_env[i][key] = value
        for i in range(self.num_envs):
            per_env[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
            if dones[i]:
                per_env[i]["terminal_observation"] = (
                    {k: v[i] for k, v in obs.items()} if isinstance(obs, dict) else obs[i])
        return obs, rewards, dones, per_env

    def reset(self):
        obs, _infos = self.venv.reset()
        return obs

    def get_attr(self, attr_name, indices=None):
        return self.venv.get_attr(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        self.venv.set_attr(attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, **method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self.venv.env_is_wrapped(wrapper_class)


class SaveOnInterruptCallback(BaseCallback):
    """Persist the latest weights when training is interrupted (CTRL+C)
    or finishes its timesteps.

    The online mode wraps ``learn()`` in ``try/except KeyboardInterrupt`` and
    calls :meth:`save_last`, so progress since the last ``best_model`` save is
    not lost; the same call runs after ``learn()`` returns normally. The model
    reference is set by SB3 during ``init_callback``.
    """

    def __init__(self, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _on_step(self) -> bool:
        return True

    def save_last(self, reason: str = "interrupt") -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        model.save(self.save_path)
        if self.verbose:
            print(f"[{reason}] saved latest model -> {self.save_path}")


class RoundRobinEvalEnv(gym.Env):
    """Eval env that reuses the training sub-envs, round-robining over hosts.

    Each eval episode resets the next sub-env (round-robin) with the env's
    special eval reset and steps it with the deterministic policy. This reuses
    the training connections (no dedicated eval host) and spreads the eval load
    across all hosts. The sub-envs are driven through the vector env's
    :meth:`call`, which routes to the single active sub-env (see
    :class:`EvalRoutingWrapper`).
    """

    metadata = {"render_modes": []}

    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self._index = -1
        self.observation_space = venv.single_observation_space
        self.action_space = venv.single_action_space

    def _next(self):
        self._index = (self._index + 1) % self.num_envs
        return self._index

    def reset(self, seed=None, options=None):
        idx = self._next()
        results = self.venv.call("_eval_reset", idx)
        obs, info = results[idx]
        return obs, info

    def step(self, action):
        idx = self._index
        return self.venv.call("_eval_step", idx, action)[idx]


def _parse_endpoints(raw):
    """Parse a ``host:port,host:port`` CSV into a list of (host, int port)."""
    endpoints = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        host, sep, port = item.rpartition(":")
        if not sep or not host:
            raise ValueError(
                f"bad endpoint {item!r} in --endpoints (expected host:port)")
        try:
            endpoints.append((host, int(port)))
        except ValueError:
            raise ValueError(
                f"bad port in endpoint {item!r} (expected host:port)")
    if not endpoints:
        raise ValueError(f"no endpoints in --endpoints {raw!r}")
    return endpoints


def _action_dim(env):
    """The per-env action dim (a vector env's action_space is batched)."""
    action_space = (env.single_action_space
                    if isinstance(env, gym.vector.VectorEnv)
                    else env.action_space)
    return action_space.shape


def _online_model_kwargs(args, env, log_dir) -> dict:
    """Constructor kwargs for the online model, per algorithm.

    PPO takes an entropy bonus. TD3 is deterministic, so it instead needs an
    exploration noise (its only way to explore) and a train_freq: the SB3
    default of one update per episode is far too infrequent for a ~20 FPS sim.
    """
    kwargs = dict(verbose=1, tensorboard_log=log_dir)
    if args.algo == "ppo":
        kwargs["ent_coef"] = args.ent_coef
    else:
        kwargs["train_freq"] = (args.train_freq, "step")
        if args.action_noise > 0:
            from stable_baselines3.common.noise import NormalActionNoise
            dim = _action_dim(env)
            kwargs["action_noise"] = NormalActionNoise(
                mean=np.zeros(dim), sigma=np.full(dim, args.action_noise))
    return kwargs


def cmd_online(args) -> None:
    MODEL, POLICY = ALGOS[args.algo]
    run_id = datetime.now().strftime("%Y.%m.%d-%H.%M")
    models_dir = Path(f"models/{MODEL.__name__}/{run_id}")
    log_dir = Path(f"logs/{MODEL.__name__}/{run_id}")
    models_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    endpoints = _parse_endpoints(args.endpoints) if args.endpoints else None
    use_beams = not args.no_beams

    def _wrap_sub_env(env):
        if use_beams:
            # the default observation swaps the BEV image for 20 ultrasonic beams
            env = BeamsWrapper(env)
        return env

    if endpoints is not None:
        # one connect-only sub-env per hosted game; per-sub-env wrappers are
        # applied in the factory (they cannot wrap the vector env itself)
        from beamng_tech_env.envs import BeamNGTechHostedVectorEnv
        venv = BeamNGTechHostedVectorEnv(endpoints, sub_env_wrapper=_wrap_sub_env)
        inner_env = venv  # the callback reads _last_info through get_attr
        # SB3 2.9 only accepts its own VecEnv subclasses (see the adapter)
        env = SB3VecEnvAdapter(venv)
        for flag, name in ((args.replays, "--replays"),
                           (args.debug_window, "--debug-window")):
            if flag:
                print(f"{name} is ignored in vector mode")
    else:
        env = gym.make("beamng_tech_env/BeamNGTech-v0.39", render_mode="human")
        env = _wrap_sub_env(env)
        inner_env = env  # gym.make's wrapper; the callback unwraps it to the raw env
        env = Monitor(env, filename=str(log_dir / "monitor.csv"))
        if args.debug_window:
            # outermost: the display thread gets every frame the agent sees; the
            # agent loop only pays for a non-blocking queue put (see debug.py)
            from beamng_tech_env.debug import DebugWindowWrapper
            env = DebugWindowWrapper(env)

    if endpoints is not None:
        # eval reuses the training sub-envs (round-robin over the hosts), so no
        # dedicated eval host is needed; the special eval reset (a fresh random
        # waypoint) is passed explicitly because the stack-based detection
        # cannot reach across the network (see EvalRoutingWrapper)
        eval_env = RoundRobinEvalEnv(venv)
        eval_callback = EvalCallback(eval_env, best_model_save_path=models_dir,
                                     log_path=log_dir, eval_freq=args.eval_freq,
                                     deterministic=True, render=False,
                                     warn=False)
    else:
        # deterministic=True uses the policy's mean action (no sampling noise)
        # for eval; it is independent of the env's own reset-time randomness
        eval_callback = EvalCallback(env, best_model_save_path=models_dir,
                                     log_path=log_dir, eval_freq=args.eval_freq,
                                     deterministic=True, render=True)
    reward_callback = RewardComponentsCallback(
        inner_env, record_replays=bool(args.replays and endpoints is None),
        replay_prefix=REPLAY_PREFIX)

    # resume only from a checkpoint whose observation space matches this run's
    # (a different layout - beams vs the BEV image - would fail SB3's load-time
    # space check); None -> train from scratch
    prev_model = find_latest_model(models_dir.parent, ResumeFrom(args.resume_from),
                                   observation_space=env.observation_space)
    save_callback = SaveOnInterruptCallback(models_dir / "last_model.zip", verbose=1)
    if args.init:
        # explicit starting point (e.g. a BC model); the critic starts fresh
        # but the actor carries over
        print(f"Initialising from {args.init}")
        model = MODEL.load(args.init, env=env, verbose=1)
        model.tensorboard_log = log_dir
        learn_kwargs = dict(reset_num_timesteps=False)
    elif prev_model is not None:
        print(f"Resuming from {prev_model}")
        model = MODEL.load(prev_model, env=env, verbose=1)
        # learn() configures its own tensorboard logger from tensorboard_log;
        # with reset_num_timesteps=False it continues into the newest run dir.
        model.tensorboard_log = log_dir
        if args.algo == "ppo" and args.ent_coef != model.ent_coef:
            model.ent_coef = args.ent_coef
            print(f"ent_coef overridden to {args.ent_coef:g}")
        learn_kwargs = dict(reset_num_timesteps=False)
    else:
        print("No saved checkpoint found; training from scratch (last resort)")
        model = MODEL(POLICY, env, **_online_model_kwargs(args, env, log_dir))
        learn_kwargs = dict()
    if args.init or prev_model is not None:
        # a loaded model inherits its hyperparameters from the zip; the saved
        # action_noise is a VectorizedActionNoise tied to the env count at save
        # time, so it goes stale when the count changes (a 2-endpoint checkpoint
        # resumed with 1 endpoint would add (2, 8) noise to a (1, 8) action and
        # crash the replay buffer's reshape). Rebuild it from the CLI so
        # _setup_learn re-wraps it for the current env count; BC seeds have no
        # exploration noise, so without this a loaded TD3 would run fully
        # deterministic (and can stall in a local optimum)
        if args.algo == "td3":
            from stable_baselines3.common.noise import NormalActionNoise
            dim = _action_dim(env)
            model.action_noise = (
                NormalActionNoise(mean=np.zeros(dim),
                                  sigma=np.full(dim, args.action_noise))
                if args.action_noise > 0 else None)
    try:
        callbacks = [callback for callback in (eval_callback, reward_callback,
                                               save_callback)
                     if callback is not None]
        model.learn(args.timesteps, callback=callbacks, **learn_kwargs)
        save_callback.save_last(reason="done")
    except KeyboardInterrupt:
        save_callback.save_last()
    except Exception:
        # the env gave up (e.g. BeamNG unresponsive N times in a row):
        # persist the weights, then let the traceback end the process
        save_callback.save_last(reason="crash")
        raise
