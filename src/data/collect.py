"""Command-line data collection.

Records trajectories (and, in ``--human`` / ``--ai`` modes, BeamNG's native
``.rpl`` replay) from the drift environment, driven by a policy of your
choice:

- no ``--model``            -> random actions (explore / seed the dataset)
- ``--model path.zip --algo ppo`` -> roll out a trained Stable-Baselines3 model
- ``--human``               -> a person drives (wheel/pedals/keyboard); the env
  sends no control, reads the applied action back from the game, and spawns at
  a random waypoint each episode
- ``--ai``                  -> synthetic mode: the game's built-in AI drives
  the spawned vehicle (its ``ai_mode`` is set on reset); same read-back and
  random-spawn mechanics as ``--human``, but no wheel or keyboard needed
- ``--drive``               -> a person drives, but nothing is collected and no
  replay is recorded (a pure debug/inspection mode; combine with ``--debug``
  for the live window). The car respawns on crash/timeout and keeps going.

Replays are recorded in ``--human`` and ``--ai`` modes only (``--no-replay``
to skip).

Examples::

    # 20 random episodes, no replay
    python -m data.collect --out data/random --episodes 20

    # 10 episodes with the trained PPO
    python -m data.collect --out data/ppo --episodes 10 \
        --model models/PPO/<run>/best_model.zip --algo ppo

    # drive it yourself: random spawn, full episode length, until Ctrl+C,
    # recording a .rpl replay per episode
    python -m data.collect --out data --human

    # let the game's built-in AI drive (synthetic data): 20 episodes, the AI
    # spans the whole road network, recording a .rpl replay per episode
    python -m data.collect --out data/ai --ai --episodes 20

    # ...or collect for an hour of wall-clock time
    python -m data.collect --out data/ai --ai --minutes 60

    # watch a recorded drift in-game (no collection); a full path or a bare
    # filename works too
    python -m data.collect --play replays/drift_2026-10-15_11-13-41_000000.rpl

    # drive it yourself with the live debug window (no collection, no replay)
    python -m data.collect --drive --debug

The local packages must be importable (``pip install -e src/env/gymnasium
-e src/data``) and, for ``--play`` / ``--drive`` / a real run, the game must be
launchable (BNG_HOME set).
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym

from data.collector import TrajectoryCollector
from data.dataset import resolve_dataset
from data.replay import BeamNGReplayRecorder

_ALGOS = {
    "ppo": ("stable_baselines3", "PPO"),
    "a2c": ("stable_baselines3", "A2C"),
    "ddpg": ("stable_baselines3", "DDPG"),
    "td3": ("stable_baselines3", "TD3"),
    "sac": ("stable_baselines3", "SAC"),
    "dqn": ("stable_baselines3", "DQN"),
}


def _load_model(path: str, algo: str):
    import importlib
    module_name, class_name = _ALGOS[algo]
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls.load(path, verbose=0)


def _normalize_replay_path(path: str) -> str:
    """Turn a user-supplied replay path into the game-relative form.

    Accepts a full path, a path already under ``replays/``, or a bare filename,
    and returns the ``replays/....rpl`` form the engine's ``loadFile`` expects.
    """
    p = path.replace("\\", "/")
    if "/replays/" in p:
        p = p.split("/replays/", 1)[1]
    if not p.startswith("replays/"):
        p = "replays/" + p.rsplit("/", 1)[-1]
    return p


def _play_replay(args) -> None:
    """Launch the game and play back a recorded replay, then wait for Ctrl+C."""
    import time
    env = gym.make(args.env, render_mode="human")
    try:
        env.reset()
        beamng = env.unwrapped.beamng
        beamng.control.resume()  # non-human mode pauses the game after reset
        filename = _normalize_replay_path(args.play)
        if not BeamNGReplayRecorder(beamng).play(filename, autoplay=True):
            return
        print(f"[collect] playing {filename!r} - press Ctrl+C to stop", flush=True)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    finally:
        env.close()


def _drive(args) -> None:
    """Launch the game in human mode and let the driver drive, with no data
    collection and no replay recording (a pure debug/inspection mode).

    The env is stepped with a dummy action; the driver's wheel/pedals/keyboard
    drive the car (the env reads the applied action back but we discard it).
    When the car crashes or the episode truncates, the env is reset - human
    mode resumes from the saved end pose, so the driver picks up where they
    left off - and driving continues. Press Ctrl+C to stop.
    """
    import numpy as np
    env = gym.make(args.env, render_mode="human" if args.render else None,
                   human_mode=True)
    if args.debug:
        # the display thread gets every frame the driver sees; the step loop
        # only pays for a non-blocking queue put (see debug.py)
        from beamng_tech_env.debug import DebugWindowWrapper
        env = DebugWindowWrapper(env)
    dummy = np.zeros(env.action_space.shape, dtype=np.float32)
    try:
        env.reset(seed=args.seed)
        print("[drive] driving - press Ctrl+C to stop", flush=True)
        while True:
            _, _, terminated, truncated, _ = env.step(dummy)
            if terminated or truncated:
                env.reset()
    except KeyboardInterrupt:
        print("\n[drive] stopped", flush=True)
    finally:
        env.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env", default="beamng_tech_env/BeamNGTech-v0.39")
    parser.add_argument("--out", default=None,
                        help="dataset output directory, or the data root "
                             "(e.g. data/) to auto-resolve the dataset (prefers "
                             "'human'); required unless --play is used")
    parser.add_argument("--episodes", type=int, default=None,
                        help="number of episodes (default 10; 0 = until Ctrl+C; "
                             "human mode defaults to 0)")
    parser.add_argument("--minutes", type=float, default=None,
                        help="collect for this many minutes of wall-clock time "
                             "instead of a fixed episode count (for --ai or "
                             "--human; with no --episodes the run is "
                             "time-limited, otherwise whichever limit is "
                             "reached first wins)")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--human", action="store_true",
                        help="a person drives (wheel/pedals/keyboard): the env "
                             "sends no control and reads the applied action back "
                             "from the game, spawning at a random waypoint each "
                             "episode")
    parser.add_argument("--ai", action="store_true",
                        help="synthetic mode: the game's built-in AI drives the "
                             "spawned vehicle (its ai_mode is set on reset); "
                             "same read-back and random-spawn mechanics as "
                             "--human, no wheel or keyboard needed")
    parser.add_argument("--ai-mode", default="span",
                        choices=["random", "span", "traffic"],
                        help="the built-in AI's mode for --ai (default span: "
                             "drive the whole road network)")
    parser.add_argument("--ai-speed", type=float, default=None, metavar="MS",
                        help="the built-in AI's speed limit in m/s for --ai "
                             "(default: no limit)")
    parser.add_argument("--ai-aggression", type=float, default=0.3,
                        help="the built-in AI's aggression for --ai, 0.3 "
                             "(normal driving) to 1 (limit of traction); "
                             "default 0.3")
    parser.add_argument("--model", default=None, help="path to an SB3 model .zip")
    parser.add_argument("--algo", choices=sorted(_ALGOS), default="ppo",
                        help="algorithm of --model (default ppo)")
    parser.add_argument("--no-replay", action="store_true",
                        help="disable native .rpl replay recording (replays are "
                             "recorded in --human and --ai modes only)")
    parser.add_argument("--replay-prefix", default="drift")
    parser.add_argument("--play", default=None, metavar="FILE",
                        help="launch the game and play back a recorded .rpl "
                             "replay (game-relative path, e.g. "
                             "replays/drift_....rpl), then wait until you press "
                             "Ctrl+C; no data is collected")
    parser.add_argument("--drive", action="store_true",
                        help="launch the game in human mode and drive it "
                             "(wheel/pedals/keyboard); no data is collected and "
                             "no replay is recorded - a pure debug/inspection "
                             "mode. The car respawns on crash/timeout; press "
                             "Ctrl+C to stop")
    parser.add_argument("--debug", action="store_true",
                        help="show a live cv2 debug window (BEV + beams + road "
                             "centerline + telemetry); used with --drive")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    if args.human and args.model:
        parser.error("--human and --model are mutually exclusive")
    if args.ai and args.model:
        parser.error("--ai and --model are mutually exclusive")
    if args.human and args.ai:
        parser.error("--human and --ai are mutually exclusive")
    if args.drive and (args.play or args.ai or args.model):
        parser.error("--drive is mutually exclusive with --play, --ai and --model")
    if args.minutes is not None and args.minutes <= 0:
        parser.error("--minutes must be positive")
    if not args.play and not args.drive and not args.out:
        parser.error("--out is required (unless --play or --drive is used)")
    max_duration = args.minutes * 60 if args.minutes else None
    if max_duration is not None and args.episodes is None:
        n_episodes = 0  # a time-limited run
    else:
        n_episodes = (args.episodes if args.episodes is not None
                      else (0 if args.human else 10))

    # make the env package importable when run from the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import beamng_tech_env  # noqa: F401  (registers the env id)

    if args.play:
        _play_replay(args)
        return
    if args.drive:
        _drive(args)
        return

    out_dir = resolve_dataset(args.out, create=True)
    env = gym.make(args.env, render_mode="human" if args.render else None,
                   human_mode=args.human,
                   ai_mode=(args.ai_mode if args.ai else False),
                   ai_speed=args.ai_speed,
                   ai_aggression=args.ai_aggression)
    policy = ("ai" if args.ai
              else ("human" if args.human
                    else (_load_model(args.model, args.algo) if args.model else None)))

    replay = None
    # replays are recorded in human and AI mode only (the env's modes are the
    # source of truth); --no-replay opts out
    if (env.unwrapped.human_mode or env.unwrapped.ai_mode) and not args.no_replay:
        # beamng only exists after the first reset, so resolve it lazily
        replay = BeamNGReplayRecorder(lambda: env.unwrapped.beamng,
                                      prefix=args.replay_prefix)
        print(f"[collect] native replay recording enabled (prefix={args.replay_prefix!r})")

    collector = TrajectoryCollector(env, out_dir=out_dir, replay_recorder=replay)
    try:
        stats = collector.collect_episodes(n_episodes=n_episodes, policy=policy,
                                           max_steps=args.max_steps, render=args.render,
                                           max_duration=max_duration)
    finally:
        env.close()

    total = sum(s["reward"] for s in stats)
    n_replays = sum(1 for s in stats if s.get("replay"))
    print(f"\n[collect] {len(stats)} episodes, total reward {total:.2f}, "
          f"{n_replays} replays -> {out_dir}")


if __name__ == "__main__":
    main()
