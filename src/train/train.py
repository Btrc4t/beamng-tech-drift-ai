# https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/callbacks.html
# https://pythonprogramming.net/saving-and-loading-reinforcement-learning-stable-baselines-3-tutorial/
# install env before with ../venv/bin/pip install -e src/env/gymnasium
"""Train the drift policy.

The single entry point for all training work; one subcommand per mode:

``online`` - PPO (default) or TD3 against the live game (needs BNG_HOME and a
display). Resumes automatically from the newest checkpoint under
``models/<ALGO>/``, and saves the latest weights on interrupt or when the run
completes normally. ``--algo td3`` switches to off-policy TD3 (replay buffer,
deterministic policy, exploration via ``--action-noise``); ``--init`` starts
from a saved model (e.g. a BC warm start). Replay recording is off by default
(replays are for human mode); ``--replays`` records the game's native ``.rpl``
replay per episode.

    python train.py online
    python train.py online --timesteps 500000 --resume-from best --replays
    python train.py online --algo td3 --init models/bc/td3/....zip

``offline`` - behavior cloning on a recorded dataset (no game required): PPO
imitates the recorded actions by maximising their log-likelihood; TD3 is
deterministic, so it regresses its action output onto them (MSE) instead.
Needs no reward signal and is the safe baseline: it can only match the
quality of the data, never exceed it. Interrupting (CTRL+C) saves the current
weights; resume with ``--init <saved zip>`` (the step counter restarts, so
pass the remaining steps in ``--steps``).

The observation is the env's float channels plus the LiDAR BEV image and the
route channels (next/attack/turns - the delta to the next route waypoint,
the attack angle, and the upcoming route turns; the default). ``--beams``
observes the 20 derived ultrasonic beams instead of the BEV image, and
``--no-route-obs`` (online) drops the route channels. ``--ai-data`` also
trains on the synthetic AI driving data (``data/ai``, collected with
``data.collect --ai``). A split dataset (one split into ``train/`` and
``validation/`` by ``data.curate split``) trains on its ``train`` split by
default; ``--split`` selects another one.

    python train.py offline --data data
    python train.py offline --data data --ai-data
    python train.py offline --algo td3 --data data
    python train.py offline --data data --init models/.../best_model.zip

``bootstrap`` - make an offline (BC) model ready to continue online. Loads the
BC model and writes it as ``last_model.zip`` into a fresh online run dir under
``models/<ALGO>/``, so a subsequent ``online --algo <algo>`` auto-resumes from
the cloned weights instead of training from scratch. This is the no-``--init``
way to do the BC -> online warm start.

    python train.py bootstrap --algo td3
    python train.py online --algo td3

``drive`` - drive the game with a saved model (SB3 .zip) and report the reward
for each episode. The env is wrapped to match the model's observation (the
route channels and/or the 20 beams instead of the BEV). ``--episodes N``
drives N scenarios back to back, each a fresh reset at a different
(reproducible) waypoint; with a fixed ``--seed`` every model starts from the
same spot.

    python train.py drive --model models/bc/td3/<...>.zip --episodes 5 --seed 44

The implementation is split into submodules - commands/ (one module per
subcommand: online, offline, drive, bootstrap) and common.py (shared
helpers). The 20-beam observation lives in the env package
(beamng_tech_env/wrappers.py). This file only parses the CLI and dispatches.
"""
import argparse
import sys
from pathlib import Path

import torch as th

# Make the sibling ``data`` package (and this package, for the submodules)
# importable no matter where this is run from.
_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from train.common import DEFAULT_EVAL_FREQ, DEFAULT_STEPS
from train.commands.bootstrap import cmd_bootstrap
from train.commands.drive import cmd_drive
from train.commands.offline import cmd_offline
from train.commands.online import cmd_online


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the drift policy (one entry point; see the subcommands)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("online", help="PPO or TD3 against the live game")
    p.add_argument("--algo", choices=["ppo", "td3"], default="ppo",
                   help="algorithm (default: ppo). TD3 is off-policy: it learns "
                        "from a replay buffer and can warm-start from a BC model "
                        "via --init")
    p.add_argument("--timesteps", type=int, default=250_000)
    p.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ,
                   help="steps between deterministic evals. The eval resets "
                        "the shared game env, which also caps training "
                        "episodes at this length (default: 5 min of sim)")
    p.add_argument("--resume-from", choices=["last", "best"], default="last",
                   help="checkpoint to resume from (last_model.zip or best_model.zip)")
    p.add_argument("--replays", action="store_true",
                   help="record the game's native .rpl replay per episode (off "
                        "by default; replays are for human mode)")
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="PPO-only entropy bonus coefficient; a small positive "
                        "value keeps exploration up so the policy can escape the "
                        "spawn stall (default: 0.0, the SB3 default)")
    p.add_argument("--init", type=str, default=None,
                   help="optional path to an SB3 model to start from (e.g. a "
                        "BC model); takes priority over --resume-from")
    p.add_argument("--train-freq", type=int, default=4,
                   help="TD3-only: one gradient step every N env steps "
                        "(default: 4). SB3's default of one update per episode "
                        "is far too infrequent for a ~20 FPS sim")
    p.add_argument("--action-noise", type=float, default=0.1,
                   help="TD3-only: std of the exploration noise. TD3 is "
                        "deterministic, so this is its only way to explore "
                        "(default: 0.1; 0 = no exploration)")
    p.add_argument("--beams", action="store_true",
                   help="observe the 20 derived ultrasonic beams instead of "
                        "the LiDAR BEV image (the BEV is the default)")
    p.add_argument("--no-route-obs", action="store_true",
                   help="don't add the route channels (next/attack/turns) to "
                        "the observation (they are on by default)")
    p.add_argument("--debug-window", action="store_true",
                   help="show a live cv2 debug window (BEV + beams + "
                        "telemetry) on a background thread, so the agent "
                        "loop is not slowed (needs a display)")
    p.add_argument("--endpoints", type=str, default=None,
                   help="comma-separated host:port list of game hosts to "
                        "train against in parallel (e.g. "
                        "1.2.3.4:5555,6.7.8.9:5556); the port is each host's "
                        "ZeroMQ port (its tcom port is port+1, internal); when "
                        "set, training uses a vector env with one ZeroMQ "
                        "client sub-env per host instead of a single local "
                        "game (launch each with host_game.py)")
    p.set_defaults(func=cmd_online)

    p = sub.add_parser("offline",
                       help="behavior cloning on a recorded dataset (no game)")
    p.add_argument("--algo", choices=["ppo", "td3"], default="ppo",
                   help="ppo: behavior cloning via the Gaussian log-likelihood; "
                        "td3: deterministic, so it regresses its action output "
                        "(MSE) instead (default: ppo)")
    p.add_argument("--data", required=True,
                   help="dataset directory, or the data root (e.g. data/) "
                        "to auto-resolve the dataset (prefers 'human')")
    p.add_argument("--split", default=None,
                   help="dataset split to train on (e.g. train, validation); "
                        "default: the 'train' split when the dataset has "
                        "splits, else all episodes")
    p.add_argument("--no-synthetic", action="store_true",
                   help="don't include the <data>/synthetic launch folder "
                        "(created by `data.curate synthetic`)")
    p.add_argument("--ai-data", action="store_true",
                   help="also train on the synthetic AI driving data "
                        "(data/ai, collected with `data.collect --ai`)")
    p.add_argument("--steps", type=int, default=None,
                   help=f"gradient steps (default: per-algo, {DEFAULT_STEPS})")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="PPO-only entropy bonus coefficient (unused for TD3)")
    p.add_argument("--init", type=str, default=None,
                   help="optional path to an SB3 model to initialise from")
    p.add_argument("--out", type=str, default=None, help="where to save the model")
    p.add_argument("--device", type=str, default="cuda" if th.cuda.is_available() else "cpu",
                   help="compute device (default: cuda if available, else cpu)")
    p.add_argument("--log-every", type=int, default=1000,
                   help="steps between progress prints")
    p.add_argument("--beams", action="store_true",
                   help="observe the 20 derived ultrasonic beams instead of "
                        "the LiDAR BEV image (the BEV is the default)")
    p.set_defaults(func=cmd_offline)

    p = sub.add_parser("drive", help="drive the game with a saved model")
    p.add_argument("--model", required=True,
                   help="SB3 model file (.zip, PPO or TD3)")
    p.add_argument("--steps", type=int, default=1200,
                   help="max steps per episode (default: 1200 = 60 sim s)")
    p.add_argument("--episodes", type=int, default=1,
                   help="scenarios (resets) to drive back to back; each uses a "
                        "derived seed (--seed + index) so they differ but stay "
                        "reproducible (default: 1)")
    p.add_argument("--seed", type=int, default=None,
                   help="reset seed for a reproducible spawn; with "
                        "--episodes > 1 each scenario uses seed + index "
                        "(default: random)")
    p.add_argument("--device", type=str,
                   default="cuda" if th.cuda.is_available() else "cpu")
    p.add_argument("--debug-window", action="store_true",
                   help="show a live cv2 debug window (BEV + beams + "
                        "telemetry) on a background thread, so the agent "
                        "loop is not slowed (needs a display)")
    p.set_defaults(func=cmd_drive)

    p = sub.add_parser("bootstrap",
                       help="make an offline (BC) model ready to continue online")
    p.add_argument("--algo", choices=["ppo", "td3"], required=True,
                   help="algorithm of the BC model to bootstrap")
    p.add_argument("--model", type=str, default=None,
                   help="BC model to bootstrap (default: the newest under "
                        "models/bc/<algo>/)")
    p.set_defaults(func=cmd_bootstrap)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
