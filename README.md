# BeamNG.tech Drift AI

BeamNG.tech machine learning project that focuses on teaching drifting.
The project provides a Gymnasium environment ([`beamng_tech_env/BeamNGTech-v0.39`](https://github.com/Btrc4t/beamng-tech-drift-env),
checked out as the `src/env/gymnasium` submodule), a recorded human
playthrough dataset ([`data/human`](https://huggingface.co/datasets/daringeyes/beamng-tech-drift), a git submodule), and the tools around them: record driving data, curate
it, train policies online (PPO, TD3) or offline (behavior cloning), and drive the trained models live.

## Published models

Trained model weights are published under `published/` (for example
`published/models/TD3/2026.10.27-11.12/best_model.zip`), so a fresh clone
can drive a trained policy without training one first. To see it in
action, the `drive` subcommand of the train module loads a saved model
and drives the game with it (run from the repo root after loading
`env-drift-tools`; see [Setup](#setup)):

```sh
"$VENV_PATH/bin/python" src/train/train.py drive \
  --model published/models/TD3/2026.10.27-11.12/best_model.zip --seed 42
```

## Documentation

The notebooks in `docs/` are meant to be a beginner friendly introduction into machine learning.
They are structured to explain concepts simply, in relation to the environment built, as it evolved.

The relevant code for each notebook represents the commit where it was introduced.

- `01_drift_setup.ipynb` - BeamNG drifting Deep RL setup
- `02_environment_setup.ipynb` - creating the training environment
- `03_vector_env.ipynb` - parallelizing the training environment
- `04_learning.ipynb` - the first PPO training iteration
- `05_realistic_env.ipynb` - summary of env improvements, behavior cloning and TD3 adoption

## Setup

The repo has two git submodules: the gymnasium environment package
([`src/env/gymnasium`](https://github.com/Btrc4t/beamng-tech-drift-env))
and the recorded human dataset
([`data/human`](https://huggingface.co/datasets/daringeyes/beamng-tech-drift)).
Clone with `--recurse-submodules` so both are checked out. The dataset
submodule stores its episodes with git-lfs, so a fresh clone leaves
pointer files behind: run `git lfs pull` inside it to download the real
data (needs the `git-lfs` tool). The `published/` artifacts are stored
with git-lfs as well; with the tool installed, a clone fetches them
automatically.

You need a Python virtualenv, the local packages installed editable, and a
copy of BeamNG.tech on disk.

1. Clone the repo with its submodules (or update an existing clone), then
   fetch the dataset's git-lfs files:

   ```sh
   git clone --recurse-submodules <repo-url>
   # or, in an existing clone:
   git submodule update --init
   git -C data/human lfs pull
   ```

2. Create the virtualenv (Python 3.14) and install the dependencies:

   ```sh
   python3.14 -m venv venv
   venv/bin/pip install -r requirements.txt
   ```

3. Install the local packages (the environment and the data tools):

   ```sh
   venv/bin/pip install -e src/env/gymnasium -e src/data
   ```

4. Create the `env-drift-tools` file from the example then place your
   machine's paths in the `env-drift-tools` file (the virtualenv and
   the BeamNG.tech install folder), then load it before running commands:

   ```sh
   cp env-drift-tools.example env-drift-tools
   source env-drift-tools
   ```

Because `data/human` is a submodule, writing episodes into it (for
example, a regeneration with `--out data/human`) shows up as changes
inside the submodule. Collect new datasets into their own folder under
`data/` instead.

## GPU processing (cupy)

The env's observation and point-cloud processing runs on the
GPU via cupy when it is installed. The `env-drift-tools` file sets `USE_CUPY=false` to force the CPU backend
(cupy is not even imported, and the GPU transfers become no-ops), equivalent to:

```sh
USE_CUPY=0 "$VENV_PATH/bin/python" src/train/train.py online
```

`USE_CUPY` is disabled in the env file on the assumption that you don't have multiple nvidia GPUs (having just one burdens the same GPU the BeamNG.tech renderer uses), so if you do have multiple GPUs, opt-in by changing it.

`CUPY_DEVICE` selects the CUDA device by numeric id (default: the last one; id's are the ones shown by `nvidia-smi`)

## Commands

Run all commands from the repo root (`drift-ai/`) after loading `env-drift-tools`.
`$VENV_PATH` and `$BNG_HOME` come from that file.

### Tests

The test suite lives in `src/env/gymnasium/tests`. The pure-math and data
tests run without the game; the env and benchmark tests launch BeamNG.drive
once per session (about 48 s) and need `BNG_HOME` plus a display.

```sh
# fast tests only (no game)
"$VENV_PATH/bin/python" -m pytest -m "not game" src/env/gymnasium/tests

# everything, including the game tests
"$VENV_PATH/bin/python" -m pytest src/env/gymnasium/tests
```

### Data collection

`data.collect` records trajectories (one self-contained `.npz` per episode;
the observation/action spaces are embedded in each file) and, in `--human`
and `--ai` mode, the game's native `.rpl` replays. `--ai` collects from the
game's built-in AI driving the spawned vehicle (synthetic data, no wheel or
keyboard needed; `--ai-mode`, `--ai-speed` and `--ai-aggression` tune it).
`--drive` launches the game in human mode to drive it yourself, but collects
nothing and records no replay (a pure debug/inspection mode; `--debug` adds
the live window): the car respawns on crash/timeout and keeps going until
Ctrl+C.

```sh
# drive it yourself: random spawn each episode, until Ctrl+C
"$VENV_PATH/bin/python" -m data.collect --out data --human

# let the game's built-in AI drive and collect (synthetic data)
"$VENV_PATH/bin/python" -m data.collect --out data/ai --ai --episodes 20

# 20 random episodes (no policy)
"$VENV_PATH/bin/python" -m data.collect --out data/random --episodes 20

# roll out a trained PPO model (replays are only recorded in --human/--ai mode)
"$VENV_PATH/bin/python" -m data.collect --out data/ppo --episodes 10 \
  --model models/PPO/<run>/best_model.zip --algo ppo

# watch a recorded drift in-game (no collection)
"$VENV_PATH/bin/python" -m data.collect --play <file.rpl>

# drive it yourself with no collection (debug/inspection; --debug adds the live window)
"$VENV_PATH/bin/python" -m data.collect --drive
```

### Data curation

`data.curate` cleans a collected dataset: cut or delete damaged episodes,
drop short ones, pull in replays, split into train/validation, and build
synthetic launch data from TensorBoard spawn observations.

```sh
# list episode lengths, damage, spawn, and file paths
"$VENV_PATH/bin/python" -m data.curate list --data data

# preview, then cut each episode at its first damage step
# (--margin also cuts that many steps before the damage)
"$VENV_PATH/bin/python" -m data.curate truncate-damage \
  --data data --dry-run
"$VENV_PATH/bin/python" -m data.curate truncate-damage \
  --data data --margin 5

# cut the last N steps off the short episodes (under 1190 steps,
# usually damage-ended); --margin is required
"$VENV_PATH/bin/python" -m data.curate truncate-short \
  --data data --margin 20

# delete episodes shorter than 10 s
"$VENV_PATH/bin/python" -m data.curate delete-short \
  --data data --min-seconds 10

# copy this dataset's .rpl replays out of the game folder
"$VENV_PATH/bin/python" -m data.curate collect-replays --data data

# split into train/ and validation/ (the Hugging Face Hub layout; the
# assignment is a deterministic hash of the episode name + seed, so
# re-running is a no-op and new episodes don't reshuffle the splits)
"$VENV_PATH/bin/python" -m data.curate split \
  --data data/human --val-ratio 0.1 --seed 42

# build synthetic launch data from the newest TensorBoard run: each spawn
# observation (episode_start/bev image + episode_start/* scalars) is paired
# with a launch action that engages a gear (80% 1st, 20% reverse) + throttle.
# Output goes to <data>/synthetic, which offline training picks up
# automatically; --data may be the data root (data/) to auto-resolve the
# dataset (prefers 'human')
"$VENV_PATH/bin/python" -m data.curate synthetic --data data
```

### Data regeneration

`data.regenerate` rebuilds a dataset's episodes from the game's `.rpl`
replays. Use it to refresh a curated dataset after the env, the reward, or
the observations change: the driving comes from the recorded `.rpl`, so the
episodes stay faithful to what actually happened, while the derived
channels (the road BEV, the reward, the beams) are recomputed with the
current code.

By default it runs offline (no game). Each episode is rebuilt from its
`.rpl` (the recorded pose and electrics are the ground truth for the
driving): the pose/electrics-derived channels (drift, airspeed, gear, rpm,
IMU, action), the road BEV (rasterized from the cached road network), and
the reward (the current formula) are recomputed, while the channels the
`.rpl` doesn't carry (the force feedback, the per-step damage) are reused
from the original episode. The `bev` channel is the corrected LiDAR: the
original episode's LiDAR BEV, corrected against the road network (a bare
`.rpl` has no LiDAR returns, so there the road BEV is the LiDAR: road
surface = road level, off-road = obstacle), and the beams are computed on
it.

For the road network data, the game is used once to place the road network
into the cache when it is missing.
The `--lidar` option plays each `.rpl` in the game and reads the LiDAR BEV +
FFB live - it needs the game.

`--data` is the dataset directory: each episode's `.npz` records the `.rpl`
that captured it, and the replays are resolved from those, in episode order
(episodes without a usable `.rpl` are skipped). A folder of `.rpl` files
(or a single `.rpl`) is also accepted. Regenerated episodes keep the source
dataset's numbering (a skipped episode leaves a gap, not a shift).

```sh
# offline (the default): no game, the road network comes from the cache
"$VENV_PATH/bin/python" -m data.regenerate \
  --data data/human-original --out data/human

# live: play each .rpl in the game and read the LiDAR BEV (FFB not possible)
"$VENV_PATH/bin/python" -m data.regenerate \
  --data data/human-original --out data/human --lidar

# a single .rpl, only the first N episodes, or one episode by its index
"$VENV_PATH/bin/python" -m data.regenerate --data <one.rpl> --out data/regen
"$VENV_PATH/bin/python" -m data.regenerate \
  --data data/human-original --out data/human --limit 10
"$VENV_PATH/bin/python" -m data.regenerate \
  --data data/human-original --out data/human --episode 17
```

The road network is cached at `<data root>/roads/<level>.npz` (the default
`--roads`, where `<data root>` is the parent of `--data`); delete the file
to force a fresh fetch from the game. `--max-steps` caps the live-mode
episode length.

### Training data videos

`data.curate video` renders an episode as an `.mp4` so you can watch what
the driver (or policy) did without the game: the corrected LiDAR BEV on the
left (road surface at the road level, everything else as the LiDAR saw it),
with the 20 ultrasonic beams overlaid (distance to the first raised
return), a readout of step, gear, drift angle, speed, force feedback, and
reward, plus drift angle / airspeed / force feedback telemetry panels on the
right. Frames render in parallel.

```sh
# render episode 6 (writes episode_000006.mp4 next to the data)
"$VENV_PATH/bin/python" -m data.curate video --data data --episode 6

# render every episode in the dataset - takes a long time!
"$VENV_PATH/bin/python" -m data.curate video --data data

# control the output: path, frame rate, BEV width, parallel workers
"$VENV_PATH/bin/python" -m data.curate video --data data \
  --episode 6 --out episode6.mp4 --fps 20 --width 384 --workers 4
```

`data.curate play` does the same in the actual game: it launches the game
(if it isn't running), loads the level, and plays the episode's `.rpl`
replay - no menu navigation or replay searches. The replay drives the car in
real time; when it ends the command exits and closes the game.

```sh
# launch the game and play episode 6's replay
"$VENV_PATH/bin/python" -m data.curate play --data data --episode 6

# or play a .rpl file directly
"$VENV_PATH/bin/python" -m data.curate play --replay <path/to/file.rpl>
```

### Training

`train.py` is the single entry point for all training work, with one
subcommand per mode. `online` runs PPO or TD3 against the live game; it
resumes automatically from the newest checkpoint under `models/PPO/` or
`models/TD3/`, and saves on interrupt. `offline` trains on a recorded
dataset with no game: behavior cloning for PPO/TD3. `drive` loads a saved
model (SB3 `.zip`) and drives the game with it. `bootstrap` makes a BC model
ready to continue online. The implementation is split into submodules in
`src/train/` (`commands/` - one module per subcommand, `common.py` - shared
helpers); the 20-beam observation lives in the env package
(`beamng_tech_env/wrappers.py`).

```sh
# online PPO (the default)
"$VENV_PATH/bin/python" src/train/train.py online
"$VENV_PATH/bin/python" src/train/train.py online --timesteps 500000 --resume-from best
# online TD3
"$VENV_PATH/bin/python" src/train/train.py online --algo td3
# evals run every 5 min of sim by default; each eval resets the shared game,
# so a smaller --eval-freq also caps training episode length
"$VENV_PATH/bin/python" src/train/train.py online --eval-freq 1000
```

`offline` logs `nll`, `entropy`, and `rmse` (mean-action error in normalized
action units) for the cloning models, saves on interrupt, and resumes with
`--init`. `--data` accepts the data root (`data`), which resolves to the
[`data/human`](https://huggingface.co/datasets/daringeyes/beamng-tech-drift) submodule, or a dataset folder. A `<data>/synthetic` folder,
if present, is included automatically (disable with `--no-synthetic`).

```sh
# behavior cloning, PPO or TD3 (no game needed)
"$VENV_PATH/bin/python" src/train/train.py offline --data data
"$VENV_PATH/bin/python" src/train/train.py offline --algo td3 --data data
# resume an interrupted run (pass the remaining steps)
"$VENV_PATH/bin/python" src/train/train.py offline --data data \
  --init models/bc/human_steps37000.zip --steps 163000
```

`drive` wraps the env the way the model needs it (the 20 beams instead of
the BEV) and, with a fixed `--seed`, starts every run from the same waypoint
so results are comparable. The game closes when the episode ends.

```sh
"$VENV_PATH/bin/python" src/train/train.py drive \
  --model models/bc/td3/human_steps100000.zip --seed 42
```

#### Parallelized training

A ZeroMQ game hosting option was created to help speed up training. 

It uses the request - response pattern where a request is a command, such as `step`, `reset`, and the expected response is made up of [obs, reward, terminated, truncated, info] for `step` or made up of [obs, info] for `reset`.

When training, it's possible to use multiple endpoints by passing a csv of `host:port` entries as the `--endpoints` argument (ex: `--endpoints 127.0.0.1:25252,192.168.1.2:25252`):

```bash
# host the game in a terminal
#on the local machine
./host_game.py 127.0.0.1 25252
#and on a machine you expect to access remotely (bind a visible IP).
./host_game.py 192.168.1.2 25252
# train using both game instances
"$VENV_PATH/bin/python" src/train/train.py online --algo td3 --action-noise 0.05 --timesteps 600000 --endpoints localhost:25252,192.168.1.2:25252
```

If you don't use `--endpoints`, one env instance is started automatically.

### Benchmark

Measures headline FPS, a phase breakdown of `step()`, physics
determinism, LiDAR point counts, and drift telemetry parity against the
game's Lua side. The benchmark scripts live in the env's test folder, next
to the pytest suite that asserts the same numbers (`test_benchmark.py`):

```sh
"$VENV_PATH/bin/python" src/env/gymnasium/tests/benchmark_single.py
```

`benchmark_remote.py` (same folder) measures the ZeroMQ hosting path
(`host_game.py`) against the direct env:

```sh
"$VENV_PATH/bin/python" src/env/gymnasium/tests/benchmark_remote.py
```

### TensorBoard

Training and eval metrics land in `logs/`, so to see them in the tensorboard web interface use:

```sh
"$VENV_PATH/bin/tensorboard" --logdir logs
```

