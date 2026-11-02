"""Shared helpers for the train.py subcommands.

The single training entry point (train.py) parses the CLI and dispatches to
one module per subcommand (online.py, offline.py, drive.py, bootstrap.py);
the pieces they share live here.
"""
import base64
import json
import pickle
import zipfile
from enum import Enum
from pathlib import Path

from stable_baselines3 import PPO, TD3
from stable_baselines3.ppo import MultiInputPolicy as PpoPolicy
from stable_baselines3.td3 import MultiInputPolicy as Td3Policy

from data.dataset import load_dataset, resolve_dataset, split_subdirs

# algorithm name -> (SB3 model class, MultiInput policy class). Both policies
# share the (policy, env, ...) constructor, so the online and offline paths are
# written once and driven by this table.
ALGOS = {
    "ppo": (PPO, PpoPolicy),
    "td3": (TD3, Td3Policy),
}
# per-algo default gradient steps for `offline`
DEFAULT_STEPS = {"ppo": 100_000, "td3": 100_000}


class ResumeFrom(str, Enum):
    """Which saved checkpoint to resume training from."""
    LAST = "last"   # last_model.zip: most recent state (saved on interrupt)
    BEST = "best"   # best_model.zip: highest eval reward

    @property
    def filename(self) -> str:
        return f"{self.value}_model.zip"


# Prefix for the game's native .rpl replays recorded per training episode.
REPLAY_PREFIX = "drift"
# steps between deterministic evals: 5 min of sim at the 20 FPS tick rate
DEFAULT_EVAL_FREQ = 5 * 60 * 20


def _obs_keys(space):
    """The key set of a gymnasium Dict observation space (None if not a Dict).

    The keys uniquely identify the observation layout (e.g. ``beams`` vs the
    ``bev`` image), so two spaces are resume-compatible iff their key sets
    match."""
    if space is None:
        return None
    if hasattr(space, "spaces"):
        return set(space.spaces.keys())
    return None


def _model_obs_keys(path) -> set | None:
    """The observation-space keys of a saved SB3 model (from the zip's data),
    or None if it cannot be read (the caller then skips the compatibility
    check and lets SB3's own load-time check decide)."""
    try:
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("data"))
        obs = data.get("observation_space", {})
        serialized = obs.get(":serialized:")
        if not serialized:
            return None
        space = pickle.loads(base64.b64decode(serialized))
        return _obs_keys(space)
    except Exception:
        return None


def find_latest_model(models_root: Path, resume_from: ResumeFrom,
                      observation_space=None) -> Path | None:
    """Newest run dir (timestamps sort chronologically) holding a checkpoint.

    Resolution order: (1) ``resume_from``, (2) the other checkpoint type
    (LAST<->BEST), (3) ``None`` -> the caller trains from scratch. Training
    from scratch is only the last resort, reached when neither checkpoint
    exists in any run.

    When ``observation_space`` is given, checkpoints whose observation space
    keys do not match it are skipped (a model saved with a different
    observation layout - e.g. beams vs the BEV image - cannot be loaded into
    this env)."""
    fallback = ResumeFrom.BEST if resume_from is ResumeFrom.LAST else ResumeFrom.LAST
    env_keys = _obs_keys(observation_space)
    for choice in (resume_from, fallback):
        for run_dir in sorted(models_root.glob("*/"), reverse=True):
            candidate = run_dir / choice.filename
            if not candidate.is_file():
                continue
            if env_keys is not None:
                model_keys = _model_obs_keys(candidate)
                if model_keys is not None and model_keys != env_keys:
                    continue   # incompatible observation space; try the next
            if choice is not resume_from:
                print(f"no {resume_from.value} found; falling back to {choice.value}")
            return candidate
    return None


def _latest_bc_model(models_root: Path, algo: str) -> Path | None:
    """Most recently saved offline (BC) model for an algorithm.

    BC saves to ``models/bc/<algo>/<dataset>_steps<N>.zip``; the newest by
    modification time is the most recently trained one."""
    bc_dir = models_root / "bc" / algo
    if not bc_dir.is_dir():
        return None
    models = list(bc_dir.glob("*.zip"))
    if not models:
        return None
    return max(models, key=lambda p: p.stat().st_mtime)


def load_training_dataset(args):
    """The ``--data`` argument resolved to a dataset, plus the synthetic
    launch folder when present (and not disabled), plus the synthetic AI
    driving data when --ai-data. A split dataset trains on its ``train``
    split by default (``args.split`` overrides). Returns (dataset, data_dir)."""
    data_dir = resolve_dataset(args.data)
    data_dirs = [data_dir]
    # the synthetic launch folder lives at the data root, a sibling of the
    # datasets (e.g. data/synthetic, not data/human/synthetic)
    synthetic = data_dir.parent / "synthetic"
    if synthetic.is_dir() and not args.no_synthetic:
        data_dirs.append(synthetic)
        print(f"including synthetic launch data: {synthetic}")
    # the synthetic AI driving data (data/ai), opt-in via --ai-data
    if args.ai_data:
        ai = data_dir.parent / "ai"
        if ai not in data_dirs:
            if ai.is_dir():
                data_dirs.append(ai)
                print(f"including synthetic AI data: {ai}")
            else:
                print(f"warning: --ai-data is set but {ai} does not exist")
    # a split dataset (train/ + validation/ subfolders) uses its train split
    # by default, so training never sees the held-out episodes
    split = getattr(args, "split", None)
    if split is None and split_subdirs(data_dir):
        split = "train"
        print(f"using the train split of {data_dir} (pass --split to change)")
    dataset = load_dataset(data_dirs, split=split)
    print(dataset.summary())
    return dataset, data_dir
