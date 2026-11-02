"""Fast, game-free tests for the data collection + offline training pipeline."""
import gymnasium as gym
import numpy as np
import pytest

from data.collector import TrajectoryCollector
from data.dataset import load_dataset, resolve_dataset, to_replay_buffer
from data.replay import BeamNGReplayRecorder
from beamng_tech_env.core.constants import gear_to_action


class HumanTinyEnv(gym.Wrapper):
    """Simulates the env's human mode: the action passed in is ignored and each
    step returns the 'applied action' the game would have read back from the
    electrics sensor (a deterministic function of the step count)."""

    def __init__(self, env):
        super().__init__(env)
        self._t = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._t = 0
        return obs, info

    def step(self, action):
        self._t += 1
        obs, reward, terminated, truncated, info = super().step(action)
        gear = 0 if self._t < 3 else 1  # matches the tiny env's obs gear
        info["applied_action"] = np.concatenate([
            np.array([
                min(1.0, self._t * 0.25),  # steering ramps up to full lock
                0.5,                       # throttle held
                0.0,                       # no parking brake
            ], dtype=np.float32),
            gear_to_action(gear),          # gear (the real env's action encoding)
        ])
        return obs, reward, terminated, truncated, info


class StubBeamng:
    """Records the lua commands it is given; the record call succeeds."""

    def __init__(self):
        self.commands = []

    def queue_lua_command(self, chunk, response=False):
        self.commands.append(chunk)
        return "ok"


class NoStreamBeamng(StubBeamng):
    """The engine has no file stream, so the record call reports no_stream."""

    def queue_lua_command(self, chunk, response=False):
        self.commands.append(chunk)
        return "no_stream" if "s:record(" in chunk else "ok"


class SpawnTinyEnv(gym.Wrapper):
    """Adds a spawn waypoint name to the reset info (like the BeamNG env)."""

    def __init__(self, env, spawn="wp_test"):
        super().__init__(env)
        self._spawn = spawn

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info = dict(info)
        info["spawn"] = self._spawn
        return obs, info


def test_collect_and_reload_roundtrip(tiny_env, tmp_path):
    collector = TrajectoryCollector(tiny_env, out_dir=tmp_path / "ds")
    stats = collector.collect_episodes(n_episodes=3, policy=None, max_steps=8, verbose=False)

    # files written: one npz per episode, no separate meta file (the spaces
    # are embedded in each episode)
    ds = tmp_path / "ds"
    assert not (ds / "meta.pkl").exists()
    assert len(list(ds.glob("episode_*.npz"))) == 3
    assert [s["terminated"] for s in stats] == [True, True, True]

    # reload and check the flat transition count (3 episodes x 8 steps)
    dataset = load_dataset(ds)
    assert len(dataset) == 3 * 8
    # the embedded spaces round-trip exactly
    assert dataset.observation_space == tiny_env.observation_space
    assert dataset.action_space == tiny_env.action_space

    item = dataset[0]
    assert set(item.keys()) == {"obs", "action", "reward", "next_obs",
                                "terminated", "truncated", "info"}
    assert set(item["obs"].keys()) == set(tiny_env.observation_space.spaces.keys())
    assert item["obs"]["bev"].shape == (1, 36, 36)
    assert item["obs"]["bev"].dtype == np.uint8
    assert item["action"].shape == tiny_env.action_space.shape

    # last step of an episode: terminated True, next_obs == obs
    last = dataset[7]
    assert last["terminated"] is True
    np.testing.assert_array_equal(last["next_obs"]["bev"], last["obs"]["bev"])

    # replay buffer builds and samples with matching shapes
    buffer = to_replay_buffer(dataset)
    assert buffer.size() == len(dataset)
    batch = buffer.sample(16)
    assert batch.actions.shape == (16, *tiny_env.action_space.shape)
    assert batch.observations["bev"].shape == (16, 1, 36, 36)
    assert batch.dones.shape == (16, 1)


def test_collect_with_replay_recorder(tiny_env, tmp_path):
    stub = StubBeamng()
    recorder = BeamNGReplayRecorder(stub, prefix="test")
    collector = TrajectoryCollector(tiny_env, out_dir=tmp_path / "ds",
                                    replay_recorder=recorder)
    stats = collector.collect_episodes(n_episodes=2, policy=None, max_steps=8, verbose=False)

    # each episode produced a replay filename recorded in the stats
    assert all(s["replay"] for s in stats)
    assert stats[0]["replay"].startswith("replays/test_")
    assert stats[0]["replay"].endswith(".rpl")
    # start + defensive-stop + stop per episode => at least 3 lua calls each
    assert len(stub.commands) >= 2 * 3
    # the record command embeds the episode filename
    assert any(stats[0]["replay"] in c for c in stub.commands)

    # the replay name is persisted in the episode npz
    dataset = load_dataset(tmp_path / "ds")
    with np.load(tmp_path / "ds" / "episode_000000.npz", allow_pickle=True) as data:
        assert str(data["replay"]) == stats[0]["replay"]


def test_replay_recorder_no_stream(tiny_env, tmp_path):
    recorder = BeamNGReplayRecorder(NoStreamBeamng(), prefix="x")
    assert recorder.start(0) is None
    assert recorder.is_recording is False
    assert recorder.stop(0) is None


def _make_dataset(root, name):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "episode_000000.npz").write_bytes(b"")
    return d


def test_resolve_dataset(tmp_path):
    human = _make_dataset(tmp_path, "human")
    random_ds = _make_dataset(tmp_path, "random")

    # a dataset folder is used as-is
    assert resolve_dataset(human) == human
    assert resolve_dataset(random_ds) == random_ds
    # the data root fills in the dataset name, preferring human
    assert resolve_dataset(tmp_path) == human

    # a root holding a single non-human dataset resolves to it
    only = tmp_path / "only"
    _make_dataset(only, "random")
    assert resolve_dataset(only) == only / "random"

    # several datasets without human -> ambiguous, error
    multi = tmp_path / "multi"
    _make_dataset(multi, "a")
    _make_dataset(multi, "b")
    with pytest.raises(FileNotFoundError):
        resolve_dataset(multi)

    # a missing path: error without create, created as a dataset with it
    with pytest.raises(FileNotFoundError):
        resolve_dataset(tmp_path / "nope")
    assert resolve_dataset(tmp_path / "nope", create=True) == tmp_path / "nope"

    # an existing root without datasets: <root>/human with create, error without
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_dataset(empty)
    assert resolve_dataset(empty, create=True) == empty / "human"
    assert (empty / "human").is_dir()


def _split_dataset(tiny_env, tmp_path, n_episodes=8, seed=7):
    """A collected dataset split with curate (val-ratio 0.5); returns the
    (data_dir, n_train, n_val) so the tests don't hard-code the hash."""
    from data.curate import main as curate_main

    ds = tmp_path / "ds"
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=n_episodes, policy=None, max_steps=8, verbose=False)
    curate_main(["split", "--data", str(ds), "--val-ratio", "0.5",
                 "--seed", str(seed)])
    n_train = len(list((ds / "train").glob("episode_*.npz")))
    n_val = len(list((ds / "validation").glob("episode_*.npz")))
    return ds, n_train, n_val


def test_curate_split(tiny_env, tmp_path, capsys):
    """split moves the episodes into train/ and validation/ (the Hub split
    layout), writes the dataset card, and is deterministic and idempotent."""
    from data.curate import _split_fraction, main as curate_main

    ds, n_train, n_val = _split_dataset(tiny_env, tmp_path)
    out = capsys.readouterr().out
    assert f"{n_train} train, {n_val} validation" in out
    assert 0 < n_train < 8 and 0 < n_val < 8  # both splits are non-empty

    # the assignment is the deterministic per-episode hash
    val = {f"episode_{i:06d}.npz" for i in range(8)
           if _split_fraction(7, f"episode_{i:06d}.npz") < 0.5}
    train = {f"episode_{i:06d}.npz" for i in range(8)} - val
    assert {p.name for p in (ds / "train").glob("episode_*.npz")} == train
    assert {p.name for p in (ds / "validation").glob("episode_*.npz")} == val
    assert not list(ds.glob("episode_*.npz"))  # nothing left at the top level

    # the dataset card carries the Hub's manual split configuration
    readme = (ds / "README.md").read_text()
    assert "configs:" in readme
    assert "split: train" in readme
    assert "split: validation" in readme
    assert 'path: "train/*.npz"' in readme
    assert 'path: "validation/*.npz"' in readme

    # idempotent: a second run moves nothing
    curate_main(["split", "--data", str(ds), "--val-ratio", "0.5", "--seed", "7"])
    out = capsys.readouterr().out
    assert "moved 0 episode(s)" in out

    # a new episode joins without reshuffling the existing ones
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=1, policy=None, max_steps=8, verbose=False)
    new = "episode_000008.npz"
    if _split_fraction(7, new) < 0.5:
        val.add(new)
    else:
        train.add(new)
    curate_main(["split", "--data", str(ds), "--val-ratio", "0.5", "--seed", "7"])
    assert {p.name for p in (ds / "train").glob("episode_*.npz")} == train
    assert {p.name for p in (ds / "validation").glob("episode_*.npz")} == val


def test_load_dataset_split(tiny_env, tmp_path):
    """A split dataset loads all episodes by default and one split on
    request; a flat dataset passes through in full regardless of split."""
    ds, n_train, n_val = _split_dataset(tiny_env, tmp_path)

    all_ds = load_dataset(ds)
    train_ds = load_dataset(ds, split="train")
    val_ds = load_dataset(ds, split="validation")
    assert all_ds.split is None
    assert train_ds.split == "train"
    assert len(all_ds) == 8 * 8
    assert len(train_ds) == n_train * 8
    assert len(val_ds) == n_val * 8

    # a split the dataset lacks yields no episodes
    with pytest.raises(FileNotFoundError):
        load_dataset(ds, split="test")

    # a single split folder is a flat dataset: it loads in full, and a split
    # request on a flat dataset passes through in full
    train_only = load_dataset(ds / "train")
    assert len(train_only) == n_train * 8
    assert len(load_dataset(ds / "train", split="train")) == n_train * 8


def test_resolve_dataset_split(tmp_path):
    human = tmp_path / "human"
    (human / "train").mkdir(parents=True)
    (human / "train" / "episode_000000.npz").write_bytes(b"")
    (human / "validation").mkdir(parents=True)
    (human / "validation" / "episode_000001.npz").write_bytes(b"")

    # a split dataset resolves to itself...
    assert resolve_dataset(human) == human
    # ...and the data root resolves to it (preferring human)
    assert resolve_dataset(tmp_path) == human

    # a single split folder is a plain dataset folder
    assert resolve_dataset(human / "train") == human / "train"

    # a root with a split dataset plus another dataset: human is preferred
    multi = tmp_path / "multi"
    _make_dataset(multi, "a")
    (multi / "human" / "train").mkdir(parents=True)
    (multi / "human" / "train" / "episode_000000.npz").write_bytes(b"")
    assert resolve_dataset(multi) == multi / "human"

    # a root with several non-human split datasets is ambiguous
    two = tmp_path / "two"
    for name in ("a", "b"):
        (two / name / "train").mkdir(parents=True)
        (two / name / "train" / "episode_000000.npz").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        resolve_dataset(two)


def test_load_training_dataset_split(tiny_env, tmp_path, capsys):
    """A split dataset trains on its train split by default; --split
    overrides it."""
    import argparse
    from train.common import load_training_dataset

    root = tmp_path / "data"
    ds, n_train, n_val = _split_dataset(tiny_env, root)

    args = argparse.Namespace(data=str(root), no_synthetic=False, ai_data=False,
                              split=None)
    dataset, data_dir = load_training_dataset(args)
    assert data_dir == ds
    assert dataset.split == "train"
    assert len(dataset) == n_train * 8
    assert "using the train split" in capsys.readouterr().out

    args = argparse.Namespace(data=str(root), no_synthetic=False, ai_data=False,
                              split="validation")
    dataset, _ = load_training_dataset(args)
    assert dataset.split == "validation"
    assert len(dataset) == n_val * 8


def test_load_training_dataset_ai_data(tiny_env, tmp_path):
    """--ai-data adds the synthetic AI data (data/ai) to the resolved
    dataset; without it the training data is the resolved dataset alone."""
    import argparse
    from train.common import load_training_dataset

    root = tmp_path / "data"
    for name in ("human", "ai"):
        TrajectoryCollector(tiny_env, out_dir=root / name).collect_episodes(
            n_episodes=1, policy=None, max_steps=8, verbose=False)

    args = argparse.Namespace(data=str(root), no_synthetic=False, ai_data=False)
    dataset, data_dir = load_training_dataset(args)
    assert data_dir == root / "human"  # the default resolution is unchanged
    assert len(dataset) == 8

    args = argparse.Namespace(data=str(root), no_synthetic=False, ai_data=True)
    dataset, data_dir = load_training_dataset(args)
    assert data_dir == root / "human"
    assert len(dataset) == 16  # human + ai


def test_collect_human_mode(tiny_env, tmp_path):
    env = HumanTinyEnv(tiny_env)
    collector = TrajectoryCollector(env, out_dir=tmp_path / "ds")
    stats = collector.collect_episodes(n_episodes=2, policy="human", max_steps=8,
                                       verbose=False)

    assert len(stats) == 2
    assert len(list((tmp_path / "ds").glob("episode_*.npz"))) == 2

    # the recorded actions must be the read-back applied actions, not the
    # dummy zeros the collector steps with
    with np.load(tmp_path / "ds" / "episode_000000.npz", allow_pickle=True) as data:
        actions = data["action"]
    assert actions[0][0] == pytest.approx(0.25)   # t=1
    assert actions[1][0] == pytest.approx(0.5)    # t=2
    assert actions[3][0] == pytest.approx(1.0)    # t=4 -> full lock
    assert actions[7][0] == pytest.approx(1.0)    # stays clipped


def test_collect_ai_mode(tiny_env, tmp_path):
    """Synthetic mode: the built-in AI drives, so the collector uses the same
    read-back mechanism as human mode (policy="ai")."""
    env = HumanTinyEnv(tiny_env)
    collector = TrajectoryCollector(env, out_dir=tmp_path / "ds")
    stats = collector.collect_episodes(n_episodes=2, policy="ai", max_steps=8,
                                       verbose=False)

    assert len(stats) == 2
    assert len(list((tmp_path / "ds").glob("episode_*.npz"))) == 2

    with np.load(tmp_path / "ds" / "episode_000000.npz", allow_pickle=True) as data:
        actions = data["action"]
    assert actions[0][0] == pytest.approx(0.25)   # t=1
    assert actions[7][0] == pytest.approx(1.0)    # stays clipped
    assert np.allclose(actions[:, 1], 0.5)        # throttle
    # the gear vector uses the env's action encoding (gear_to_action), not the
    # obs encoding: neutral (t<3) -> gear_to_action(0), 1st (t>=3) -> gear_to_action(1)
    assert np.allclose(actions[:2, 3:], gear_to_action(0))
    assert np.allclose(actions[2:, 3:], gear_to_action(1))
    assert np.allclose(actions[:, 2], 0.0)        # parking brake

    # the dataset loads and the actions are the applied ones
    dataset = load_dataset(tmp_path / "ds")
    assert len(dataset) == 2 * 8
    assert dataset[0]["action"][0] == pytest.approx(0.25)


def test_collect_human_infinite_interrupt(tiny_env, tmp_path):
    env = HumanTinyEnv(tiny_env)
    original_step = env.step
    calls = {"n": 0}

    def stepping(action):
        calls["n"] += 1
        if calls["n"] == 12:  # 8 (ep1) + 4 (into ep2)
            raise KeyboardInterrupt
        return original_step(action)

    env.step = stepping
    collector = TrajectoryCollector(env, out_dir=tmp_path / "ds")
    stats = collector.collect_episodes(n_episodes=0, policy="human", max_steps=8,
                                       verbose=False)

    # only the first full episode is kept; the in-progress one is discarded
    assert len(stats) == 1
    assert len(list((tmp_path / "ds").glob("episode_*.npz"))) == 1


def test_bc_training_reduces_loss(tiny_env, tmp_path, capsys):
    from train.commands.offline import train_bc, make_dummy_env
    import argparse
    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MultiInputPolicy as PPOMultiInputPolicy

    ds = tmp_path / "ds"
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=4, policy=None, max_steps=8, verbose=False)
    dataset = load_dataset(ds)
    buffer = to_replay_buffer(dataset)

    args = argparse.Namespace(algo="ppo", steps=60, batch_size=32, lr=3e-3,
                              entropy_coef=0.0, log_every=30, device="cpu",
                              init=None, out=str(tmp_path / "bc.zip"))
    train_bc(args, buffer, dataset.observation_space, dataset.action_space)
    out = capsys.readouterr().out
    assert "rmse=" in out  # the log line reports nll, entropy, and rmse
    assert (tmp_path / "bc.zip").exists()

    # the saved model must load and predict
    model = PPO.load(args.out, env=make_dummy_env(dataset.observation_space,
                                                  dataset.action_space), device="cpu")
    obs, _ = tiny_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == dataset.action_space.shape


def test_bc_td3_regression(tiny_env, tmp_path, capsys):
    from train.commands.offline import train_bc, make_dummy_env
    import argparse
    from stable_baselines3 import TD3

    ds = tmp_path / "ds"
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=4, policy=None, max_steps=8, verbose=False)
    dataset = load_dataset(ds)
    buffer = to_replay_buffer(dataset)

    args = argparse.Namespace(algo="td3", steps=60, batch_size=32, lr=3e-3,
                              entropy_coef=0.0, log_every=30, device="cpu",
                              init=None, out=str(tmp_path / "bc_td3.zip"))
    train_bc(args, buffer, dataset.observation_space, dataset.action_space)
    out = capsys.readouterr().out
    assert "mse=" in out and "rmse=" in out  # the TD3 log line reports mse + rmse
    assert (tmp_path / "bc_td3.zip").exists()

    # the saved model must load as a TD3 and predict
    model = TD3.load(args.out, env=make_dummy_env(dataset.observation_space,
                                                  dataset.action_space), device="cpu")
    obs, _ = tiny_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == dataset.action_space.shape


def test_bc_saves_on_interrupt(tiny_env, tmp_path, monkeypatch):
    from train.commands.offline import train_bc
    import argparse

    monkeypatch.chdir(tmp_path)  # the default out path is relative
    ds = tmp_path / "ds"
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=2, policy=None, max_steps=8, verbose=False)
    dataset = load_dataset(ds)
    buffer = to_replay_buffer(dataset)

    args = argparse.Namespace(algo="ppo", steps=100, batch_size=8, lr=3e-3,
                              entropy_coef=0.0, log_every=1000, device="cpu",
                              init=None, out=None, data=str(ds))
    calls = {"n": 0}
    original_sample = buffer.sample

    def sample(batch_size):
        calls["n"] += 1
        if calls["n"] == 3:  # interrupt mid-training
            raise KeyboardInterrupt
        return original_sample(batch_size)

    buffer.sample = sample
    train_bc(args, buffer, dataset.observation_space, dataset.action_space)

    # the partial model is saved with the step count in the name, under the
    # algorithm's subfolder
    saved = list((tmp_path / "models/bc/ppo").glob("*.zip"))
    assert [p.name for p in saved] == ["ds_steps3.zip"]


def test_resume_from_cli_values(tmp_path):
    from train.common import ResumeFrom, find_latest_model

    # the CLI passes "last"/"best", which must map onto the enum
    assert ResumeFrom("last") is ResumeFrom.LAST
    assert ResumeFrom("best") is ResumeFrom.BEST

    run = tmp_path / "2026.01.01-00.00"
    run.mkdir()
    (run / "last_model.zip").write_bytes(b"x")
    assert find_latest_model(tmp_path, ResumeFrom("last")) == run / "last_model.zip"

    # missing checkpoint falls back to the other type
    (run / "last_model.zip").unlink()
    (run / "best_model.zip").write_bytes(b"x")
    assert find_latest_model(tmp_path, ResumeFrom("last")) == run / "best_model.zip"

    # no checkpoint in any run -> train from scratch
    (run / "best_model.zip").unlink()
    assert find_latest_model(tmp_path, ResumeFrom("last")) is None


def test_curate_list(tiny_env, tmp_path, capsys):
    from data.curate import main as curate_main

    env = SpawnTinyEnv(tiny_env)
    ds = tmp_path / "ds"
    TrajectoryCollector(env, out_dir=ds).collect_episodes(
        n_episodes=2, policy=None, max_steps=8, verbose=False)

    curate_main(["list", "--data", str(ds)])
    out = capsys.readouterr().out
    assert "episode_000000.npz" in out
    assert "episode_000001.npz" in out
    assert "wp_test" in out
    assert "8 steps (0.4 s)" in out
    assert "0.000" in out  # the tiny env takes no damage
    assert "reward:" in out

    # episodes are listed from highest to lowest reward per step
    from data import curate
    # ep0: 8 steps x 1.5 = 12.0 total (1.5/step)
    # ep1: 4 steps x 2.5 = 10.0 total (2.5/step)
    # the total order (ep0 first) differs from the per-step order (ep1 first)
    p0 = ds / "episode_000000.npz"
    ep = curate._load_episode(p0)
    ep["reward"][:] = 1.5
    curate._save_episode(p0, ep)
    p1 = ds / "episode_000001.npz"
    ep = curate._load_episode(p1)
    ep["reward"] = ep["reward"][:4]
    ep["reward"][:] = 2.5
    curate._save_episode(p1, ep)
    curate_main(["list", "--data", str(ds)])
    out = capsys.readouterr().out
    assert out.index("episode_000001.npz") < out.index("episode_000000.npz")

    # --min-seconds lists only the episodes at least that long; with several
    # values it lists the episodes above each threshold in turn
    curate_main(["list", "--data", str(ds), "--min-seconds", "0.3"])
    out = capsys.readouterr().out
    assert "episode_000000.npz" in out
    assert "episode_000001.npz" not in out

    curate_main(["list", "--data", str(ds), "--min-seconds", "0.3", "0.2"])
    out = capsys.readouterr().out
    assert out.index(">= 0.2s") < out.index(">= 0.3s")
    assert out.count("episode_000000.npz") == 4  # in both sections
    assert out.count("episode_000001.npz") == 2  # only in the >= 0.2s section

    # a single-episode listing prints only that episode
    curate_main(["list", "--data", str(ds), "--episode", "1"])
    out = capsys.readouterr().out
    assert "episode_000001.npz" in out
    assert "episode_000000.npz" not in out

    # episodes without a spawn field (collected before the env had one)
    # list the spawn as "-"
    ds2 = tmp_path / "ds2"
    TrajectoryCollector(tiny_env, out_dir=ds2).collect_episodes(
        n_episodes=1, policy=None, max_steps=8, verbose=False)
    curate_main(["list", "--data", str(ds2)])
    out = capsys.readouterr().out
    assert "spawn:  -" in out

    # a missing episode index is reported
    curate_main(["list", "--data", str(ds), "--episode", "99"])
    out = capsys.readouterr().out
    assert "no such episode" in out


def _inject_damage(path, step, amount=1.0):
    """Mark one step of an episode as taking damage (via info reward_components)."""
    from data import curate

    ep = curate._load_episode(path)
    rc = dict(ep["info"][step].get("reward_components", {}))
    rc["dmg_delta"] = amount
    ep["info"][step]["reward_components"] = rc
    curate._save_episode(path, ep)


def test_curate_truncate_damage(tiny_env, tmp_path, capsys):
    from data import curate
    from data.curate import main as curate_main

    ds = tmp_path / "ds"
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=3, policy=None, max_steps=8, verbose=False)
    _inject_damage(ds / "episode_000000.npz", step=6)
    _inject_damage(ds / "episode_000001.npz", step=2)

    # dry run: ep0 first damage at 6 -> cut 4 (margin 2), ep1 at 2 -> cut 0
    # which is below min-keep 1, so it would be deleted instead
    curate_main(["truncate-damage", "--data", str(ds),
                 "--margin", "2", "--min-keep", "1", "--dry-run"])
    out = capsys.readouterr().out
    assert "would truncate episode_000000.npz: 8 -> 4 steps" in out
    assert "would delete episode_000001.npz (8 steps)" in out
    assert "would truncate 1, would delete 1 episode(s), losing 12 of 24 steps (50.0%)" in out
    assert len(list(ds.glob("episode_*.npz"))) == 3  # nothing changed

    # for real: ep0 truncated to 4 steps, ep1 deleted, ep2 untouched
    curate_main(["truncate-damage", "--data", str(ds),
                 "--margin", "2", "--min-keep", "1"])
    out = capsys.readouterr().out
    assert "truncated 1, deleted 1 episode(s)" in out
    assert not (ds / "episode_000001.npz").exists()

    ep = curate._load_episode(ds / "episode_000000.npz")
    assert len(ep["reward"]) == 4
    assert len(ep["info"]) == 4
    assert not (curate._damage_series(ep) > 0.1).any()  # damage step removed
    assert not ep["terminated"].any()
    assert ep["truncated"][-1] and not ep["truncated"][:-1].any()
    assert int(ep["meta"]["ep_length"]) == 4
    assert ep["meta"]["ep_terminated"] == False
    assert ep["meta"]["ep_truncated"] == True
    assert float(ep["meta"]["ep_reward"]) == pytest.approx(float(ep["reward"].sum()))

    untouched = curate._load_episode(ds / "episode_000002.npz")
    assert len(untouched["reward"]) == 8


def test_curate_truncate_short(tiny_env, tmp_path, capsys):
    from data import curate
    from data.curate import main as curate_main

    ds = tmp_path / "ds"
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=2, policy=None, max_steps=8, verbose=False)
    # one longer episode into the same dir (the numbering continues)
    tiny_env.horizon = 16
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=1, policy=None, max_steps=16, verbose=False)
    # damage is irrelevant: the cut is at the end, not at the damage step
    _inject_damage(ds / "episode_000001.npz", step=3)

    # --margin is required
    with pytest.raises(SystemExit):
        curate_main(["truncate-short", "--data", str(ds)])

    # dry run: only the two short episodes are considered, cut at end - margin
    curate_main(["truncate-short", "--data", str(ds),
                 "--max-steps", "10", "--margin", "2", "--min-keep", "1",
                 "--dry-run"])
    out = capsys.readouterr().out
    assert "would truncate episode_000000.npz: 8 -> 6 steps (margin 2)" in out
    assert "would truncate episode_000001.npz: 8 -> 6 steps (margin 2)" in out
    assert "would truncate 2, would delete 0 episode(s), losing 4 of 32 steps (12.5%)" in out
    assert len(list(ds.glob("episode_*.npz"))) == 3  # nothing changed

    curate_main(["truncate-short", "--data", str(ds),
                 "--max-steps", "10", "--margin", "2", "--min-keep", "1"])
    out = capsys.readouterr().out
    assert "truncated 2, deleted 0 episode(s)" in out

    ep0 = curate._load_episode(ds / "episode_000000.npz")
    assert len(ep0["reward"]) == 6
    ep1 = curate._load_episode(ds / "episode_000001.npz")
    assert len(ep1["reward"]) == 6  # cut at the end, not at the damage step
    long_ep = curate._load_episode(ds / "episode_000002.npz")
    assert len(long_ep["reward"]) == 16  # untouched


def test_curate_truncate_start(tiny_env, tmp_path, capsys):
    from data import curate
    from data.curate import main as curate_main

    ds = tmp_path / "ds"
    # tiny_env shifts neutral -> 1st on step 3, so the leading idle is steps 0-1
    TrajectoryCollector(tiny_env, out_dir=ds).collect_episodes(
        n_episodes=1, policy=None, max_steps=8, verbose=False)

    # dry run: --lead 1 starts the episode 1 step before the shift (step 2),
    # dropping the first 2 steps; nothing is written
    curate_main(["truncate-start", "--data", str(ds), "--lead", "1", "--dry-run"])
    out = capsys.readouterr().out
    assert "would truncate episode_000000.npz: drop first 2 steps" in out
    assert "first gear change at 3" in out
    assert "would truncate 1, skip 0 episode(s)" in out
    assert len(list(ds.glob("episode_*.npz"))) == 1  # nothing changed

    curate_main(["truncate-start", "--data", str(ds), "--lead", "1"])
    out = capsys.readouterr().out
    assert "truncated 1, skipped 0 episode(s)" in out

    ep = curate._load_episode(ds / "episode_000000.npz")
    assert len(ep["reward"]) == 6  # 8 - 2
    gear = curate._gear_series(ep)
    assert gear[0] == 0            # first step is the last neutral step
    assert (gear[1:] == 1).all()    # the rest is 1st; the shift is step 1 now

    # idempotent: a second run finds no leading idle to drop (shift at step 1,
    # --lead 1 -> start 0) and skips the episode
    curate_main(["truncate-start", "--data", str(ds), "--lead", "1", "--dry-run"])
    out = capsys.readouterr().out
    assert "would truncate 0, skip 1 episode(s)" in out

def test_find_pairs_from_dataset(tmp_path, monkeypatch, capsys):
    """A dataset directory yields the replays its episodes reference, in
    episode order (not the loose .rpl files it may contain)."""
    from data.regenerate import _find_pairs

    data = tmp_path / "human"
    (data / "replays").mkdir(parents=True)
    game = tmp_path / "game"
    (game / "replays").mkdir(parents=True)
    monkeypatch.setenv("BNG_SAVE_DIR", str(game))

    # episode 0's replay lives in the dataset's replays/ subfolder
    r0 = data / "replays" / "a.rpl"
    r0.write_bytes(b"rpl")
    # episode 1's replay lives in the game's replays folder
    r1 = game / "replays" / "b.rpl"
    r1.write_bytes(b"rpl")
    for i, rep in enumerate(["replays/a.rpl", "replays/b.rpl"]):
        np.savez_compressed(data / f"episode_{i:06d}.npz",
                            replay=np.array(rep))
    # episode 2 has no replay recorded: it is skipped with a note
    np.savez_compressed(data / "episode_000002.npz", replay=np.array(""))
    # a stale .rpl loose in the dataset dir is not picked up
    (data / "stale.rpl").write_bytes(b"rpl")

    assert _find_pairs(data) == [(data / "episode_000000.npz", r0),
                                   (data / "episode_000001.npz", r1)]
    out = capsys.readouterr().out
    assert "episode_000002.npz" in out
    assert "no replay recorded" in out

    # the old modes still work: a single .rpl, and a bare folder of .rpl
    assert _find_pairs(r1) == [(None, r1)]
    folder = tmp_path / "folder"
    folder.mkdir()
    f0 = folder / "x.rpl"
    f0.write_bytes(b"rpl")
    assert _find_pairs(folder) == [(None, f0)]


def test_compute_reward():
    from beamng_tech_env.core.constants import (
        PROGRESS_BREAK_DIST,
        PROGRESS_RATE,
    )
    from beamng_tech_env.core.reward import compute_reward

    # stationary, undamaged: nothing to reward, the reward is 0
    total, comps = compute_reward(0.0, 0.0, 0.0)
    assert total == 0.0
    assert comps["progress_reward"] == 0.0
    assert comps["drift_reward"] == 0.0

    # jiggling (below the break-even distance): the distance reward is held 0
    total, comps = compute_reward(PROGRESS_BREAK_DIST * 0.5, 0.0, 0.0)
    assert comps["progress_reward"] == 0.0
    assert total == 0.0

    # moving, drifting: progress + drift, no other components
    # (centerline_dist=0.0 = on the centerline, so the progress is unscaled)
    total, comps = compute_reward(0.5, 50.0, 0.0, 0.0)
    assert comps["progress_reward"] == PROGRESS_RATE * 0.5 - PROGRESS_BREAK_DIST
    assert comps["drift_reward"] == min(0.8, 50.0 * 0.01)
    assert total == comps["progress_reward"] + comps["drift_reward"]

    # damage is penalized in the total (capped at -1.0)
    total, comps = compute_reward(0.5, 50.0, 200.0, 0.0)
    assert comps["dmg_delta"] == 200.0
    assert comps["dmg_reward"] == -1.0
    assert total == (min(0.8, 50.0 * 0.01)
                     + (PROGRESS_RATE * 0.5 - PROGRESS_BREAK_DIST) - 1.0)

    # a big drift score is capped at the drift reward ceiling
    total, comps = compute_reward(0.0, 200.0, 0.0)
    assert comps["drift_reward"] == 0.8


def test_compute_reward_multipliers():
    from beamng_tech_env.core.constants import (
        OFFROAD_REWARD_MULT,
        PROGRESS_BREAK_DIST,
        PROGRESS_RATE,
        REVERSING_REWARD_MULT,
    )
    from beamng_tech_env.core.reward import compute_reward

    base = (min(0.8, 50.0 * 0.01)
            + (PROGRESS_RATE * 0.5 - PROGRESS_BREAK_DIST))

    # on-road, moving forward: the full reward, no multiplier
    total, comps = compute_reward(0.5, 50.0, 0.0, 0.0)
    assert comps["reward_mult"] == 1.0
    assert total == base

    # off-road: the total reward is scaled to OFFROAD_REWARD_MULT
    total, comps = compute_reward(0.5, 50.0, 0.0, 0.0, offroad=True)
    assert comps["offroad"] is True
    assert comps["reward_mult"] == OFFROAD_REWARD_MULT
    assert total == base * OFFROAD_REWARD_MULT

    # reversing (the drift data's flag): the total reward is scaled to
    # REVERSING_REWARD_MULT
    total, comps = compute_reward(0.5, 50.0, 0.0, 0.0, reversing=True)
    assert comps["reversing"] is True
    assert comps["reward_mult"] == REVERSING_REWARD_MULT
    assert total == base * REVERSING_REWARD_MULT

    # both: the lowest multiplier wins (the two never stack)
    total, comps = compute_reward(0.5, 50.0, 0.0, 0.0, offroad=True,
                                  reversing=True)
    assert comps["reward_mult"] == min(OFFROAD_REWARD_MULT,
                                       REVERSING_REWARD_MULT)
    assert total == base * min(OFFROAD_REWARD_MULT, REVERSING_REWARD_MULT)

    # the damage penalty is part of the total, so the multiplier scales it
    total, comps = compute_reward(0.5, 50.0, 200.0, 0.0, offroad=True)
    assert total == (base - 1.0) * OFFROAD_REWARD_MULT


def test_compute_reward_centerline():
    from beamng_tech_env.core.constants import (
        CENTERLINE_DIST_MAX,
        PROGRESS_BREAK_DIST,
        PROGRESS_RATE,
    )
    from beamng_tech_env.core.reward import compute_reward

    base = PROGRESS_RATE * 0.5 - PROGRESS_BREAK_DIST   # the progress at 0.5 m

    # on the centerline (dist 0): the full progress reward is kept
    total, comps = compute_reward(0.5, 0.0, 0.0, 0.0)
    assert comps["centerline_norm"] == 1.0
    assert comps["progress_reward"] == base

    # halfway (dist 20 of 40): the progress reward is halved
    total, comps = compute_reward(0.5, 0.0, 0.0, 20.0)
    assert comps["centerline_norm"] == 0.5
    assert comps["progress_reward"] == base * 0.5

    # at the max (dist 40): the progress reward is 0
    total, comps = compute_reward(0.5, 0.0, 0.0, CENTERLINE_DIST_MAX)
    assert comps["centerline_norm"] == 0.0
    assert comps["progress_reward"] == 0.0

    # beyond the max is clamped (still 0 progress)
    total, comps = compute_reward(0.5, 0.0, 0.0, 100.0)
    assert comps["centerline_dist"] == CENTERLINE_DIST_MAX
    assert comps["progress_reward"] == 0.0

    # no centerline (off-road, NaN): treated as far (0 progress)
    total, comps = compute_reward(0.5, 0.0, 0.0, float("nan"))
    assert comps["centerline_norm"] == 0.0
    assert comps["progress_reward"] == 0.0
    # the drift reward is unaffected by the centerline
    assert comps["drift_reward"] == 0.0
    assert total == 0.0


def test_align_episode():
    from data.regenerate import _align_episode

    # the episode's airspeed is the .rpl's, shifted by 9 s (a curated
    # episode starts part-way into its .rpl); the speed ramps and wobbles,
    # so the offset is distinctive (a pure ramp would match every window)
    t_rpl = np.arange(0.0, 25.0, 0.05)
    air_rpl = _synthetic_speed(t_rpl) * 3.6   # kph
    T = 100
    air_ep = air_rpl[180:180 + T]             # 9 s in
    offset, corr = _align_episode(air_ep, air_rpl)
    assert offset == 9.0
    assert corr > 0.999

    # a (nearly) stationary episode has no signal: offset 0 is assumed
    offset, corr = _align_episode(np.zeros(50), air_rpl)
    assert offset == 0.0
    assert corr == 1.0

    # a .rpl shorter than the episode can't hold it
    offset, corr = _align_episode(air_ep, air_rpl[:50])
    assert offset is None


def _synthetic_speed(t):
    """The synthetic .rpl's speed (m/s): a ramp with a chirp wobble, so the
    alignment has a distinctive signal (a pure ramp matches every window, and
    a fixed-period wobble aliases with the 5 s window)."""
    return 5.0 + 2.0 * t + 8.0 * np.sin(2.0 * np.pi * t * t / 40.0)


def _synthetic_rpl():
    """A 25 s 40 Hz .rpl: the car drives -Y (its forward) at
    _synthetic_speed, in 2nd gear. Built the same way RplTrajectory builds
    itself."""
    from beamng_tech_env.core.rpl import RplTrajectory

    t = np.arange(0.0, 25.0, 0.025)
    v = _synthetic_speed(t)
    y = -np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1])
                                         * (t[1] - t[0]))])
    pos = np.column_stack([np.zeros_like(t), y, np.zeros_like(t)])
    quat = np.zeros((len(t), 4))
    quat[:, 3] = 1.0   # identity: forward = (0, -1, 0)
    rpl = RplTrajectory.__new__(RplTrajectory)
    rpl.t = t
    rpl.pos = pos
    rpl.quat = quat
    rpl.vel = np.empty_like(pos)
    rpl.vel[1:-1] = (pos[2:] - pos[:-2]) / (t[2:] - t[:-2])[:, None]
    rpl.vel[0] = (pos[1] - pos[0]) / (t[1] - t[0])
    rpl.vel[-1] = (pos[-1] - pos[-2]) / (t[-1] - t[-2])
    e = {"gear": 2.0, "rpm": 3000.0, "maxrpm": 6250.0, "isShifting": 0.0,
         "throttle_input": 0.5, "brake_input": 0.0, "parkingbrake_input": 0.0,
         "steering": 0.0}
    rpl.electrics = [dict(e) for _ in t]
    return rpl


def _synthetic_navigraph():
    """A straight road along the synthetic .rpl's path, 3.5 m half-width.
    Short segments (like the real navigraph's edges): Navigraph.nearby only
    checks a segment's endpoints/midpoint, so a single long segment would
    drop the road mid-way."""
    from beamng_tech_env.core.roads import Navigraph

    ys = np.arange(0.0, -300.0, -10.0)
    pts = np.column_stack([np.zeros_like(ys), ys, np.zeros_like(ys)])
    nav = Navigraph.__new__(Navigraph)
    nav.a = pts[:-1]
    nav.b = pts[1:]
    nav.hw_a = np.full(len(nav.a), 3.5)
    nav.hw_b = np.full(len(nav.b), 3.5)
    nav.mid = 0.5 * (nav.a + nav.b)
    nav.max_hw = 3.5
    return nav


def test_regen_offline_roundtrip():
    """Offline regeneration: the .rpl drives the rebuild (airspeed/gear/
    action/road BEV/reward recomputed), the FFB + damage are reused from the
    original episode, the LiDAR BEV is corrected in place (the "bev" channel
    is the original LiDAR corrected against the road network), and the
    curated episode's offset into the .rpl is found."""
    from beamng_tech_env.core.constants import (
        AIRSPEED_MAX_KPH,
        BEV_ROAD_LEVEL,
        PROGRESS_BREAK_DIST,
        PROGRESS_RATE,
        gear_to_action,
    )
    from data.regenerate import _regenerate_offline

    rpl = _synthetic_rpl()
    navigraph = _synthetic_navigraph()
    T = 100
    offset = 9.0
    t20 = offset + np.arange(T) * 0.05
    v = _synthetic_speed(t20)                 # m/s at each step
    # the original LiDAR: below the road level everywhere (the correction
    # must fill the road surface to the road level and leave the off-road
    # returns untouched - not replace them with the road network)
    bev = np.full((T, 1, 64, 64), 100, np.uint8)
    ffb = np.zeros((T, 1), np.float32)
    ep = {
        "obs": {
            "airspeed": (v * 3.6 / AIRSPEED_MAX_KPH).astype(np.float32)
                        .reshape(T, 1),
            "bev": bev,
            "ffb": ffb,
        },
        "action": np.zeros((T, 4), np.float32),
        "reward": np.zeros(T, np.float32),
        "terminated": np.zeros(T, bool),
        "truncated": np.ones(T, bool),
        "info": [{"reward_components": {"dmg_delta": 0.0}} for _ in range(T)],
        "meta": {"replay": np.array("replays/x.rpl"),
                 "ep_reward": np.float32(0.0)},
    }
    out = _regenerate_offline(ep, rpl, navigraph)
    assert out is not None

    # the FFB is reused from the original, untouched; the LiDAR BEV is the
    # original, corrected against the road network (road surface = road
    # level, the off-road returns as the LiDAR saw them)
    assert np.array_equal(out["obs"]["ffb"], ffb)
    bev = out["obs"]["bev"]
    assert bev.dtype == np.uint8
    road = out["obs"]["road_bev"].reshape(T, 64, 64)
    np.testing.assert_array_equal(
        bev.reshape(T, 64, 64),
        np.where(road > 0.5, BEV_ROAD_LEVEL, 100).astype(np.uint8))
    # the .rpl-derived channels are recomputed at the aligned time
    np.testing.assert_allclose(out["obs"]["airspeed"].reshape(-1),
                               v * 3.6 / AIRSPEED_MAX_KPH, atol=1e-2)
    assert (out["obs"]["gear"] == 3.0 / 7.0).all()
    assert (out["obs"]["rpm"] == 3000.0 / 6250.0).all()
    # the recorded action: half throttle, 2nd gear (encoded via ALLOWED_GEARS)
    np.testing.assert_allclose(out["action"][:, 1], 0.5, atol=1e-6)
    np.testing.assert_allclose(
        out["action"][:, 3:],
        np.tile(gear_to_action(2), (out["action"].shape[0], 1)), atol=1e-6)
    # the road BEV covers the road the car is on
    assert (out["obs"]["road_bev"].reshape(T, 64, 64) > 0.5).any(axis=(1, 2)).all()
    # the reward: progress (per-step distance - break-even, held at 0 when
    # below it) + zero drift (the car drives straight); the per-step distance
    # is the chord of the .rpl's path on the episode's 20 Hz grid
    y20 = rpl.pos[(t20 / 0.025).round().astype(int), 1]
    dist = np.zeros(T)
    dist[1:] = np.abs(np.diff(y20))
    expected = np.maximum(0.0, PROGRESS_RATE * dist - PROGRESS_BREAK_DIST)
    np.testing.assert_allclose(out["reward"], expected, atol=0.02)
    assert out["info"][0]["reward_components"]["dmg_delta"] == 0.0
    # the episode's flags/replay are carried over
    assert np.array_equal(out["truncated"], ep["truncated"])
    assert str(out["meta"]["replay"]) == "replays/x.rpl"
    np.testing.assert_allclose(float(out["meta"]["ep_reward"]),
                               float(expected.sum()), atol=1e-3)

    # an episode the .rpl can't match is skipped (None): a profile with no
    # correlation to the .rpl's (an alternating one, not a ramp - two ramps
    # would correlate perfectly)
    ep2 = dict(ep)
    ep2["obs"] = dict(ep["obs"])
    ep2["obs"]["airspeed"] = np.where(np.arange(T) % 2 == 0,
                                      0.1, 0.9).astype(np.float32).reshape(T, 1)
    assert _regenerate_offline(ep2, rpl, navigraph) is None


def test_out_index_keeps_numbering():
    """Regenerated episodes keep the source dataset's numbering: a dataset
    episode takes its own index (a skipped one leaves a gap, not a shift),
    and a bare .rpl takes the next free index."""
    from pathlib import Path
    from data.regenerate import _episode_index, _out_index

    # the index is read from the episode filename
    assert _episode_index(Path("episode_000017.npz")) == 17
    assert _episode_index(Path("episode_000000.npz")) == 0
    assert _episode_index(Path("drift_2026-10-16_000000.rpl")) is None
    assert _episode_index(Path("other.npz")) is None

    # a dataset episode keeps its source index
    idx, next_index = _out_index(Path("episode_000017.npz"), 0)
    assert idx == 17
    assert next_index == 18
    # a bare .rpl takes the next free index
    idx, next_index = _out_index(Path("r0.rpl"), 5)
    assert idx == 5
    assert next_index == 6
    # a skipped episode leaves a gap: the next one keeps its own index
    idx, next_index = _out_index(Path("episode_000001.npz"), 18)
    assert idx == 1
    assert next_index == 18
