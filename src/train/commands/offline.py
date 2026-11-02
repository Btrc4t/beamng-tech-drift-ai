"""Offline training on a recorded dataset (no game required).

The observation is the env's float channels plus 20 ultrasonic beams derived
from the road BEV (the default - distance to the road edge; the LiDAR BEV is
the fallback for old datasets); ``--no-beams`` observes the raw BEV images
instead.

``train.py offline --algo {ppo,td3}`` - behavior cloning (SB3): PPO imitates
the recorded actions by maximising their log-likelihood; TD3 is deterministic,
so it regresses its action output onto them (MSE) instead. Needs no reward
signal; it can only match the quality of the data, never exceed it.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th

from data.dataset import resolve_dataset
from beamng_tech_env.wrappers import beams_space, obs_to_beams
from train.common import ALGOS, DEFAULT_STEPS, load_training_dataset


def make_dummy_env(obs_space: gym.spaces.Space, act_space: gym.spaces.Space) -> gym.Env:
    """A do-nothing env with the dataset's spaces.

    Only used to instantiate an SB3 model (which reads the spaces and builds
    the policy); it is never stepped, so the dynamics are irrelevant.
    """

    class _Dummy(gym.Env):
        observation_space = obs_space
        action_space = act_space

        def reset(self, seed=None, options=None):
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, True, False, {}

    return _Dummy()


def build_replay_buffer(dataset, use_beams: bool = True,
                        device: str = "cpu"):
    """An SB3 DictReplayBuffer from a dataset, in the requested observation
    mode: "beams" (20 floats, the default) or the raw "bev" image.
    ``done`` is the terminated flag and each info carries
    TimeLimit.truncated so SB3 bootstraps across truncations."""
    from stable_baselines3.common.buffers import DictReplayBuffer

    float_keys = [k for k in dataset.observation_space.spaces.keys()
                  if k not in ("bev", "road_bev")]
    space = beams_space(dataset.observation_space, use_beams)
    buffer = DictReplayBuffer(len(dataset), space, dataset.action_space,
                              device=device)
    for ep in dataset.episodes:
        t = len(ep["reward"])
        beams = obs_to_beams(ep["obs"]) if use_beams else None
        bev_keys = [k for k in ("bev", "road_bev") if k in ep["obs"]]
        for s in range(t):
            obs = {k: ep["obs"][k][s] for k in float_keys}
            if use_beams:
                obs["beams"] = beams[s]
            else:
                for key in bev_keys:
                    obs[key] = ep["obs"][key][s]
            if s + 1 < t:
                next_obs = {k: ep["obs"][k][s + 1] for k in float_keys}
                if use_beams:
                    next_obs["beams"] = beams[s + 1]
                else:
                    for key in bev_keys:
                        next_obs[key] = ep["obs"][key][s + 1]
            else:
                next_obs = obs
            buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=ep["action"][s],
                reward=np.asarray(float(ep["reward"][s]), dtype=np.float32),
                done=np.asarray([bool(ep["terminated"][s])], dtype=np.float32),
                infos=[{"TimeLimit.truncated": bool(ep["truncated"][s])}],
            )
    buffer.full = True
    return buffer


def _move(batch_obs, device):
    if isinstance(batch_obs, dict):
        return {k: v.to(device) for k, v in batch_obs.items()}
    return batch_obs.to(device)


def train_bc(args, buffer, obs_space, act_space) -> None:
    """Behavior cloning: imitate the recorded actions.

    Uses the policy's own objective. A Gaussian policy (PPO) has a
    distribution over actions, so it maximises the log-likelihood of the
    recorded actions. A deterministic policy (TD3) has no distribution, so it
    regresses its action output onto the recorded actions (MSE) instead.
    """
    MODEL, POLICY = ALGOS[args.algo]
    print(f"Training config: {MODEL} {POLICY}")
    model = MODEL(POLICY, make_dummy_env(obs_space, act_space),
                  verbose=0, tensorboard_log=None, device=args.device)
    if args.init:
        model = MODEL.load(args.init, env=make_dummy_env(obs_space, act_space),
                           verbose=0, device=args.device)
    policy = model.policy
    policy.to(args.device)
    optimizer = th.optim.Adam(policy.parameters(), lr=args.lr)
    policy.set_training_mode(True)
    # PPO policies expose a Gaussian distribution; TD3 is deterministic and
    # only has forward(), so the cloning objective differs
    stochastic = hasattr(policy, "get_distribution")

    step = 0
    try:
        for step in range(1, args.steps + 1):
            batch = buffer.sample(args.batch_size)
            obs = _move(batch.observations, args.device)
            actions = batch.actions.to(args.device)
            if stochastic:
                dist = policy.get_distribution(obs)
                log_prob = dist.log_prob(actions)
                # optional entropy bonus keeps the policy from collapsing early
                entropy = dist.entropy()
                loss = -log_prob.mean() - args.entropy_coef * entropy.mean()
                pred = dist.mode()
            else:
                pred = policy.forward(obs, deterministic=True)
                loss = th.nn.functional.mse_loss(pred, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                # rmse between the policy's mean action and the human's, in
                # normalized action units ([-1, 1])
                rmse = th.sqrt(((pred - actions) ** 2).mean()).item()
                if stochastic:
                    print(f"[bc] step {step:7d}/{args.steps}  "
                          f"nll={-log_prob.mean().item():.4f}  "
                          f"entropy={entropy.mean().item():.4f}  rmse={rmse:.4f}")
                else:
                    print(f"[bc] step {step:7d}/{args.steps}  "
                          f"mse={loss.item():.4f}  rmse={rmse:.4f}")
    except KeyboardInterrupt:
        out = _save(model, _out_path(args, step))
        print(f"[bc] interrupted at step {step}; resume with "
              f"--init {out} --steps <remaining>")
        return
    _save(model, _out_path(args))


def _out_path(args, step=None) -> Path:
    """Model output path; the default name tags the algo and (partial) steps."""
    if args.out:
        return Path(args.out)
    total = step if step is not None else args.steps
    return Path(f"models/bc/{args.algo}/{resolve_dataset(args.data).name}"
                f"_steps{total}.zip")


def _save(model, out) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"saved model -> {out}")
    return out


def cmd_offline(args) -> None:
    args.steps = args.steps or DEFAULT_STEPS[args.algo]
    use_beams = not args.no_beams
    dataset, _ = load_training_dataset(args)
    # SB3 behavior cloning: the observation is beams (the default) or the raw
    # BEV (--no-beams)
    buffer = build_replay_buffer(dataset, use_beams=use_beams, device=args.device)
    obs_space = beams_space(dataset.observation_space, use_beams)
    train_bc(args, buffer, obs_space, dataset.action_space)
