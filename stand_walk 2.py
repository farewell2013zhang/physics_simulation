"""
Dreamer-style minimal pipeline for Humanoid:
1) collect random/weak-policy data into replay
2) train deterministic latent model with multi-step loss
3) run CEM planner in latent space with MPC loop
Notes:
- You MUST check/adjust score_obs_batch() to match your env obs layout.
- Requires: torch, gym, numpy, tqdm
- Recommended: run with a GPU for planner/model speed.
"""

import os
import time
import math
import random
import numpy as np
from collections import deque
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from stable_baselines3.common.utils import LinearSchedule, get_latest_run_id, safe_mean
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import mujoco
from gymnasium.spaces import Box
import math

# -------- Env wrapper ----------
class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        model = self.env.unwrapped.model
        model.actuator_ctrlrange[:, 0] = -0.4
        model.actuator_ctrlrange[:, 1] =  0.4
        self.action_space = self.env.unwrapped._set_action_space()
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        self.x_hist = []

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        x = self.data.qpos[0]
        dx = x - self.x_hist[0]
        self.x_hist.append(x)
        if len(self.x_hist) > 0.5 / self.dt:
            self.x_hist = self.x_hist[1:]
        # if dx > 0:
        #     reward += dx / len(self.x_hist) * ( 0.5 / self.dt)
        # reward += max(self.data.qpos[2] - 1, 0)
        # reward -= sum((self.data.qpos[4:6] - self.init_qpos[4:6]) ** 2)
        self._last_obs = obs.copy()
        return obs, reward, done, terminated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        model = self.model
        data = self.data
        qpos = data.qpos.copy()
        # Reset the bottom part of body to touch ground
        qpos[2] -= np.min(data.xipos[1:,2]) - 0.1
        self.init_qpos = qpos
        qvel = data.qvel.copy()
        # Reset velocities to 0
        qvel[:] = 0.0

        # Apply to simulation
        data.qpos = qpos
        data.qvel = qvel
        mujoco.mj_forward(model, data)

        obs = self.env.unwrapped._get_obs()

        self._last_obs = obs.copy()
        self.x_hist = [qpos[0]]
        return obs, info
    
# ---------------------------
# User: create or return your env here
# ---------------------------
def create_env(healthy_z_range_low=1, exclude_current_positions_from_observation=False, render_mode=None):
    env = gym.make("Humanoid-v5", max_episode_steps=100000, frame_skip=5,
                   exclude_current_positions_from_observation=exclude_current_positions_from_observation,
                   contact_cost_weight=0, forward_reward_weight=1,
                   ctrl_cost_weight=0.1, healthy_reward=5,
                   healthy_z_range=(healthy_z_range_low, 1e10),
                   render_mode=render_mode)
    return Monitor(EnvWrapper(env))

# ---------------------------
# Replay (episode-based, contiguous chunk sampling)
# ---------------------------
class EpisodeReplay:
    def __init__(self, obs_dim, act_dim, max_episodes=2000):
        # store episodes as lists of numpy arrays
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_episodes = max_episodes
        self.episodes = deque(maxlen=max_episodes)  # each item: dict with obs, act, next_obs, done
    def add_episode(self, obs_list, act_list, done_list):
        # store clipped
        self.episodes.append({
            'obs': np.array(obs_list, dtype=np.float32),
            'act': np.array(act_list, dtype=np.float32),
            'done': np.array(done_list, dtype=np.bool_)
        })
    def total_steps(self):
        return sum(ep['obs'].shape[0] for ep in self.episodes)
    def sample_chunk(self, chunk_len, batch_size):
        """
        sample contiguous chunks of length chunk_len from random episodes.
        Ensures chunk fits entirely within one episode (no crossing done boundary).
        """
        obs_batch = np.zeros((batch_size, chunk_len+1, self.obs_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, chunk_len, self.act_dim), dtype=np.float32)
        for i in range(batch_size):
            ep = random.choice(self.episodes)
            L = ep['obs'].shape[0]
            if L <= chunk_len + 1:
                continue
            start = np.random.randint(0, L - chunk_len - 1)
            # accept chunk
            obs_batch[i] = ep['obs'][start:start+chunk_len+1]
            act_batch[i] = ep['act'][start:start+chunk_len]
        return obs_batch, act_batch

    def random_batch(self, batch_size):
        # sample random transitions
        obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        act = np.zeros((batch_size, self.act_dim), dtype=np.float32)
        next_obs = np.zeros((batch_size, self.obs_dim), dtype=np.float32)
        for i in range(batch_size):
            ep = random.choice(self.episodes)
            L = ep['obs'].shape[0]
            idx = np.random.randint(0, L-1)
            obs[i] = ep['obs'][idx]
            act[i] = ep['act'][idx]
            next_obs[i] = ep['obs'][idx+1]
        return obs, act, next_obs

# ---------------------------
# Latent model: Encoder / Dynamics / Decoder (deterministic residual)
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim=64, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
    def forward(self, x):
        return self.net(x)

class Dynamics(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
    def forward(self, z, a):
        # residual update z <- z + f(z,a)
        inp = torch.cat([z, a], dim=-1)
        dz = self.net(inp)
        return z + dz

class Decoder(nn.Module):
    def __init__(self, latent_dim, obs_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, obs_dim)
        )
    def forward(self, z):
        return self.net(z)

class LatentModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=64):
        super().__init__()
        self.enc = Encoder(obs_dim, latent_dim)
        self.dyn = Dynamics(latent_dim, action_dim)
        self.dec = Decoder(latent_dim, obs_dim)
    def encode(self, obs):
        return self.enc(obs)
    def step(self, z, a):
        return self.dyn(z, a)
    def decode(self, z):
        return self.dec(z)

# ---------------------------
# Scoring function for Humanoid
# ---------------------------
def score_obs_batch_humanoid(obs_pre, obs_batch):
    # obs_pre: (obs_dim, )
    # obs_batch: (N, obs_dim)
    score = obs_batch[:, 2].copy()
    score[obs_batch[:, 2] > 1.1] += 5
    score += obs_batch[:, 0] - obs_pre[None, 0]
    return score

# ---------------------------
# Vectorized CEM planner in latent
# ---------------------------
def cem_planner_latent(env, obs_rms, model, model_ppo, obs, horizon, action_low, action_high,
                       n_samples=1024, iterations=6, elite_frac=0.05, device='cpu'):
    """
    model: LatentModel on device
    z0: torch tensor shape (latent_dim,) or (1,latent_dim)
    returns: planned mean action sequence (horizon, action_dim) numpy
    """
    obs[:2] = 0
    obs = np.clip(obs, -100, 100)
    z0 = model.encode(torch.tensor(obs, dtype=torch.float32, device=device)).squeeze(0).detach()
    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)
    # obs_pre = env.unnormalize_obs(obs)
    obs_pre = obs.copy()
    latent_dim = z0.shape[-1]
    action_dim = action_low.shape[0]
    mu = np.zeros((horizon, action_dim), dtype=np.float32)
    std = np.ones_like(mu) * ((action_high - action_low) / 2.0)[None, :]
    n_elite = max(1, int(n_samples * elite_frac))

    # move model to device (assume already)
    for it in range(iterations):
        # batch rollout
        with torch.no_grad():
            # sample N sequences: shape (N, H, A)
            samples = np.random.normal(mu[None,:,:], std[None,:,:], size=(n_samples, horizon, action_dim)).astype(np.float32)
            samples = np.clip(samples, action_low[None,None,:], action_high[None,None,:])
            if model_ppo is not None and it == 0:
                obs = torch.tensor(obs, dtype=torch.float32, device=device).repeat(n_samples, 1)
                samples_t, _ = model_ppo.predict(obs[:, 2:])
                samples_t = np.clip(samples_t, action_low[None,:], action_high[None,:])
                samples[:,0,:] = samples_t
            z_batch = z0.repeat(n_samples, 1).to(device)  # (N, latent)
            scores = np.zeros(n_samples)
            survival = np.ones(n_samples, dtype=np.bool)
            for t in range(horizon):
                a_t = torch.tensor(samples[:,t,:], dtype=torch.float32, device=device)
                z_batch = model.step(z_batch, a_t)
                obs_pred = model.decode(z_batch)  # (N, obs_dim)
                if model_ppo is not None and t < horizon - 1 and it == 0:
                    samples_t, _ = model_ppo.predict(obs_pred[:, 2:])
                    samples_t = np.clip(samples_t, action_low[None,:], action_high[None,:])
                    samples[:,t+1,:] = samples_t
                # obs_np = env.unnormalize_obs(obs_pred.cpu().numpy())
                obs_np = obs_pred.cpu().numpy() 
                obs_np[:, 2:] = obs_np[:, 2:] * np.sqrt(obs_rms.var + 1e-8) + obs_rms.mean
                survival = survival & (obs_np[:, 2]>1)
                scores_new = score_obs_batch_humanoid(obs_pre, obs_np)
                scores_new[~survival] = 0
                scores += scores_new
        elite_idx = np.argsort(scores)[::-1][:n_elite]
        elite = samples[elite_idx]  # (n_elite, H, A)
        mu = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-6
    return mu, scores[elite_idx].mean()  # (H, A)

# ---------------------------
# Training latent model (multi-step)
# ---------------------------
def train_latent_model(replay, obs_dim, act_dim, device='cuda',
                       latent_dim=64, batch_size=128, k_step=8,
                       epochs=20000, lr=1e-4, validate_every=1000, save_dir="checkpoints"):
    model = LatentModel(obs_dim, act_dim, latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    weights = torch.ones([batch_size, obs_dim])
    weights[:, 47:] = 0.1
    weights_eval = torch.ones([batch_size * 16, obs_dim])
    weights_eval[:, 47:] = 0.1

    pbar = trange(epochs)
    for it in pbar:
        # if not enough data, skip
        if len(replay.episodes) == 0 or replay.total_steps() < (batch_size * 2):
            time.sleep(0.1)
            continue
        obs_b, act_b = replay.sample_chunk(chunk_len=k_step, batch_size=batch_size)
        xy = obs_b[:, [0], :2].copy()
        obs_b[:, :, :2] -= xy
        obs_b = np.clip(obs_b, -100, 100)
        # obs_b shape (B, k+1, obs_dim)
        obs0 = torch.tensor(obs_b[:, 0, :], dtype=torch.float32, device=device)
        z0 = model.encode(obs0)
        loss_pred_total = 0.0
        loss_rec_total = 0.0
        z_pred = z0
        obs_rec = model.decode(z_pred)
        loss_rec = F.mse_loss(obs_rec[:, :], obs0, weight=weights)
        loss_rec_total = loss_rec_total + loss_rec
        for t in range(k_step):
            a_t = torch.tensor(act_b[:, t, :], dtype=torch.float32, device=device)
            z_pred = model.step(z_pred, a_t)
            obs_true = torch.tensor(obs_b[:, t+1, :], dtype=torch.float32, device=device)
            z_true = model.encode(obs_true)
            loss_pred = F.mse_loss(z_pred, z_true)
            loss_pred_total = loss_pred_total + loss_pred
            obs_rec = model.decode(z_pred)
            loss_rec = F.mse_loss(obs_rec[:, :], obs_true[:, :], weight=weights)
            loss_rec_total = loss_rec_total + loss_rec
        loss = (loss_pred_total / k_step) * min(0.5, it/epochs) + (loss_rec_total / (k_step + 1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (it+1) % 10 == 0:
            pbar.set_description(f"it={it} loss={loss.item():.6f} pred={((loss_pred_total/k_step).item()):.6f}")
        if (it+1) % validate_every == 0:
            # quick validation: print one-step mse on a random batch
            obs_b, act_b = replay.sample_chunk(chunk_len=k_step, batch_size=batch_size * 16)
            xy = obs_b[:, [0], :2].copy()
            obs_b[:, :, :2] -= xy
            obs_b = np.clip(obs_b, -100, 100)
            with torch.no_grad():
                obs0 = torch.tensor(obs_b[:, 0, :], dtype=torch.float32, device=device)
                z0 = model.encode(obs0)
                loss_rec_total = 0.0
                z_pred = z0
                obs_rec = model.decode(z_pred)
                loss_rec = F.mse_loss(obs_rec[:, :], obs0, weight=weights_eval)
                loss_rec_total = loss_rec_total + loss_rec
                for t in range(k_step):
                    a_t = torch.tensor(act_b[:, t, :], dtype=torch.float32, device=device)
                    z_pred = model.step(z_pred, a_t)
                    obs_true = torch.tensor(obs_b[:, t+1, :], dtype=torch.float32, device=device)
                    obs_rec = model.decode(z_pred)
                    loss_rec = F.mse_loss(obs_rec[:, :], obs_true[:, :], weight=weights_eval)
                    loss_rec_total = loss_rec_total + loss_rec
                loss = loss_rec_total / (k_step + 1)
            print(f"[val] it={it} loss={loss:.6f}")
            # 保存 best model
            if loss < best_val:
                best_val = loss
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "latent_model_best.pt"))
                print(f"Saved best latent model (loss={loss:.6f})")

    model.load_state_dict(torch.load(os.path.join(save_dir, "latent_model_best.pt")))
    return model

# ---------------------------
# Random data collection
# ---------------------------
def collect_random_data(env, replay, obs_rms, model=None, min_steps=100000):
    """
    Collect data using random uniform actions with small scaling.
    Writes entire episodes to replay.
    """
    steps = 0
    ep_count = 0
    print("Starting random data collection...")
    o = env.reset()
    o[:, 2:] = (o[:, 2:] - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
    obs_list = [None] * env.num_envs
    act_list = [None] * env.num_envs
    next_obs_list = [None] * env.num_envs
    done_list = [None] * env.num_envs
    a_low = env.action_space.low
    a_high = env.action_space.high
    while steps < min_steps:
        if model is None:
            # small random actions centered at 0
            a = np.zeros((0, env.action_space.shape[0]), dtype=np.float64)
            for _ in range(env.num_envs):
                a = np.concat([a, np.random.uniform(env.action_space.low, env.action_space.high, size=env.action_space.shape).reshape(1, -1)], axis=0)
        else:
            random_number = np.random.uniform(0, 1, size=1)
            if random_number > 0.4:
                a, _ = model.predict(o[:, 2:])
                a = np.clip(a, a_low[None,:], a_high[None,:])
            else:
                a = np.zeros((0, env.action_space.shape[0]), dtype=np.float64)
                for _ in range(env.num_envs):
                    a = np.concat([a, np.random.uniform(env.action_space.low, env.action_space.high, size=env.action_space.shape).reshape(1, -1)], axis=0)
        o2, r, done, info = env.step(a)
        o2[:, 2:] = (o2[:, 2:] - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
        for i, done_cur in enumerate(done):
            next_obs = o2[i].copy()
            if done_cur:
                next_obs = info[i]["terminal_observation"]
                next_obs[2:] = (next_obs[2:] - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
            if obs_list[i] is not None:
                obs_list[i].append(o[i].copy())
                act_list[i].append(a[i].copy())
                next_obs_list[i].append(next_obs)
                done_list[i].append(done[i])
            else:
                obs_list[i] = [o[i].copy()]
                act_list[i] = [a[i].copy()]
                next_obs_list[i] = [next_obs]
                done_list[i] = [done[i]]
            if done_cur:
                replay.add_episode(obs_list[i]+next_obs_list[i][-1:], act_list[i], done_list[i])
                obs_list[i] = None
                act_list[i] = None
                next_obs_list[i] = None
                done_list[i] = None
                ep_count += 1
        o = o2
        steps += 1
        if steps % 1000 == 0:
            print(f"collected steps={steps}")
    print(f"Random data collection done: collected {steps} steps in {ep_count} episodes.")

# ---------------------------
# MPC run: plan_every steps replanning, online store new episodes into replay
# ---------------------------
def run_mpc(env, model, obs_rms, replay, model_ppo=None, device='cuda', plan_horizon=30, plan_every=8,
            n_samples=1024, iterations=6, elite_frac=0.05, max_steps=200000, eval_every=5000):
    model.to(device)
    model.eval()
    obs = env.reset()
    obs[:, 2:] = (obs[:, 2:] - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
    obs_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]
    a_low = env.action_space.low
    a_high = env.action_space.high
    total_steps = 0
    ep_obs = [None] * env.num_envs
    ep_act = [None] * env.num_envs
    ep_next = [None] * env.num_envs
    ep_done = [None] * env.num_envs
    ep_len = []
    planned_seq = [None] * env.num_envs
    scores = [None] * env.num_envs
    step_in_plan = np.zeros(env.num_envs, dtype=np.int32)
    step_in_env = np.zeros(env.num_envs, dtype=np.int32)
    while total_steps < max_steps:
        a = np.zeros([env.num_envs, action_dim], dtype=np.float64)
        for ienv in range(env.num_envs):
            if planned_seq[ienv] is None or (step_in_plan[ienv] % plan_every == 0):
                planned_seq[ienv], scores[ienv] = cem_planner_latent(env, obs_rms, model, model_ppo, obs[ienv], plan_horizon, a_low, a_high,
                                                n_samples=n_samples, iterations=iterations,
                                                elite_frac=elite_frac, device=device)
                step_in_plan[ienv] = 0
            planned_seq_cur = planned_seq[ienv]
            a[ienv] = planned_seq_cur[step_in_plan[ienv]].copy()
            step_in_plan[ienv] += 1
        next_obs, r, done, info = env.step(a)
        next_obs[:, 2:] = (next_obs[:, 2:] - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
        step_in_env += 1
        total_steps += 1
        for ienv in range(env.num_envs):
            o2 = next_obs[ienv].copy()
            if done[ienv]:
                o2 = info[ienv]["terminal_observation"]
                o2[2:] = (o2[2:] - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8)
            if ep_obs[ienv] is not None:
                ep_obs[ienv].append(obs[ienv].copy())
                ep_act[ienv].append(a[ienv].copy())
                ep_next[ienv].append(o2)
                ep_done[ienv].append(done[ienv])
            else:
                ep_obs[ienv] = [obs[ienv].copy()]
                ep_act[ienv] = [a[ienv].copy()]
                ep_next[ienv] = [o2]
                ep_done[ienv] = [done[ienv]]
            if done[ienv]:
                replay.add_episode(ep_obs[ienv]+ep_next[ienv][-1:], ep_act[ienv], ep_done[ienv])
                ep_obs[ienv], ep_act[ienv], ep_next[ienv], ep_done[ienv] = [], [], [], []
                ep_len.append(step_in_env[ienv])
                planned_seq[ienv] = None
                scores[ienv] = None
                step_in_plan[ienv] = 0
                step_in_env[ienv] = 0
        obs = next_obs
        # -------------------
        # evaluation + save best model
        # -------------------
        if total_steps % eval_every == 0 and len(ep_len):
            # quick eval using heuristic score
            ep_len_mean = np.mean(np.array(ep_len[-20:])) if len(ep_len) >= 20 else np.mean(np.array(ep_len))
            print(f"[Eval] total_steps={total_steps}, mean_score={ep_len_mean:.3f}")
    print("MPC finished running.")

# ---------------------------
# Main: run collection, model training, MPC
# ---------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = DummyVecEnv([lambda: create_env(exclude_current_positions_from_observation=True) for _ in range(1)])
    env_norm = VecNormalize.load("./humanoid_log/Stand_Walk/MlpPolicy/PPO_1/vec_normalize.pkl", env)
    model_ppo = PPO.load("./humanoid_log/Stand_Walk/MlpPolicy/PPO_1/best_model.zip", env=env_norm)
    obs_rms = env_norm.obs_rms
    # inspect obs dim and action dim
    env = DummyVecEnv([lambda: create_env(healthy_z_range_low=0.9) for _ in range(7)])
    sample_obs = env.reset()
    obs_dim = sample_obs.shape[1]
    action_dim = env.action_space.shape[0]
    print("env obs dim:", obs_dim, "action dim:", action_dim)
    # create replay
    replay = EpisodeReplay(obs_dim, action_dim, max_episodes=30000)
    # 1) collect random data
    collect_random_data(env, replay, obs_rms, model_ppo, min_steps=40000)
    # 2) train latent model
    print("Start training latent model...")
    save_model_path = "./humanoid_log/Stand_Walk/Dreamer"
    os.makedirs(save_model_path, exist_ok=True)
    latest_run_id = get_latest_run_id(save_model_path, 'Dreamer')
    save_path = os.path.join(save_model_path, f"Dreamer_{latest_run_id+1}")
    model = train_latent_model(replay, obs_dim, action_dim, device=device,
                               latent_dim=256, batch_size=128, k_step=10,
                               epochs=25000, lr=5e-4, validate_every=1000, save_dir=save_path)
    # 3) run MPC using model (online collect more data)
    print("Start MPC using trained model...")
    env = DummyVecEnv([lambda: create_env(healthy_z_range_low=1) for _ in range(7)])
    run_mpc(env, model, obs_rms, replay, model_ppo, device=device, plan_horizon=10, plan_every=3,
            n_samples=1024, iterations=1, elite_frac=0.05, max_steps=200000, eval_every=1000)

if __name__ == "__main__":
    main()
