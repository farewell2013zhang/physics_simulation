import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.buffers import RolloutBuffer
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper, VecMonitor, VecNormalize, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import LinearSchedule, get_latest_run_id, safe_mean, obs_as_tensor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
import os
import random
import mujoco

# -------- Env wrapper ----------
class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        model = self.env.unwrapped.model
        model.actuator_ctrlrange[:, 0] = -1
        model.actuator_ctrlrange[:, 1] =  1
        self.action_space = self.env.unwrapped._set_action_space()
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        self.env.reset()
        self.zmax_hist = [self.data.qpos[2] - (np.min(self.data.xipos[1:,2]) - 0.1)]

    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        reward += obs[0]
        zamx_all = max(self.zmax_hist)
        if obs[0] > zamx_all:
            reward += 100
        self.z_hist.append(obs[0])
        if done or terminated:
            zmax = max(self.z_hist)
            if zmax > max(self.zmax_hist):
                max_index = self.z_hist.index(zmax)
                if max_index >= 1:
                    self.zmax_hist.append(zmax)
        self._last_obs = obs.copy()
        return obs, reward, done, terminated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        model = self.env.unwrapped.model
        data = self.env.unwrapped.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()

        # Reset velocities to zero
        qvel[:] = 0.0

        # Apply to simulation
        data.qpos = qpos
        data.qvel = qvel
        mujoco.mj_forward(model, data)

        qpos = data.qpos.copy()
        # Reset the bottom part of body to touch ground
        qpos[2] -= np.min(data.xipos[1:,2]) - 0.1
        data.qpos = qpos
        mujoco.mj_forward(model, data)

        obs = self.env.unwrapped._get_obs()

        self._last_obs = obs.copy()
        self.z_hist = [obs[0]]
        return obs, info
    

# -------- Callback ----------
class EarlyStopAndSaveBestCallback(BaseCallback):
    def __init__(self, save_path, patience=10, min_delta=0.0, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = 0
        self.counter = 0
        self.zmax = []

    def _on_step(self):
        # Early stop condition
        if self.counter >= self.patience:
            print("⏹️ Early stopping: value stopped improving.")
            return False  # stops training loop

        for env in self.model.get_env().venv.envs:
            self.zmax.extend(env.env.zmax_hist[1:])
        if len(self.zmax):
            self.zmax = sorted(self.zmax)
            max_len = 10
            if len(self.zmax) > max_len:
                len_zmax = len(self.zmax)
                self.zmax = self.zmax[len_zmax-max_len:]
            for env in self.model.get_env().venv.envs:
                env.env.zmax_hist = [self.zmax[0]]

        return True

    def _on_rollout_end(self):
        # current_value = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        if len(self.zmax):
            current_value = self.zmax[-1]
        else:
            current_value = 0
        self.logger.record("rollout/zmax", current_value)
        if current_value > self.best_value + self.min_delta:
            if hasattr(self.model.get_env(), 'save'):
                self.model.get_env().save(os.path.join(self.save_path, "vec_normalize.pkl"))
            self.best_value = current_value
            self.counter = 0
            best_model_path = os.path.join(self.save_path, "best_model.zip")
            self.model.save(best_model_path)
            if self.verbose > 0:
                print(f"✅ Saved new best model, value={current_value:.4f}")
        else:
            self.counter += 1

        if self.verbose > 0:
            print(f"patience={self.counter}/{self.patience}")

        return True

policy_kwargs = dict(
    net_arch = dict(pi=[256, 256], vf=[256, 256])
)

def make_env(render_mode=None):
    env = gym.make("Humanoid-v5", max_episode_steps=240, frame_skip=3,
                    contact_cost_weight=0, forward_reward_weight=0,
                    ctrl_cost_weight=0.02, healthy_reward=1,
                    healthy_z_range=(0.8, 1e10),
                    render_mode=render_mode)
    return Monitor(EnvWrapper(env))

def main():
    save_model_path = "./humanoid_log/Jump/Standing Position/MlpPolicy"
    latest_run_id = get_latest_run_id(save_model_path, 'PPO')
    save_path = os.path.join(save_model_path, f"PPO_{latest_run_id + 1}")

    venv = DummyVecEnv([lambda: make_env() for _ in range(7)])

    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_reward=10.0)

    model = PPO("MlpPolicy", venv, learning_rate=LinearSchedule(3e-5, 1e-5, 1), clip_range=0.2, n_epochs=5, vf_coef=1.5,
                n_steps=4096, batch_size=256, gamma=0.995, ent_coef=0, policy_kwargs=policy_kwargs, verbose=1, 
                target_kl=None, tensorboard_log=save_model_path)

    callback = EarlyStopAndSaveBestCallback(save_path, patience=50, min_delta=0, verbose=0)
    model.learn(total_timesteps=100_000_000, log_interval=1, callback=callback)

def eval():
    venv = DummyVecEnv([lambda: make_env("human") for _ in range(1)])

    venv = VecNormalize.load("./humanoid_log/Jump/Standing Position/MlpPolicy/PPO_1/vec_normalize.pkl", venv)
    venv.training = False
    venv.norm_reward = False
    obs = venv.reset()
    model = PPO.load("./humanoid_log/Jump/Standing Position/MlpPolicy/PPO_1/best_model", env=venv)
    dt_per_step = venv.get_attr('model')[0].opt.timestep * venv.get_attr('frame_skip')[0]

    t = 0
    for _ in range(10000):
        action, _ = model.predict(torch.tensor(obs, dtype=torch.float32))
        obs, reward, done, info = venv.step(action)
        t += 1
        venv.render()
        if done[0]:
            print(f"time lasts: {dt_per_step * t}")
            t = 0
            obs = venv.reset()
    venv.close()

if __name__ == "__main__":
    main()
    # eval()
