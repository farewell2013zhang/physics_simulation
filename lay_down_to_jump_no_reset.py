import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3.common.utils import LinearSchedule, get_latest_run_id, safe_mean
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from collections import deque
import random
import heapq
import os
import mujoco
import numpy as np
import bisect
import time
import torch

# -------- Env wrapper ----------
class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        model = self.env.unwrapped.model
        model.actuator_ctrlrange[:, 0] = -1
        model.actuator_ctrlrange[:, 1] =  1
        # self.env.unwrapped.observation_space = Box(
        #     low=-np.inf, high=np.inf, shape=(self.env.unwrapped.observation_space.shape[0]*2,), dtype=np.float64
        # )
        self.action_space = self.env.unwrapped._set_action_space()
        self.dt = self.env.unwrapped.model.opt.timestep * self.env.unwrapped.frame_skip
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        self.x_hist = []
        self.z_hist = []
        self.z_target = 1
        self.z_sum = 0
        self.z_cnt = 0

    def step(self, action):
        self.index += 1
        obs, reward, done, terminated, info = self.env.step(action)
        reward += 5 * (1 - (2.5 - min(obs[0], 2.5)) ** 2 / 2.5 ** 2)
        if len(self.z_hist) >= 2:
            reward += 100 * ((obs[0] - self.z_hist[-1]) - (self.z_hist[-1] - self.z_hist[-2]) +
                           0.05 * (self.z_hist[-2] > 1))
        # z_all = self.z_sum/self.z_cnt
        # if obs[0] > z_all and obs[0] < 1.4:
        #     reward += 100
        self.reward += reward
        # x = self.data.qpos[0]
        # dx = x - self.x_hist[0]
        # self.x_hist.append(x)
        self.z_hist.append(obs[0])
        self.z_sum += obs[0]
        self.z_cnt += 1
        # if len(self.x_hist) > 0.5 / self.dt:
        #     self.x_hist = self.x_hist[1:]
        if len(self.z_hist) > 2.5e4:
            self.z_sum -= self.z_hist[0]
            self.z_cnt -= 1
            self.z_hist = self.z_hist[1:]
        # if dx > 0:
        #     reward += dx / len(self.x_hist) * ( 0.5 / self.dt)
        # if obs[0] < self.z_max - (self.z_max - self.z_init) * 0.32 or \
        #     self.index - self.max_index > 10:
        #     done = True
        #     info["max_height"] = self.z_max
        if self.index - self.max_index > 10:
            # done = True
            # info["episode"] = {"max_height": self.z_max, "l": self.index, "r": self.reward}
            # while abs(self._last_obs[0]-obs[0]) > 0.005:
            #     self._last_obs = obs.copy()
            #     obs, _, _, _, _ = self.env.step(np.zeros_like(action))
            self.z_max = obs[0]
            self.z_init = obs[0]
            self.obs_init = obs.copy()
            self.index = 0
            self.max_index = 0
            self.reward = 0
        else:
            if obs[0] > self.z_max:
                self.z_max = min(1.4, obs[0])
                self.max_index = self.index
            if obs[0] > 1:
                self.max_index = self.index
        self._last_obs = obs.copy()
        return obs, reward, done, terminated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        model = self.model
        data = self.data
        qpos = data.qpos.copy()
        # Reset standing
        # qpos[2] -= np.min(data.xipos[1:,2]) - 0.1
        # Reset laying on the ground
        qpos[2] = 0.15
        qpos[3:7] = [0.707, 0, -0.707, 0]  # quaternion
        qpos[7:] *= 0.5

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
        self.z_hist.append(obs[0])
        self.z_sum += obs[0]
        self.z_cnt += 1
        self.z_init = obs[0]
        self.obs_init = obs.copy()
        self.z_max = obs[0]
        self.index = 0
        self.max_index = 0
        self.reward = 0
        return obs, info
    
class ReplayBufferC1(ReplayBuffer):
    def __init__(self,
        buffer_size,
        observation_space,
        action_space,
        device = "auto",
        n_envs = 1,
        optimize_memory_usage = False,
        handle_timeout_termination = True,
        n_steps: int = 3, 
        gamma: float = 0.99
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, 
                         optimize_memory_usage, handle_timeout_termination)
        self.time = []
        self.n_steps = n_steps
        self.gamma = gamma
        self.height_max = []
        self.observations2 = np.zeros((self.buffer_size//100*self.n_envs, 100, *self.obs_shape), dtype=observation_space.dtype)
        self.next_observations2 = np.zeros((self.buffer_size//100*self.n_envs, 100, *self.obs_shape), dtype=observation_space.dtype)
        self.actions2 = np.zeros(
            (self.buffer_size//100*self.n_envs, 100, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards2 = np.zeros((self.buffer_size//100*self.n_envs, 100), dtype=np.float32)
        self.dones2 = np.zeros((self.buffer_size//100*self.n_envs, 100), dtype=np.float32)
        self.timeouts2 = np.zeros((self.buffer_size//100*self.n_envs, 100), dtype=np.float32)
        self.pos2_tmp = 0
        self.pos2 = 0
        self.full2 = False
        self.idx_list = deque([])
        self.warmup = 50 * np.ones((self.n_envs,))
        self.height_max2 = []
        self.observations3 = np.zeros((self.buffer_size//50*self.n_envs, 50, *self.obs_shape), dtype=observation_space.dtype)
        self.next_observations3 = np.zeros((self.buffer_size//50*self.n_envs, 50, *self.obs_shape), dtype=observation_space.dtype)
        self.actions3 = np.zeros(
            (self.buffer_size//50*self.n_envs, 50, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )
        self.rewards3 = np.zeros((self.buffer_size//50*self.n_envs, 50), dtype=np.float32)
        self.dones3 = np.zeros((self.buffer_size//50*self.n_envs, 50), dtype=np.float32)
        self.timeouts3 = np.zeros((self.buffer_size//50*self.n_envs, 50), dtype=np.float32)
        self.pos3_tmp = 0
        self.pos3 = 0
        self.full3 = False
        self.idx_list2 = deque([])
        self.warmup2 = 49 * np.ones((self.n_envs,))
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos,
    ) -> None:

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        while len(self.idx_list) and (self.pos - self.idx_list[0][1]) % self.buffer_size == 49:
            a, _, c = self.idx_list.popleft()
            if self.pos >= 99:
                score = self.next_observations[self.pos-99:self.pos+1, c, 0].max()
            else:
                score = max(self.next_observations[self.pos-99+self.buffer_size:, c, 0].max(),
                                self.next_observations[:self.pos+1, c, 0].max())
            new_ele_flag = False
            if len(self.height_max) < self.buffer_size//100:
                heapq.heappush(self.height_max, (score, a))
                new_ele_flag = True
            elif score >= self.height_max[0][1]:
                _, a = heapq.heapreplace(self.height_max, (score, self.height_max[0][1]))
                new_ele_flag = True
            if new_ele_flag:
                indices = (self.pos - 99 + np.arange(100)) % self.buffer_size
                self.observations2[a] = self.observations[indices, c]
                self.next_observations2[a] = self.next_observations[indices, c]
                self.actions2[a] = self.actions[indices, c]
                self.rewards2[a] = self.rewards[indices, c]
                self.pos2 += 1
                if not self.full2 and self.pos2 == self.buffer_size//100:
                    self.pos2 = 0
                    self.full2 = True
        for ienv in range(self.n_envs):
            if self.warmup[ienv] == 0:
                score = next_obs[ienv, 0]
                if self.pos2_tmp < self.buffer_size//100:
                    self.idx_list.append([self.pos2_tmp, self.pos, ienv])
                    self.pos2_tmp += 1
                    self.warmup[ienv] = 100
                else:
                    if score >= self.height_max[0][0]:
                        self.idx_list.append([self.height_max[0][1], self.pos, ienv])
                        self.warmup[ienv] = 100
            else:
                self.warmup[ienv] -= 1
        
        for ienv in range(self.n_envs):
            if self.warmup2[ienv] == 0:
                pre_pos = (self.pos-1) % self.buffer_size
                score = (next_obs[ienv, 0] - self.next_observations[pre_pos, ienv, 0]) - (obs[ienv, 0] - self.observations[pre_pos, ienv, 0]) + 1 * (self.observations[pre_pos, ienv, 0] > 1)
                if self.pos3_tmp < self.buffer_size//50:
                    self.idx_list2.append([self.pos3_tmp, self.pos, ienv, score])
                    self.pos3_tmp += 1
                    self.warmup2[ienv] = 50
                else:
                    if score >= self.height_max2[0][0]:
                        self.idx_list2.append([self.height_max2[0][1], self.pos, ienv, score])
                        self.warmup2[ienv] = 50
            else:
                self.warmup2[ienv] -= 1
        while len(self.idx_list2) and (self.pos - self.idx_list2[0][1]) % self.buffer_size == 0:
            a, _, c, d = self.idx_list2.popleft()
            score = d
            new_ele_flag2 = False
            if len(self.height_max2) < self.buffer_size//50:
                heapq.heappush(self.height_max2, (score, a))
                new_ele_flag2 = True
            elif score >= self.height_max2[0][1]:
                _, a = heapq.heapreplace(self.height_max2, (score, self.height_max2[0][1]))
                new_ele_flag2 = True
            if new_ele_flag2:
                indices = (self.pos - 49 + np.arange(50)) % self.buffer_size
                self.observations3[a] = self.observations[indices, c]
                self.next_observations3[a] = self.next_observations[indices, c]
                self.actions3[a] = self.actions[indices, c]
                self.rewards3[a] = self.rewards[indices, c]
                self.pos3 += 1
                if not self.full3 and self.pos3 == self.buffer_size//50:
                    self.pos3 = 0
                    self.full3 = True

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env = None):
        
        if not self.optimize_memory_usage:
            if self.pos2 > 0 and self.pos3 > 0:
                env_indices = np.random.randint(0, high=self.n_envs, size=(batch_size//3,))
                batch_inds = np.random.randint(0, high=self.pos if not self.full else self.buffer_size, size=(batch_size//3,))
                env_indices2 = np.random.randint(0, high=100, size=(batch_size//3,))
                batch_inds2 = np.random.randint(0, high=self.pos2 if not self.full2 else self.buffer_size//100, size=(batch_size//3,))
                env_indices3 = np.random.randint(0, high=100, size=(batch_size//3,))
                batch_inds3 = np.random.randint(0, high=self.pos3 if not self.full3 else self.buffer_size//100, size=(batch_size//3,))
            elif self.pos2 > 0:
                env_indices = np.random.randint(0, high=self.n_envs, size=(batch_size//2,))
                batch_inds = np.random.randint(0, high=self.pos if not self.full else self.buffer_size, size=(batch_size//2,))
                env_indices2 = np.random.randint(0, high=100, size=(batch_size//2,))
                batch_inds2 = np.random.randint(0, high=self.pos2 if not self.full2 else self.buffer_size//100, size=(batch_size//2,))
                env_indices3 = np.array([],dtype=int)
                batch_inds3 = np.array([],dtype=int)
            elif self.pos3 > 0:
                env_indices = np.random.randint(0, high=self.n_envs, size=(batch_size//2,))
                batch_inds = np.random.randint(0, high=self.pos if not self.full else self.buffer_size, size=(batch_size//2,))
                env_indices2 = np.array([],dtype=int)
                batch_inds2 = np.array([],dtype=int)
                env_indices3 = np.random.randint(0, high=100, size=(batch_size//2,))
                batch_inds3 = np.random.randint(0, high=self.pos3 if not self.full3 else self.buffer_size//100, size=(batch_size//2,))
            else:
                env_indices = np.random.randint(0, high=self.n_envs, size=(batch_size,))
                batch_inds = np.random.randint(0, high=self.pos if not self.full else self.buffer_size, size=(batch_size,))
                env_indices2 = np.array([],dtype=int)
                batch_inds2 = np.array([],dtype=int)
                env_indices3 = np.array([],dtype=int)
                batch_inds3 = np.array([],dtype=int)
            return self._get_samples(batch_inds, env_indices, batch_inds2, env_indices2, batch_inds3, env_indices3, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds, env_indices, batch_inds2, env_indices2, batch_inds3, env_indices3, env = None):
        t0 = time.time()
        if self.n_steps == 1:
            if self.optimize_memory_usage:
                next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
            else:
                next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)
                next_obs2 = self._normalize_obs(self.next_observations2[batch_inds2, env_indices2, :], env)
                next_obs3 = self._normalize_obs(self.next_observations3[batch_inds3, env_indices3, :], env)

            data = (
                np.concat([self._normalize_obs(self.observations[batch_inds, env_indices, :], env), 
                           self._normalize_obs(self.observations2[batch_inds2, env_indices2, :], env), 
                           self._normalize_obs(self.observations3[batch_inds3, env_indices3, :], env)], axis=0),
                np.concat([self.actions[batch_inds, env_indices, :], self.actions2[batch_inds2, env_indices2, :], self.actions3[batch_inds3, env_indices3, :]], axis=0),
                np.concat([next_obs, next_obs2, next_obs3], axis=0),
                # Only use dones that are not due to timeouts
                # deactivated by default (timeouts is initialized as an array of False)
                np.concat([(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
                           np.zeros_like(env_indices2).reshape(-1, 1), np.zeros_like(env_indices3).reshape(-1, 1)], axis=0),
                np.concat([self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env), 
                           self._normalize_reward(self.rewards2[batch_inds2, env_indices2].reshape(-1, 1), env), 
                           self._normalize_reward(self.rewards3[batch_inds3, env_indices3].reshape(-1, 1), env)], axis=0),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

        # Note: the self.pos index is dangerous (will overlap two different episodes when buffer is full)
        # so we set self.pos-1 to truncated=True (temporarily) if done=False and truncated=False
        last_valid_index = self.pos - 1
        original_timeout_values = self.timeouts[last_valid_index].copy()
        self.timeouts[last_valid_index] = np.logical_or(original_timeout_values, np.logical_not(self.dones[last_valid_index]))

        # Compute n-step indices with wrap-around
        steps = np.arange(self.n_steps).reshape(1, -1)  # shape: [1, n_steps]
        indices = (batch_inds[:, None] + steps) % self.buffer_size  # shape: [batch, n_steps]

        # Retrieve sequences of transitions
        rewards_seq = self._normalize_reward(self.rewards[indices, env_indices[:, None]], env)  # [batch, n_steps]
        dones_seq = self.dones[indices, env_indices[:, None]]  # [batch, n_steps]
        truncated_seq = self.timeouts[indices, env_indices[:, None]]  # [batch, n_steps]

        # Compute masks: 1 until first done/truncation (inclusive)
        done_or_truncated = np.logical_or(dones_seq, truncated_seq)
        done_idx = done_or_truncated.argmax(axis=1)
        # If no done/truncation, keep full sequence
        has_done_or_truncated = done_or_truncated.any(axis=1)
        done_idx = np.where(has_done_or_truncated, done_idx, self.n_steps - 1)

        mask = np.arange(self.n_steps).reshape(1, -1) <= done_idx[:, None]  # shape: [batch, n_steps]
        # Compute discount factors for bootstrapping (using target Q-Value)
        # It is gamma ** n_steps by default but should be adjusted in case of early termination/truncation.
        target_q_discounts = self.gamma ** mask.sum(axis=1, keepdims=True).astype(np.float32)  # [batch, 1]

        # Apply discount
        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32).reshape(1, -1)  # [1, n_steps]
        discounted_rewards = rewards_seq * discounts * mask
        n_step_returns = discounted_rewards.sum(axis=1, keepdims=True)  # [batch, 1]

        # Compute indices of next_obs/done at the final point of the n-step transition
        last_indices = (batch_inds + done_idx) % self.buffer_size
        next_obs = self._normalize_obs(self.next_observations[last_indices, env_indices], env)
        next_dones = self.dones[last_indices, env_indices][:, None].astype(np.float32)
        next_timeouts = self.timeouts[last_indices, env_indices][:, None].astype(np.float32)
        final_dones = next_dones * (1.0 - next_timeouts)

        # Revert back tmp changes to avoid sampling across episodes
        self.timeouts[last_valid_index] = original_timeout_values

        # Gather observations and actions
        obs = self._normalize_obs(self.observations[batch_inds, env_indices], env)
        actions = self.actions[batch_inds, env_indices]

        return ReplayBufferSamples(
            observations=self.to_torch(obs),  # type: ignore[arg-type]
            actions=self.to_torch(actions),
            next_observations=self.to_torch(next_obs),  # type: ignore[arg-type]
            dones=self.to_torch(final_dones),
            rewards=self.to_torch(n_step_returns),
            discounts=self.to_torch(target_q_discounts),
        )

    
class EarlyStopAndSaveBestCallback(BaseCallback):
    def __init__(self, save_path, patience=10, patience2=2, min_delta=0.0, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.patience = patience
        self.patience2 = patience2
        self.min_delta = min_delta
        self.best_value = 0
        self.best_value_all = 0
        self.counter = 0
        self.counter2 = 0
        self.snapshot = []
        self.round = 0
        self.warmup = 0
        self.i_stage = 0

    def _on_step(self):
        # Early stop condition
        if self.counter2 >= self.patience2:
            print("⏹️ Early stopping: value stopped improving.")
            return False  # stops training loop
        return True

    def _on_rollout_end(self):
        # self.round += 1
        # if self.round == 25:
        #     tracemalloc.start()
        # elif self.round % 25 == 0:
        #     self.snapshot.append(tracemalloc.take_snapshot())
        # current_value = safe_mean([ep_info["l"]+ep_info["max_height"] if ep_info["max_height"] > 1 else ep_info["max_height"] for ep_info in self.model.ep_info_buffer])
        # current_value = safe_mean([safe_mean(env.env.z_hist[max(1,len(env.env.z_hist)-4096):]) for env in self.model.env.venv.unwrapped.envs])
        # self.max_height = max(self.max_height, max([max(env.env.z_hist[max(1,len(env.env.z_hist)-4096):]) for env in self.model.env.venv.unwrapped.envs]))
        current_value = safe_mean([safe_mean(env.z_hist[max(0,len(env.z_hist)-25000):]) for env in self.model.env.venv.unwrapped.envs])
        max_height = max([max(env.z_hist[max(0,len(env.z_hist)-25000):]) for env in self.model.env.venv.unwrapped.envs])
        # for env in self.model.env.venv.unwrapped.envs:
        #     env.env.z_target = min(1, current_value)
        self.model.logger.record("rollout/total_rew", current_value)
        self.model.logger.record("rollout/max_height", max_height)

        if self.warmup < 0:
            self.warmup += 1
            if self.verbose:
                print(f"warmup={-self.warmup}/20, value={current_value:.4f}")
            return True

        # Check improvement
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            if self.best_value_all < self.best_value:
                self.best_value_all = self.best_value
                # Save the best model at current stage
                if self.counter > 100:
                    best_model_path = os.path.join(self.save_path, f"best_model_stage{self.i_stage}.zip")
                    self.model.save(best_model_path)
                    if hasattr(self.model.get_env(), "save"):
                        self.model.get_env().save(os.path.join(self.save_path, f"vec_normalize_stage{self.i_stage}.pkl"))
                    np.save(os.path.join(self.save_path, f"obs_init_stage{self.i_stage}.npy"), self.model.env.venv.unwrapped.envs[0].obs_init)
                    self.i_stage += 1
                # Save the best model
                best_model_path = os.path.join(self.save_path, "best_model.zip")
                self.model.save(best_model_path)
                if hasattr(self.model.get_env(), "save"):
                    self.model.get_env().save(os.path.join(self.save_path, "vec_normalize.pkl"))
                np.save(os.path.join(self.save_path, "obs_init.npy"), self.model.env.venv.unwrapped.envs[0].obs_init)
                if self.verbose > 0:
                    print(f"✅ Saved new best model, value={current_value:.4f}")
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.counter2 += 1
                self.counter = 0
                self.best_value = 0
                self.warmup = -20
                if self.model.ent_coef_optimizer is not None and self.model.log_ent_coef is not None:
                    self.model.log_ent_coef = torch.log(torch.ones(1, device=self.model.device) * 0.005).requires_grad_(True)
                    self.model.ent_coef_optimizer = torch.optim.Adam([self.model.log_ent_coef], lr=self.model.lr_schedule(1))
                else:
                    self.model.ent_coef_tensor = torch.tensor(float(0.005), device=self.model.device)
                print(f"Reset patience and best value!!")
            if self.verbose > 0:
                print(f"patience={self.counter},{self.counter2}/{self.patience},{self.patience2}, value={current_value:.4f}")

        return True

# policy_kwargs = dict(
#     net_arch = dict(pi=[256, 256], vf=[256, 256])
# )
policy_kwargs = dict(
    net_arch = dict(pi=[256, 256], qf=[256, 256])
)

def make_env(render_mode=None):
    env = gym.make("Humanoid-v5", max_episode_steps=int(1e10), frame_skip=5,
                    contact_cost_weight=0, forward_reward_weight=0,
                    ctrl_cost_weight=0.1, healthy_reward=0,
                    healthy_z_range=(-1e10, 1e10),
                    render_mode=render_mode)
    return EnvWrapper(env)
    return Monitor(EnvWrapper(env), info_keywords=("max_height", ))

def main():
    save_model_path = "./humanoid_log/Jump/Lay Down Position/MlpPolicy"
    latest_run_id = get_latest_run_id(save_model_path, 'SAC')
    save_path = os.path.join(save_model_path, f"SAC_{latest_run_id + 1}")
    env = DummyVecEnv([lambda: make_env() for _ in range(1)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.995)

    callback = EarlyStopAndSaveBestCallback(
        save_path=save_path,
        patience=500,
        patience2=2,
        min_delta=0,
        verbose=1
    )

    # model = PPO("MlpPolicy", env, learning_rate=LinearSchedule(3e-5, 3e-5, 0.1), clip_range=0.2, n_epochs=5, vf_coef=1.5,
    #             n_steps=4096, batch_size=256, gamma=0.995, ent_coef=0.005, policy_kwargs=policy_kwargs, verbose=1, 
    #             target_kl=0.1, tensorboard_log=save_model_path)
    model = SAC("MlpPolicy", env, learning_rate=LinearSchedule(3e-5, 3e-5, 0.1), buffer_size=int(4e5), replay_buffer_class=ReplayBufferC1,
                n_steps=1, batch_size=6144, gamma=0.995, ent_coef="auto_0.005", target_entropy=-8.5, policy_kwargs=policy_kwargs, verbose=1, 
                train_freq=4096, gradient_steps=80, tensorboard_log=save_model_path)

    # 3. Train
    model.learn(total_timesteps=100_000_000, log_interval=20, callback=callback)

def eval():
    env = DummyVecEnv([lambda: make_env(render_mode="human") for _ in range(1)])
    env = VecNormalize.load("./humanoid_log/Jump/Lay Down Position/MlpPolicy/SAC_1/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    obs_init = np.load("./humanoid_log/Jump/Lay Down Position/MlpPolicy/SAC_1/obs_init.npy")
    obs = env.reset()
    env.venv.unwrapped.envs[0].data.qpos[2:] = obs_init[:22]
    env.venv.unwrapped.envs[0].data.qvel[:] = 0
    obs = obs_init.reshape(1,-1)
    mujoco.mj_forward(env.venv.unwrapped.envs[0].model, env.venv.unwrapped.envs[0].data)

    # Load pretrained model
    model = SAC.load("./humanoid_log/Jump/Lay Down Position/MlpPolicy/SAC_1/best_model.zip", env=env)
    
    dt_per_step = env.get_attr('model')[0].opt.timestep * env.get_attr('frame_skip')[0]

    t = 0
    z_hist = []
    for i in range(1000000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        z_hist.append(env.unnormalize_obs(obs)[0,0])
        if len(z_hist) > 4096:
            z_hist = z_hist[1:]
        if i % 4096 == 0:
            a = sum(z_hist)/len(z_hist)
            print(f"average height: {a}")
        t += 1
        env.render()
        if done[0]:
            print(f"time lasts: {dt_per_step * t}")
            t = 0
            obs = env.reset()
    env.close()

def save_gif():
    import imageio
    model_names = sorted([f for f in os.listdir("./humanoid_log/Jump/Lay Down Position/MlpPolicy/SAC_1") if f.startswith("best_model") and f.endswith(".zip")])
    for model_name in model_names:
        suffix = model_name[10:-4]
        if suffix == "":
            final = "final"
        else:
            final = suffix[1:]
        env = DummyVecEnv([lambda: make_env(render_mode="rgb_array") for _ in range(1)])
        env = VecNormalize.load(f"./humanoid_log/Jump/Lay Down Position/MlpPolicy/SAC_1/vec_normalize{suffix}.pkl", env)
        env.training = False
        env.norm_reward = False
        obs_init = np.load(f"./humanoid_log/Jump/Lay Down Position/MlpPolicy/SAC_1/obs_init{suffix}.npy")
        obs = env.reset()
        env.venv.unwrapped.envs[0].data.qpos[2:] = obs_init[:22]
        env.venv.unwrapped.envs[0].data.qvel[:] = 0
        obs = obs_init.reshape(1,-1)
        mujoco.mj_forward(env.venv.unwrapped.envs[0].model, env.venv.unwrapped.envs[0].data)

        # Load pretrained model
        model = SAC.load(f"./humanoid_log/Jump/Lay Down Position/MlpPolicy/SAC_1/{model_name}", env=env)
        
        dt_per_step = env.get_attr('model')[0].opt.timestep * env.get_attr('frame_skip')[0]

        t = 0
        z_hist = []
        frames = []
        for i in range(1000):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            z_hist.append(env.unnormalize_obs(obs)[0,0])
            if len(z_hist) > 4096:
                z_hist = z_hist[1:]
            t += 1
            frames.append(env.render())
            if done[0]:
                print(f"time lasts: {dt_per_step * t}")
                t = 0
                obs = env.reset()
        a = sum(z_hist)/len(z_hist)
        print(f"average height: {a}")
        imageio.mimsave(f"./humanoid_log/Jump/Lay Down Position/MlpPolicy/SAC_1/{final}.gif", frames, fps=30)
        env.close()

if __name__ == "__main__":
    # main()
    # eval()
    save_gif()