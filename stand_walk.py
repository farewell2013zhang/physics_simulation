import gymnasium as gym
from stable_baselines3.common.utils import LinearSchedule, get_latest_run_id, safe_mean
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import os
import mujoco
import numpy as np

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
        self._last_obs = obs.copy()
        return obs, reward, done, terminated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        model = self.model
        data = self.data
        qpos = data.qpos.copy()
        # Reset the bottom part of body to touch ground
        qpos[2] -= np.min(data.xipos[1:,2]) - 0.1
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
    
class EarlyStopAndSaveBestCallback(BaseCallback):
    def __init__(self, save_path, patience=10, min_delta=0.0, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = 0
        self.counter = 0

    def _on_step(self):
        # Early stop condition
        if self.counter >= self.patience:
            print("⏹️ Early stopping: value stopped improving.")
            return False  # stops training loop
        return True

    def _on_rollout_end(self):
        current_value = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

        # Check improvement
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.counter = 0

            # Save the best model
            best_model_path = os.path.join(self.save_path, "best_model.zip")
            self.model.save(best_model_path)
            if hasattr(self.model.get_env(), "save"):
                self.model.get_env().save(os.path.join(self.save_path, "vec_normalize.pkl"))
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
    env = gym.make("Humanoid-v5", max_episode_steps=100000, frame_skip=5,
                    contact_cost_weight=0, forward_reward_weight=1,
                    ctrl_cost_weight=0.1, healthy_reward=5,
                    healthy_z_range=(1, 1e10),
                    render_mode=render_mode)
    return Monitor(EnvWrapper(env))

def main():
    save_model_path = "./humanoid_log/Stand_Walk/MlpPolicy"
    latest_run_id = get_latest_run_id(save_model_path, 'PPO')
    save_path = os.path.join(save_model_path, f"PPO_{latest_run_id + 1}")
    env = DummyVecEnv([lambda: make_env() for _ in range(7)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    callback = EarlyStopAndSaveBestCallback(
        save_path=save_path,
        patience=10,
        min_delta=0.1,
        verbose=0
    )

    model = PPO("MlpPolicy", env, learning_rate=LinearSchedule(3e-5, 1e-5, 0.1), clip_range=0.2, n_epochs=5, vf_coef=1.5,
                n_steps=4096, batch_size=256, gamma=0.995, ent_coef=0.005, policy_kwargs=policy_kwargs, verbose=1, 
                target_kl=0.1, tensorboard_log=save_model_path)

    # 3. Train
    model.learn(total_timesteps=100_000_000, log_interval=1, callback=callback)

def eval():
    env = DummyVecEnv([lambda: make_env(render_mode="human") for _ in range(1)])
    env = VecNormalize.load("./humanoid_log/Stand_Walk/MlpPolicy/PPO_1/vec_normalize.pkl", env)
    env.training = False
    env.norm_reward = False
    obs = env.reset()

    # Load pretrained model
    model = PPO.load("./humanoid_log/Stand_Walk/MlpPolicy/PPO_1/best_model.zip", env=env)
    
    dt_per_step = env.get_attr('model')[0].opt.timestep * env.get_attr('frame_skip')[0]

    t = 0
    for _ in range(10000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        t += 1
        env.render()
        if done[0]:
            print(f"time lasts: {dt_per_step * t}")
            t = 0
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    # main()
    eval()