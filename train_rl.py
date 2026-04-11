#!/usr/bin/env python3
"""
Train a PPO agent on the ICU environment.
After training, the model is saved to `models/ppo_icu.zip`.
"""
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os

# Import your environment wrapper (converts your env to Gymnasium API)
from app.environment import ICUEnvironment

class ICUEnvWrapper(gym.Env):
    """
    Wrapper to make your ICUEnvironment compatible with Gymnasium.
    """
    def __init__(self, max_steps=30, task="hard"):
        super().__init__()
        self.env = ICUEnvironment(seed=None, max_steps=max_steps)
        self.task = task
        # Define observation space (flattened state)
        # We'll use a Dict space or flatten. For simplicity, flatten.
        # State includes: vitals (6) + labs (3) + resources (2) + qSOFA (1) + survival (1) + step (1) + alive (1)
        # But we only need the clinical vars for the agent.
        self.observation_space = gym.spaces.Box(
            low=-float('inf'), high=float('inf'), shape=(14,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)  # 0: administer_drug, 1: adjust_ventilator, 2: request_lab, 3: escalate_care

    def _get_obs(self, state_dict):
        s = state_dict
        v = s["vitals"]
        l = s["labs"]
        r = s["resources"]
        # Flatten: HR, BP, SpO2, Temp, RR, GCS, lactate, pH, WBC, icu_beds, ventilators, fio2, qSOFA, survival_prob
        obs = np.array([
            v["HR"], v["BP"], v["SpO2"], v["Temp"], v["RR"], v["GCS"],
            l["lactate"], l["pH"], l["WBC"],
            r["icu_beds"], r["ventilators"], r["fio2"],
            s["qSOFA"], s["survival_probability"]
        ], dtype=np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.env.state_manager.rng = np.random.RandomState(seed)
        data = self.env.reset(task=self.task)
        self.state_dict = data["state"]
        return self._get_obs(self.state_dict), {}

    def step(self, action):
        action_map = {0: "administer_drug", 1: "adjust_ventilator", 2: "request_lab", 3: "escalate_care"}
        act_str = action_map[action]
        result = self.env.step(act_str)
        self.state_dict = result["state"]
        reward = result["reward"]
        done = result["done"]
        truncated = self.state_dict["step"] >= self.state_dict["max_steps"]
        return self._get_obs(self.state_dict), reward, done, truncated, {}

if __name__ == "__main__":
    import numpy as np

    # Create vectorized environment for parallel training (optional)
    env = ICUEnvWrapper(task="hard")  # train on hardest task
    # For better training, use vectorized:
    # env = make_vec_env(lambda: ICUEnvWrapper(task="hard"), n_envs=4)

    # Stop training if mean reward > 0.8 (optional)
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.85, verbose=1)
    eval_callback = EvalCallback(env, best_model_save_path="./models/",
                                 log_path="./logs/", eval_freq=500,
                                 deterministic=True, render=False,
                                 callback_after_eval=callback_on_best)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tb_logs/",
                learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)

    model.learn(total_timesteps=200000, callback=eval_callback)
    model.save("models/ppo_icu_final.zip")
    print("Training complete. Model saved to models/ppo_icu_final.zip")