import json
import time
from typing import List, Dict, Any

class EpisodeRecorder:
    def __init__(self):
        self.episodes = []

    def start_episode(self, task: str, seed: int):
        self.current_episode = {
            "task": task,
            "seed": seed,
            "steps": [],
            "start_time": time.time()
        }

    def record_step(self, step: int, action: str, state: Dict, reward: float, done: bool):
        self.current_episode["steps"].append({
            "step": step,
            "action": action,
            "state": state,
            "reward": reward,
            "done": done
        })

    def end_episode(self, final_survival: float, total_reward: float):
        self.current_episode["end_time"] = time.time()
        self.current_episode["duration"] = self.current_episode["end_time"] - self.current_episode["start_time"]
        self.current_episode["final_survival"] = final_survival
        self.current_episode["total_reward"] = total_reward
        self.episodes.append(self.current_episode)
        # Optionally save to file
        with open(f"episode_{int(time.time())}.json", "w") as f:
            json.dump(self.current_episode, f, indent=2)
        return self.current_episode