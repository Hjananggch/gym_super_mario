import numpy as np
import torch
import torch.nn.functional as F

import gym
import gym_super_mario_bros
from gym.spaces import Box
from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace

from mario_rl.actions import ACTION_SETS
from mario_rl.config import EnvConfig
from mario_rl.utils import reset_env, seed_everything, step_env


class SkipFrame(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int) -> None:
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        observation = None
        for _ in range(self.skip):
            observation, reward, done, info = step_env(self.env, action)
            total_reward += reward
            if done:
                break
        return observation, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        height, width = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=(height, width), dtype=np.uint8)

    def observation(self, observation):
        gray = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
        return gray.astype(np.uint8)


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, shape: int) -> None:
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=self.shape, mode="bilinear", align_corners=False)
        return resized.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8).cpu().numpy()


def build_env(config: EnvConfig) -> gym.Env:
    if config.action_set not in ACTION_SETS:
        raise ValueError(f"Unsupported action set: {config.action_set}")

    env = gym_super_mario_bros.make(config.env_id)
    env = JoypadSpace(env, ACTION_SETS[config.action_set])
    env = SkipFrame(env, skip=config.skip_frames)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=config.resize_shape)
    env = FrameStack(env, num_stack=config.stack_frames)

    seed_everything(config.seed)
    try:
        env.reset(seed=config.seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(config.seed)
        reset_env(env)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(config.seed)

    return env
