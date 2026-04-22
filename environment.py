import gymnasium as gym
import cv2
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env, height=42, width=42, n_frames=4):
        super().__init__(env)
        self.img_size = (height, width)
        self.frames = np.zeros((n_frames, height, width), dtype=np.float32)
        self.observation_space = Box(0.0, 1.0, self.frames.shape)

    def reset(self, **kwargs):
        self.frames = np.zeros_like(self.frames)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, img):
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
        self.frames = np.roll(self.frames, -1, axis=0)
        self.frames[-1] = img
        return self.frames

def make_env():
    env = gym.make("KungFuMasterDeterministic-v0", render_mode="rgb_array")
    return PreprocessAtari(env)
