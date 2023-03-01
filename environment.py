import retro
from gym import Env
from gym.spaces import Box, MultiBinary
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import numpy as np
import cv2

class MortalKombat(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = env = retro.make("MortalKombat-Genesis", inttype=retro.data.Integrations.ALL)
    
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        # Shape reward
        reward = (self.enemy_health - info['enemy_health'])*2 - (self.health - info['health'])

        return obs, reward, done, info 
    
    def render(self, *args, **kwargs): 
        self.game.render()

    def reset(self):
        self.previous_frame = np.zeros(self.game.observation_space.shape)
        
        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.health = 166
        self.enemy_health = 166

        return obs
    
    def preprocess(self,observation):
        crop = observation[50:224, 0:320]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84,84), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (84,84,1))
        return state
    
    def close(self): 
        self.game.close()

def create_env(LOG_DIR):
    env = MortalKombat()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    return env