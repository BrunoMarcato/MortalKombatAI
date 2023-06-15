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
        self.observation_space = Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(game='MortalKombat-Genesis', inttype=retro.data.Integrations.ALL, use_restricted_actions = retro.Actions.FILTERED)
    
    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        # Preprocess frame from game
        frame_delta = obs
#         - self.previous_frame
#         self.previous_frame = obs 
        
        # Shape reward
        reward = info['health'] - info['enemy_health'] + 200*(info['matches_won'] - self.matches_won) - 200*(info['enemy_matches_won'] - self.enemy_matches_won)
        self.matches_won = info['matches_won']
        self.enemy_matches_won = info['enemy_matches_won']

        return frame_delta, reward, done, info 
    
    def render(self, *args, **kwargs): 
        self.game.render()
    
    def reset(self):
        self.previous_frame = np.zeros(self.game.observation_space.shape)
        
        # Frame delta
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        
        self.health = 120
        self.enemy_health = 120
        self.matches_won = 0
        self.enemy_matches_won = 0

        return obs
    
    def preprocess(self, observation): 
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (128,128), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (128,128,1))
        return state
    
    def close(self):
        self.game.close()

def create_env(LOG_DIR):
    env = MortalKombat()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')

    return env