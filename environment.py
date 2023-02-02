import retro
import cv2 as cv
import numpy as np
from gym import Env
from gym.spaces import Box, MultiBinary

class MortalKombat(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        
        #instancia do jogo
        self.game = retro.make(game='MortalKombat-Genesis', 
                              use_restricted_actions=retro.Actions.FILTERED)
        
        
    def step(self, action):
        observation, reward, done, info = self.game.step(action)
        observation = self.preprocess(observation)
        
        #frame delta
        frame_change = observation - self.previous_frame
        self.previous_frame = observation
        
        #reward function
        reward = info['score']  - self.score
        self.score  = info['score']
        
        return frame_change, reward, done, info
    
    def reset(self):
        observation = self.game.reset()
        observation = self.preprocess(observation)
        self.previous_frame = observation
        self.score = 0
        return observation
        
    def render(self):
        self.game.render()
        
    def close(self):
        self.game.close()
    
    def preprocess(self, observation):
        gray = cv.cvtColor(observation, cv.COLOR_BGR2GRAY)
        resize = cv.resize(gray, (84, 84), interpolation = cv.INTER_CUBIC)
        channels = np.reshape(resize, (84, 84, 1))
        return channels

