import sys
import os
import time

import optuna
from environment import create_env
from callback import Callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from math import floor

from HPO import optimize_agent

# ------------------------------------------------------------------------------------

CHECKPOINT_DIR = './train/'
OPT_DIR = './opt/'
LOG_DIR = './logs/'

# ------------------------------------------------------------------------------------

def find_parameters(n_trials, n_jobs):
    study = optuna.create_study(direction = 'maximize')
    study.optimize(optimize_agent, n_trials=n_trials, n_jobs=n_jobs)

    return study

def main():
    print('... Starting training ...\n')

    model = PPO.load(os.path.join(OPT_DIR, f'best_model'))

    #creating callback instance
    callback = Callbacks(check_freq=100000, save_path=CHECKPOINT_DIR)

    model.load(os.path.join(OPT_DIR,f'best_model.zip')) # to initialize with the best model weights
    model.learn(total_timesteps=5500000, callback=callback)

    print('\nDone!\n\nEvaluating the model...\n')

    #model  = PPO.load(os.path.join(OPT_DIR, f'model_5500000'))

    env = create_env(LOG_DIR=LOG_DIR)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=True)

    print(f'\n\n\nThe mean_reward is: {mean_reward}\n\n\n')

    env.close()

    print('\nDone!\n\nTesting the model...\n')

    for episode in range(1): 
        obs = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            time.sleep(0.01)
            total_reward += reward
        print(f'Total Reward for episode {episode} is {total_reward}')
        time.sleep(2)

# ------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------------