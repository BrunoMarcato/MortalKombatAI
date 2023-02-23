import sys
import os
import time

import optuna
from environment import create_env
from callback import Callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from math import floor

from hpTuning import optimize_agent

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
    print('\n\n... Finding Hyper Parameters ...\n\n')
    study = find_parameters(n_trials = 10, n_jobs = 1)

    print('\n\n... Creating environment ...', end='\n\n')

    # Create environment 
    env = create_env(LOG_DIR= LOG_DIR)
    
    print('\n\nDone!\n\n... Creating the model ...\n\n')

    model_params = study.best_params
    model_params['n_steps'] = floor(model_params['n_steps']/64) * 64 # to not have to truncate the n_steps param every time
    model_params['learning_rate'] = 5e-7

    best_trial = study.best_trial

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)

    print('\nDone!\n\n... Starting training ...\n')

    #creating callback instance
    callback = Callbacks(check_freq=10000, save_path=CHECKPOINT_DIR)

    model.load(os.path.join(OPT_DIR,f'trial_{best_trial.number}_best_model.zip')) # to initialize with the best model weights
    model.learn(total_timesteps=100000, callback=callback)

    # print('\nDone!\n\nEvaluating the model...\n')

    # model = PPO.load('./train/best_model_90000')

    # env = create_env(LOG_DIR=LOG_DIR)

    # mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=True)

    # print(f'\n\n\nThe mean_reward is: {mean_reward}\n\n\n')

    # env.close()

    # print('\nDone!\n\nTesting the model...\n')

    # for episode in range(1): 
    #     obs = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done: 
    #         action, _ = model.predict(obs)
    #         obs, reward, done, info = env.step(action)
    #         env.render()
    #         time.sleep(0.01)
    #         total_reward += reward
    #     print(f'Total Reward for episode {episode} is {total_reward}')
    #     time.sleep(2)

# ------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------------