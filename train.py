import sys
import os

import optuna
from environment import MortalKombat
from stable_baselines3.common.monitor import Monitor 
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3 import PPO
from math import floor

from hpTuning import optimize_agent

# ------------------------------------------------------------------------------------

CHECKPOINT_DIR = './train/'
OPT_DIR = './opt/'
LOG_DIR = './logs/'

# ------------------------------------------------------------------------------------

def find_parameters():
    study = optuna.create_study(direction = 'maximize')
    study.optimize(optimize_agent, n_trials=10, n_jobs=1)

    return study


def main():
    print('... Finding Parameters ...')
    study = find_parameters()

    print('... Creating environment ...', end='\n\n')

    # Create environment 
    env = MortalKombat()
    env = Monitor(env, LOG_DIR)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    
    print('Done!\n \
        ... Creating the model ...', end='\n\n')

    model_params = study.best_params
    model_params['n_steps'] = floor(model_params['n_steps']/64) * 64 # to not have to truncate the n_steps param every time
    model_params['learning_rate'] = 5e-7

    best_trial = study.best_trial

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)

    print('Done! \
        Starting training...')

    model.load(os.path.join(OPT_DIR,f'trial_{best_trial}_best_model.zip')) # to initialize with the best model weights
    model.learn(total_timesteps=sys.argv[1])

# ------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------------