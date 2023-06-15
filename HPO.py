import optuna

from stable_baselines3 import PPO #Algoritmo PPO (Proximal Policy Optimization)

from stable_baselines3.common.evaluation import evaluate_policy
from environment import create_env
from math import floor
import os

# ------------------------------------------------------------------------------------

LOG_DIR = './logs/'
OPT_DIR = './opt/'

# ------------------------------------------------------------------------------------

def objective(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_float('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99)
    }

# ------------------------------------------------------------------------------------

def optimize_agent(trial):
    try:
        model_parameters = objective(trial)
        
        #Set environment
        env = create_env(LOG_DIR=LOG_DIR)
        
        # Algoritmo
        model = PPO('CnnPolicy', env, tensorboard_log = LOG_DIR, verbose = 0, **model_parameters)
        model.learn(total_timesteps=100000)
        
        #Evaluate
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes = 100)
        env.close()
        
        SAVE_PATH = os.path.join(OPT_DIR, f'trial_{trial.number}_model')
        model.save(SAVE_PATH)
        
        return mean_reward
        
    except Exception:
        return -10

# ------------------------------------------------------------------------------------

def main():
    print('\n\n... Finding Hyper Parameters ...\n\n')

    study = optuna.create_study(direction = 'maximize')
    study.optimize(optimize_agent, n_trials=10, n_jobs=1)

    # Create environment 
    env = create_env(LOG_DIR= LOG_DIR)
    
    print('\n\nDone!\n\n... Creating the model ...\n\n')

    model_params = study.best_params
    model_params['n_steps'] = floor(model_params['n_steps']/64) * 64 # to not have to truncate the n_steps param every time
    model_params['learning_rate'] = float('%.2g' % (model_params['learning_rate']))

    model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=0, **model_params)
    model.save(os.path.join(OPT_DIR, f'best_model'))

# ------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()