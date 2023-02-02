import optuna
from stable_baselines3 import PPO #Algoritmo PPO (Proximal Policy Optimization)

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from environment import MortalKombat
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
        
        #Ambiente
        env = MortalKombat()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order = 'last')
        
        # Algoritmo
        model = PPO('CnnPolicy', env, tensorboard_log = LOG_DIR, verbose = 0, **model_parameters)
        model.learn(total_timesteps=30000)
        
        #Evaluate
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes = 5)
        env.close()
        
        SAVE_PATH = os.path.join(OPT_DIR, f'trial_{trial.number}_best_model')
        model.save(SAVE_PATH)
        
        return mean_reward
        
    except Exception as e:
        return -10

# ------------------------------------------------------------------------------------
