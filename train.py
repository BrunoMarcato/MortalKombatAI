import os
import time

from environment import create_env
from callback import Callbacks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

# ------------------------------------------------------------------------------------

CHECKPOINT_DIR = './train/'
OPT_DIR = './opt/'
LOG_DIR = './logs/'

# ------------------------------------------------------------------------------------

def main():
    print('... Training ...\n')

    env = create_env(LOG_DIR=LOG_DIR)

    model = PPO.load(os.path.join(OPT_DIR, f'best_model'))
    model.set_env(env)

    #creating callback instance
    callback = Callbacks(check_freq=100000, save_path=CHECKPOINT_DIR)

    model.learn(total_timesteps=5500000, callback=callback)

    print('\nDone!\n\nEvaluating the model...\n')

    model  = PPO.load(os.path.join(CHECKPOINT_DIR, f'model_100000'))

    env = create_env(LOG_DIR=LOG_DIR)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1, render=True)

    print(f'\n\n\nThe mean_reward is: {mean_reward}\n\n\n')

    env.close()

    print('\nDone!\n\nTesting the model...\n')

    env = create_env(LOG_DIR=LOG_DIR)
    obs = env.reset()
    done = False
    total_reward = 0
    info = {'matches_won': 0}
    while True:
        if info.get('matches_won') == 2:
            pass
        else:
            action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        info = dict(info[0])
        print(info)
        env.render()
        time.sleep(0.01)
        total_reward += reward
        print(reward)

        if done:
            env.reset()

# ------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------------