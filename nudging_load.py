from stable_baselines3 import PPO
from nudging_env import NudgingEnv

env = NudgingEnv()
env.reset()
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# model = PPO.load('models/1668908290/1500000.zip', env = env)
# model = PPO.load('models/1668883124/6100000.zip', env = env)
model = PPO.load('models/1669314790/520000.zip', env = env)
i = 0
for ep in range(1):
    print(f'\n\n\n\nHELLO ep number {ep}\n\n\n')
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)