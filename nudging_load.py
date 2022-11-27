from stable_baselines3 import PPO
from nudging_env import NudgingEnv

env = NudgingEnv()
env.reset()
model = PPO.load('models/1669330674/2570000.zip', env = env)
i = 0
for ep in range(1):
    print(f'\n\n\n\nHELLO ep number {ep}\n\n\n')
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)