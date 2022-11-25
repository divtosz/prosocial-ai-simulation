from nudging_env import NudgingEnv
env = NudgingEnv()
episodes = 1

for episode in range(episodes):
    done = False
    obs = env.reset()                               
    last_step = False
    while not done:
        random_action = env.action_space.sample()
        print("action",random_action)
        obs, reward, done, info = env.step(random_action)
        print('reward',reward)