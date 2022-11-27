from stable_baselines3 import PPO
from nudging_env import NudgingEnv
from copy import deepcopy
import random

TOTAL_RESOURCES_AVAILABLE = 120
TOTAL_RESOURCES_REQUIRED = 100
NUM_COMMUNITIES = 4

# compare random and trained agent
num_of_steps = []
rewards = []
NUM_EPS = 3

# checks that the overall distribution is insufficient
def insufficient(available_resources, required_resources):
    for i in range(NUM_COMMUNITIES):
        if available_resources[i] < required_resources[i]:
            return True
    return False

for episode in range(NUM_EPS):
    while True:
        available_resources = [random.randint(0,TOTAL_RESOURCES_AVAILABLE//4) for i in range(NUM_COMMUNITIES)]
        required_resources = [random.randint(0,TOTAL_RESOURCES_REQUIRED//4) for i in range(NUM_COMMUNITIES)]
        if insufficient(available_resources, required_resources):
            break
    env_random = NudgingEnv(available_resources, required_resources)
    env_learned = NudgingEnv(available_resources, required_resources)
    model = PPO.load('models/1669480859/2690000.zip', env = env_learned)
    num_steps = 0
    curr_episode_steps = []
    curr_episode_reward = []
    # random agent
    obs = env_random.reset()
    done = False
    episode_reward = 0
    while not done:
        random_action = env_random.action_space.sample()
        obs, reward, done, info = env_random.step(random_action)
        episode_reward += reward
        num_steps += 1
    curr_episode_reward.append(episode_reward)
    curr_episode_steps.append(num_steps)

    # trained agent
    obs = env_learned.reset()
    done = False
    num_steps = 0
    episode_reward = 0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env_learned.step(action)
        episode_reward += reward
        num_steps += 1
    curr_episode_reward.append(episode_reward)
    curr_episode_steps.append(num_steps)

    num_of_steps.append(deepcopy(curr_episode_steps))
    rewards.append(deepcopy(curr_episode_reward))

print('STEPS\t RANDOM \t TRAINED')
for i in range(NUM_EPS):
    print(f'\t {num_of_steps[i][0]} \t {num_of_steps[i][1]}')

print('REWARDS\t RANDOM \t TRAINED')
for i in range(NUM_EPS):
    print(f'\t {rewards[i][0]} \t {rewards[i][1]}')
    

