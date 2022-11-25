import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
import pyactr as actr
from community_model import CommunityModel
from community_manager import CommunityManager
from message_bandit import MessageBandit

PREV_ACTIONS_LEN = 30
NUM_COMMUNITIES = 4
TOTAL_RESOURCES_AVAILABLE = 105
TOTAL_RESOURCES_REQUIRED = 100


class NudgingEnv(gym.Env):

    def __init__(self):
        super(NudgingEnv, self).__init__()
        # Define action and observation space
        
        self.action_space = spaces.Discrete(12)
        # observations: each of the communities' resources, needs, prev actions
        # 4 + 4 + 30
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(8+PREV_ACTIONS_LEN,), dtype=np.float64)

        # communities remain the same over episodes, with new resource values initialized
        # CommunityManager stores the community objects
        self.community_manager = CommunityManager()
        self.communities = self.community_manager.communities

        # store the bandit agents for each community, which will be updated when they learn the messages that communities respond to
        # they will learn over multiple episodes
        self.message_bandit_map = dict()
        for i in range(NUM_COMMUNITIES):
            message_bandit = MessageBandit(self.communities[i])
            self.message_bandit_map[self.communities[i]] = message_bandit
        

    def step(self, action):
        self.prev_actions.append(action)

        # from the discrete action value, get the doner and donee communities
        donor = action//(NUM_COMMUNITIES-1)
        donee_index = action%(NUM_COMMUNITIES-1)
        i = 0
        steps = 0
        if donor == 0:
            i = 1
        while steps < donee_index:
            if i == donor:
                i += 1
                continue
            i += 1
            steps += 1
        donee = i
        print(f'\nAction: {action}')
        print(f'\nNudge suggested: Community {donor} donates to Community {donee}.')

        if self.communities[donor].available_resources <= self.communities[donor].required_resources or self.communities[donee].available_resources >= self.communities[donee].required_resources:
            # transaction doesnt make sense
            self.reward = -100
            self.negative_reward += 1
            if self.negative_reward == 1000: # capping the episode when 1000 senseless transactions are suggested
                self.done = True
                print('\nCOMMUNITY RESOURCES')
                for i in range(NUM_COMMUNITIES):
                    print(f'Community {i}: Available Resources: {self.communities[i].available_resources} \t Required Resources: {self.communities[i].required_resources} \t Karma Points: {self.communities[i].karma_points}')
                print('\nFINAL COMMUNITY SENTIMENTS:')
                for i in range(NUM_COMMUNITIES):
                    print(self.communities[i].sentiments)

        else:
            # check if this nudge is accepted by the communities
            self.communities[donee].simulate_current_conditions()
            print(f'\nCurrent conditions in Community {self.communities[donee].id}: {self.communities[donee].current_conditions}')
            # generate nudge message for the donor based on a bandit for this community and the current conditions for the donee community
            message_bandit = self.message_bandit_map[self.communities[donor]]
            nudge_message, option = message_bandit.suggest(self.communities[donee].current_conditions)
            self.negative_reward = 0 # reset this, since a non-senseless transaction has been suggested
            
            # get response from both parties
            response_donee  =False
            response_donor = self.communities[donor].get_response(action, nudge_message)
            print(f'DONOR RESPONSE: {response_donor}')

            # learn about the message from the donor's response
            message_bandit.learn(option, response_donor)

            if response_donor:
                response_donee = self.communities[donee].get_response(action)
                print(f'DONEE RESPONSE: {response_donee}')

            if response_donor and response_donee:
                self.communities[donor].karma_points += 0.0001
                message_bandit.print_feedback(option)
                self.communities[donor].available_resources -= 1
                self.communities[donee].available_resources += 1
                response_reward = 250

            else:
                response_reward = 0

            if self.sufficient():
                # all communities are self sufficient
                self.done = True
                self.reward = response_reward + 10000
                print('\nSELF SUFFICIENT COMMUNITIES')
                for i in range(NUM_COMMUNITIES):
                    print(f'Community {i}: Available Resources: {self.communities[i].available_resources} \t Required Resources: {self.communities[i].required_resources} \t Karma Points: {self.communities[i].karma_points}')

                print('\nFINAL COMMUNITY SENTIMENTS:')
                for i in range(NUM_COMMUNITIES):
                    print(self.communities[i].sentiments)
            
            else:
                # penalize the RL agent by the difference in sufficiency
                self.calculate_insufficiency()
                self.reward = 50 + response_reward - self.insufficiency_amt

        info = {}

        # create observation:                
        observation = []
        for i in range(NUM_COMMUNITIES):
            observation.append(self.communities[i].available_resources)
            observation.append(self.communities[i].required_resources)
        observation = np.array(observation + list(self.prev_actions))                          

        return observation, self.reward, self.done, info

    def reset(self):

        # use the same community objects, to train over multiple episodes incorporating their changes too
        karma_points = [self.communities[i].karma_points for i in range(NUM_COMMUNITIES)]

        # initialize resources based on communities' karma points, making sure there is an insufficiency
        [available_resources, required_resources] = self.community_manager.initialize_resources(karma_points)
        for i in range(NUM_COMMUNITIES):
            self.communities[i].set_available_resources(available_resources[i])
            self.communities[i].set_required_resources(required_resources[i])
        
        print('\nRESOURCE STATISTICS')
        for i in range(NUM_COMMUNITIES):
            print(f'Community {i}: Available Resources: {self.communities[i].available_resources} \t Required Resources: {self.communities[i].required_resources} \t Karma Points: {self.communities[i].karma_points}')
            
        # initialize action history
        self.prev_actions = deque(maxlen = PREV_ACTIONS_LEN)
        for i in range(PREV_ACTIONS_LEN):
            self.prev_actions.append(-1)

        # create observation:
        observation = []
        for i in range(NUM_COMMUNITIES):
            observation.append(self.communities[i].available_resources)
            observation.append(self.communities[i].required_resources)
        observation = np.array(observation + list(self.prev_actions))    
        self.done = False
        self.negative_reward = 0

        return observation

    # check if all communities are self sufficient
    def sufficient(self):
        for community in self.communities:
            if community.available_resources < community.required_resources:
                return False
        return True

    # calculate amount of insufficiency amongst all communities
    def calculate_insufficiency(self):
        self.insufficiency_amt = 0
        for community in self.communities:
            self.insufficiency_amt += max(0, community.required_resources- community.available_resources)
