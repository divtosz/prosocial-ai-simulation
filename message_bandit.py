import sys
import random

# bandit agent to learn the messages that communities respond to
class MessageBandit:
    def __init__(self, community):
        self.epsilon = sys.float_info.epsilon
        self.community = community # user associated with this agent
        self.exp = 0.3
        self.dist = 0.2
        self.decay = 0.8
        self.rwt = 0.8
        self.w0 = 0.5
        # nudge message and feedback
        self.num_messages = 4
        self.messages =  {0: ['Infants and babies in this community are starving everyday.',
                            'The children of this community are in dire need of donations.',
                            'The sick and the elderly of this community are dying due to lack of resources.',
                            'There is a family in this community that needs resources to survive.'],
                        1: ['You helped infants and babies in this community survive today with your generous donation.',
                            'You helped children of this community today.',
                            'You helped save some of the sick and elderly people of this community today.',
                            'You prevented starvation in a family today.']}
        self.keywords = [['infants','babies'], 'children', ['sick', 'elderly'], ['family']]
        self.options = [i for i in range(self.num_messages)]
        self.probs = [1/self.num_messages for i in range(self.num_messages)]
        self.w = [self.w0 for i in range(self.num_messages)]
        self.cum_rew = 0

    # make a nudge suggestion, based on what the agent knows about the donor, and current conditions in the recipient community
    def suggest(self, recipient_current_conditions):
        
        # in the case none of the conditions are currently present in the recipient community
        if len(recipient_current_conditions) == 0:
            return ['This community needs help.', -1]
        
        # only generate a true message, something that is actually happening in the recipient community
        currently_present_mask = [0 for i in range(self.num_messages)]
        for i, message in enumerate(self.messages[0]):
            if message in recipient_current_conditions:
                currently_present_mask[i] = 1

        # suggest nudge message option based on these factors
        wts = [(a*b)+ self.epsilon for a,b in zip(self.probs, currently_present_mask)]
        wts_sum = sum(wts)
        wts = [wt/wts_sum for wt in wts] # normalizing weights to use in making a suggestion
        suggested_option = random.choices(self.options, weights = wts)[0]
        return [self.messages[0][suggested_option], suggested_option]

    # learn from the community's response
    def learn(self, suggested_option, community_response):
        # learn from user's response to the suggestion

        # default message, nothing to learn
        if suggested_option == -1:
            return
        reward = community_response
        self.w[suggested_option] = self.decay * self.w[suggested_option] + self.rwt * reward
        normalized_w = self.normalized_weight(suggested_option)

        # update probability of chosen option
        self.probs[suggested_option] = normalized_w*(1-self.exp) + self.dist * self.exp
        self.normalize_probs()
        self.cum_rew += reward

    def normalize_probs(self):
        min_prob = min(self.probs)
        if min_prob < 0: # removing negative probabilities
            self.probs += abs(min_prob)
        p_sum = sum(self.probs)
        self.probs  = [prob/p_sum for prob in self.probs] # normalizing to add to 1

    def normalized_weight(self, choice):
        min_w = min(self.w)
        max_w = max(self.w)
        if max_w == min_w:
            return self.w[choice]/max_w
        return (self.w[choice]-min_w)/(max_w - min_w)

    # in the case of an accepted transaction on both sides, print positive feedback
    def print_feedback(self, suggested_option):
        if suggested_option == -1:
            print(f'Thank you, Community {self.community.id}! You now have {self.community.karma_points} karma points.')
        print(f'{self.messages[1][suggested_option]} Thank you, Community {self.community.id}! You now have {self.community.karma_points} karma points.')