import pyactr as actr
import numpy as np
import random
import re
from copy import deepcopy

# sentiments start fixed between communities
sentiments = [[1, 0.7, 0.6, 0.3],
            [0.8, 1, 0.2, 0.5],
            [0.6, 0.1, 1, 0.8],
            [0.2, 0.6, 0.9, 1]]

NUM_COMMUNITIES = 4
NUM_TRIGGER_WORDS = 2 # assuming each community has 2 trigger words for simplicity
POSSIBLE_TRIGGER_WORDS = ['infants', 'babies', 'children', 'sick', 'elderly', 'family']


# define the model for a community, modeling their responses to a nudge with the cognitive architecture ACT-R (pyactr)
class CommunityModel:
    def __init__(self, id, karma_points, **kwargs):
        self.id = id # 0,1,2,3
        self.karma_points = karma_points
        self.actr_response_model = actr.ACTRModel(**kwargs)

        # initialize pyactr chunk types
        actr.chunktype("start_donor", "sentiment, resource_amount")
        actr.chunktype("start_recipient", "sentiment, resource_requirement")

        self.utilities = [0 for i in range(24)]
        self.sentiment_val = 0
        self.donor = None
        self.recipient = None
        self.sentiments = sentiments[id]

        # possible conditions that can exist in a community at any given time
        self.possible_conditions = ['Infants and babies in this community are starving everyday.',
                            'The children of this community are in dire need of donations.',
                            'The sick and the elderly of this community are dying due to lack of resources.',
                            'There is a family in this community that needs resources to survive.']

        # assuming each community has 2 trigger words they respond to, which increases their chances of accepting a nudge to donate
        self.trigger_words = deepcopy(random.sample(POSSIBLE_TRIGGER_WORDS, NUM_TRIGGER_WORDS))
        self.donor_to_recipient = {0: [1,2,3],
                                    1: [0,2,3],
                                    2:[0,1,3],
                                    3:[0,1,2]}


    def set_available_resources(self, available_resources):
        self.available_resources = available_resources


    def set_required_resources(self, required_resources):
        self.required_resources = required_resources


    # initialize all possible donor productions
    def initialize_donor_productions(self):
        self.actr_response_model.productionstring(name="neutral_surplus_accept", string="""
        =g>
        isa start_donor
        sentiment 'neutral'
        resource_amount 'surplus'
        ==>
        ~g>
        """, utility = self.utilities[0], reward = self.calculate_reward('neutral_surplus_accept')
        )

        self.actr_response_model.productionstring(name="neutral_surplus_reject", string="""
        =g>
        isa start_donor
        sentiment 'neutral'
        resource_amount 'surplus'
        ==>
        ~g>
        """, utility = self.utilities[1], reward = self.calculate_reward('neutral_surplus_reject')
        )

        self.actr_response_model.productionstring(name="neutral_maintenance_accept", string="""
        =g>
        isa start_donor
        sentiment 'neutral'
        resource_amount 'maintenance'
        ==>
        ~g>
        """, utility = self.utilities[2], reward = self.calculate_reward('neutral_maintenance_accept')
        )

        self.actr_response_model.productionstring(name="neutral_maintenance_reject", string="""
        =g>
        isa start_donor
        sentiment 'neutral'
        resource_amount 'maintenance'
        ==>
        ~g>
        """, utility = self.utilities[3], reward = self.calculate_reward('neutral_maintenance_reject')
        )

        self.actr_response_model.productionstring(name="positive_surplus_accept", string="""
        =g>
        isa start_donor
        sentiment 'positive'
        resource_amount 'surplus'
        ==>
        ~g>
        """, utility = self.utilities[4], reward = self.calculate_reward('positive_surplus_accept')
        )

        self.actr_response_model.productionstring(name="positive_surplus_reject", string="""
        =g>
        isa start_donor
        sentiment 'positive'
        resource_amount 'surplus'
        ==>
        ~g>
        """, utility = self.utilities[5], reward = self.calculate_reward('positive_surplus_reject')
        )

        self.actr_response_model.productionstring(name="positive_maintenance_accept", string="""
        =g>
        isa start_donor
        sentiment 'positive'
        resource_amount 'maintenance'
        ==>
        ~g>
        """, utility = self.utilities[6], reward = self.calculate_reward('positive_maintenance_accept')
        )

        self.actr_response_model.productionstring(name="positive_maintenance_reject", string="""
        =g>
        isa start_donor
        sentiment 'positive'
        resource_amount 'maintenance'
        ==>
        ~g>
        """, utility = self.utilities[7], reward = self.calculate_reward('positive_maintenance_reject')
        )

        self.actr_response_model.productionstring(name="negative_surplus_accept", string="""
        =g>
        isa start_donor
        sentiment 'negative'
        resource_amount 'surplus'
        ==>
        ~g>
        """, utility = self.utilities[8], reward = self.calculate_reward('negative_surplus_accept')
        )

        self.actr_response_model.productionstring(name="negative_surplus_reject", string="""
        =g>
        isa start_donor
        sentiment 'negative'
        resource_amount 'surplus'
        ==>
        ~g>
        """, utility = self.utilities[9], reward = self.calculate_reward('negative_surplus_reject')
        )

        self.actr_response_model.productionstring(name="negative_maintenance_accept", string="""
        =g>
        isa start_donor
        sentiment 'negative'
        resource_amount 'maintenance'
        ==>
        ~g>
        """, utility = self.utilities[10], reward = self.calculate_reward('negative_maintenance_accept')
        )

        self.actr_response_model.productionstring(name="negative_maintenance_reject", string="""
        =g>
        isa start_donor
        sentiment 'negative'
        resource_amount 'maintenance'
        ==>
        ~g>
        """, utility = self.utilities[11], reward = self.calculate_reward('negative_maintenance_reject')
        )


    # initialize all possible recipient productions
    def initialize_recipient_productions(self):

        self.actr_response_model.productionstring(name="neutral_desirable_accept_donation", string="""
        =g>
        isa start_recipient
        sentiment 'neutral'
        resource_requirement 'desirable'
        ==>
        ~g>
        """, utility = self.utilities[12], reward = self.calculate_reward('neutral_desirable_accept_donation')
        )

        self.actr_response_model.productionstring(name="neutral_desirable_reject_donation", string="""
        =g>
        isa start_recipient
        sentiment 'neutral'
        resource_requirement 'desirable'
        ==>
        ~g>
        """, utility = self.utilities[13], reward = self.calculate_reward('neutral_desirable_reject_donation')
        )

        self.actr_response_model.productionstring(name="neutral_desperate_accept_donation", string="""
        =g>
        isa start_recipient
        sentiment 'neutral'
        resource_requirement 'desperate'
        ==>
        ~g>
        """, utility = self.utilities[14], reward = self.calculate_reward('neutral_desperate_accept_donation')
        )

        self.actr_response_model.productionstring(name="neutral_desperate_reject_donation", string="""
        =g>
        isa start_recipient
        sentiment 'neutral'
        resource_requirement 'desperate'
        ==>
        ~g>
        """, utility = self.utilities[15], reward = self.calculate_reward('neutral_desperate_reject_donation')
        )

        self.actr_response_model.productionstring(name="positive_desirable_accept_donation", string="""
        =g>
        isa start_recipient
        sentiment 'positive'
        resource_requirement 'desirable'
        ==>
        ~g>
        """, utility = self.utilities[16], reward = self.calculate_reward('positive_desirable_accept_donation')
        )

        self.actr_response_model.productionstring(name="positive_desirable_reject_donation", string="""
        =g>
        isa start_recipient
        sentiment 'positive'
        resource_requirement 'desirable'
        ==>
        ~g>
        """, utility = self.utilities[17], reward = self.calculate_reward('positive_desirable_reject_donation')
        )

        self.actr_response_model.productionstring(name="positive_desperate_accept_donation", string="""
        =g>
        isa start_recipient
        sentiment 'positive'
        resource_requirement 'desperate'
        ==>
        ~g>
        """, utility = self.utilities[18], reward = self.calculate_reward('positive_desperate_accept_donation')
        )

        self.actr_response_model.productionstring(name="positive_desperate_reject_donation", string="""
        =g>
        isa start_recipient
        sentiment 'positive'
        resource_requirement 'desperate'
        ==>
        ~g>
        """, utility = self.utilities[19], reward = self.calculate_reward('positive_desperate_reject_donation')
        )

        self.actr_response_model.productionstring(name="negative_desirable_accept_donation", string="""
        =g>
        isa start_recipient
        sentiment 'negative'
        resource_requirement 'desirable'
        ==>
        ~g>
        """, utility = self.utilities[20], reward = self.calculate_reward('negative_desirable_accept_donation')
        )

        self.actr_response_model.productionstring(name="negative_desirable_reject_donation", string="""
        =g>
        isa start_recipient
        sentiment 'negative'
        resource_requirement 'desirable'
        ==>
        ~g>
        """, utility = self.utilities[21], reward = self.calculate_reward('negative_desirable_reject_donation')
        )

        self.actr_response_model.productionstring(name="negative_desperate_accept_donation", string="""
        =g>
        isa start_recipient
        sentiment 'negative'
        resource_requirement 'desperate'
        ==>
        ~g>
        """, utility = self.utilities[22], reward = self.calculate_reward('negative_desperate_accept_donation')
        )

        self.actr_response_model.productionstring(name="negative_desperate_reject_donation", string="""
        =g>
        isa start_recipient
        sentiment 'negative'
        resource_requirement 'desperate'
        ==>
        ~g>
        """, utility = self.utilities[23], reward = self.calculate_reward('negative_desperate_reject_donation')
        )


    # convert action to its meaning and run simulation to get response
    def get_response(self, action, nudge_message = None):

        is_donor = False
        if nudge_message:
            is_donor = True
            self.nudge_message = nudge_message
        [self.donor, self.recipient] = self.convert_action(action)

        print(f'\nCommunity {self.id}')
        if is_donor:
            print(f'Nudge message: {nudge_message} Would you be willing to help them by making a donation?')
            print(f'Community {self.id}\'s trigger words: {self.trigger_words}')

            # run donating simulation
            self.initialize_donor_productions()
            self.sentiment_val = self.sentiments[self.recipient]
            if 0.4 <= self.sentiment_val <= 0.6:
                sentiment = "neutral"
            elif self.sentiment_val < 0.4:
                sentiment = "negative"
            else:
                sentiment = "positive"
            if self.available_resources >= 1.25 * self.required_resources: # surplus
                self.actr_response_model.goal.add(actr.makechunk(typename = "start_donor", sentiment = sentiment, resource_amount = "surplus"))
            else:
                self.actr_response_model.goal.add(actr.makechunk(typename = "start_donor", sentiment = sentiment, resource_amount = "maintenance"))
        else:
            # run recipient simulation
            self.initialize_recipient_productions()
            self.sentiment_val = self.sentiments[self.donor]
            if 0.4 <= self.sentiment_val <= 0.6:
                sentiment = "neutral"
            elif self.sentiment_val < 0.4:
                sentiment = "negative"
            else:
                sentiment = "positive"
            if self.available_resources <= 0.75 * self.required_resources: # desperate
                self.actr_response_model.goal.add(actr.makechunk(typename = "start_recipient", sentiment = sentiment, resource_requirement = "desperate"))
            else:
                self.actr_response_model.goal.add(actr.makechunk(typename = "start_recipient", sentiment = sentiment, resource_requirement = "desirable"))

        sim = self.actr_response_model.simulation(trace = False)
        sim.steps(2)
        print(f'PRODUCTION FIRED: {sim.current_event}')
        response = sim.current_event.action.split(': ')[1].split('_')[2]
        self.response = (response == 'accept')

        if self.response and not is_donor:
            # increase recipient's sentiments towards donor
            self.sentiments[self.donor] = min(self.sentiments[self.donor] * 1.0001, 1)
            self.sentiment_val = self.sentiments[self.donor]

        sim.run()
        if is_donor:
            self.set_donor_utilities()
        else:
            self.set_recipient_utilities()
        return self.response

    def convert_action(self, action):
        donor = action//(NUM_COMMUNITIES-1)
        recipient_index = action%(NUM_COMMUNITIES-1)
        recipient = self.donor_to_recipient[donor][recipient_index]
        return [donor, recipient]

    def set_donor_utilities(self):
        self.utilities[:12] = [self.actr_response_model.productions['neutral_surplus_accept']['utility'],
                        self.actr_response_model.productions['neutral_surplus_reject']['utility'],
                        self.actr_response_model.productions['neutral_maintenance_accept']['utility'],
                        self.actr_response_model.productions['neutral_maintenance_reject']['utility'],
                        self.actr_response_model.productions['positive_surplus_accept']['utility'],
                        self.actr_response_model.productions['positive_surplus_reject']['utility'],
                        self.actr_response_model.productions['positive_maintenance_accept']['utility'],
                        self.actr_response_model.productions['positive_maintenance_reject']['utility'],
                        self.actr_response_model.productions['negative_surplus_accept']['utility'],
                        self.actr_response_model.productions['negative_surplus_reject']['utility'],
                        self.actr_response_model.productions['negative_maintenance_accept']['utility'],
                        self.actr_response_model.productions['negative_maintenance_reject']['utility']]
        

    def set_recipient_utilities(self):
        self.utilities[12:] = [self.actr_response_model.productions['neutral_desirable_accept_donation']['utility'],
                    self.actr_response_model.productions['neutral_desirable_reject_donation']['utility'],
                    self.actr_response_model.productions['neutral_desperate_accept_donation']['utility'],
                    self.actr_response_model.productions['neutral_desperate_reject_donation']['utility'],
                    self.actr_response_model.productions['positive_desirable_accept_donation']['utility'],
                    self.actr_response_model.productions['positive_desirable_reject_donation']['utility'],
                    self.actr_response_model.productions['positive_desperate_accept_donation']['utility'],
                    self.actr_response_model.productions['positive_desperate_reject_donation']['utility'],
                    self.actr_response_model.productions['negative_desirable_accept_donation']['utility'],
                    self.actr_response_model.productions['negative_desirable_reject_donation']['utility'],
                    self.actr_response_model.productions['negative_desperate_accept_donation']['utility'],
                    self.actr_response_model.productions['negative_desperate_reject_donation']['utility']]


    # calculate reward for the productions
    def calculate_reward(self, response_string):
        if self.available_resources == self.required_resources:
            return 0
        response_array = response_string.split('_') 
        response = (response_array[2] == 'accept')
        
        if len(response_array) == 3:
            # donor
            self.sentiment_val = self.sentiments[self.recipient]
            message_words = set(re.split('[ !,.]', self.nudge_message.lower()))

            # if the message contains trigger words for the community, they are more likely to accept
            trigger_factor = 1
            for trigger_word in self.trigger_words:
                if trigger_word in message_words:
                    trigger_factor *= 1.5
            if response:
                # donor acceptance
                reward = trigger_factor * self.sentiment_val * (self.available_resources - self.required_resources)
            else:
                # donor rejection
                reward = 1/trigger_factor * (1-self.sentiment_val) * self.required_resources/(self.available_resources - self.required_resources) # if more resources, they should be more likely to share 
                                                                                            # so inversely proportion to both sentiment (positive feelings) and extra amount, as well as trigger factor (if more than 1)
        else:
            # recipient
            self.sentiment_val = self.sentiments[self.donor]

            # factor of 50% extra reward because people are likely to be less picky about accepting a donation than donating
            if response:
                reward = (1.5 * self.sentiment_val) * (self.required_resources - self.available_resources)
            else:
                reward = (1.5 *(1-self.sentiment_val)) * self.available_resources/(self.required_resources - self.available_resources)
        return reward


    # simulate current conditions in the community at a given time, so the message bandit can generate a true message nudge
    def simulate_current_conditions(self):
        self.current_conditions = []      
        for condition in self.possible_conditions:
            # 50% probability of each possible condition being true at any time
            if random.uniform(0,1) >= 0.5:
                self.current_conditions.append(condition)



        




        
        















       
    
