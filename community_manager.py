from community_model import CommunityModel
import random
from copy import deepcopy

NUM_COMMUNITIES = 4
TOTAL_RESOURCES_AVAILABLE = 120
TOTAL_RESOURCES_REQUIRED = 100

# manager to store community objects and provide their features to the RL agent over episodes
class CommunityManager:

    def __init__(self):
        # initialize communities, with 0 karma points
        self.communities = []
        for i in range(NUM_COMMUNITIES):
            self.communities.append(CommunityModel(i, 1,
            subsymbolic=True, utility_noise=5, utility_learning=True, strict_harvesting=True))

    # initialize resources between communities    
    def initialize_resources(self, karma_points):
        while True:
            # try allocating           
            # agency has a varying amount of resources each time
            agency_resources = random.randint(25, 50)

            # agency allocates resources solely based on karma points
            available_resources_agency = self.agency_allocate(agency_resources, karma_points)

            # randomly initialize each community's required resources
            required_resources = [random.randint(0,TOTAL_RESOURCES_REQUIRED//4) for i in range(NUM_COMMUNITIES)]
            for i in range(NUM_COMMUNITIES):
                print(f'Community {i}: Karma Points: {self.communities[i].karma_points}, Agency Allocation: {available_resources_agency[i]}')
            available_resources = deepcopy(available_resources_agency)
            total_resources_required = sum(required_resources)

            # randomly allocate initial resources possessed by the community in addition to agency allocation
            for i in range(NUM_COMMUNITIES):
                available_resources[i] += random.randint(0,(TOTAL_RESOURCES_AVAILABLE-agency_resources)//4)

            # enough resources are available but they are not distributed for sufficiency
            if sum(available_resources) >= sum(required_resources) and self.insufficient(available_resources, required_resources):
                return [available_resources, required_resources]

    # agency allocates their limited resources based on communities' karma points using a greedy allocation algorithm
    def agency_allocate(self, agency_resources, karma_points):
        denominator = sum(karma_points)
        return ([round(agency_resources * karma_points[i]/denominator) for i in range(NUM_COMMUNITIES)])

    # checks that the overall distribution is insufficient
    def insufficient(self, available_resources, required_resources):
        for i in range(NUM_COMMUNITIES):
            if available_resources[i] < required_resources[i]:
                return True
        return False
                
