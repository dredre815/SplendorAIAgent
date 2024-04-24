from template import Agent
from copy import deepcopy
import random
import math

class SplendorState:
    '''
    Class to represent the state of the game at any point in time
    '''
    def __init__(self, gems, cards, scores, nobles, reserved_cards):
        self.gems = gems  # Dictionary of available gems
        self.cards = cards  # Cards available for purchase
        self.scores = scores  # Dictionary of player scores
        self.nobles = nobles  # Noble tiles available for claiming
        self.reserved_cards = reserved_cards  # Cards reserved by players

    def copy(self):
        '''
        Returns a deep copy of the current state
        '''
        return SplendorState(deepcopy(self.gems), deepcopy(self.cards), deepcopy(self.scores),
                             deepcopy(self.nobles), deepcopy(self.reserved_cards))


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
    
    def SelectAction(self, actions, game_state):
        return actions[0]
    