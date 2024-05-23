import numpy as np
from template import Agent
from Splendor.splendor_model import SplendorGameRule
from math import log, sqrt
import time
import random
from copy import deepcopy
from collections import Counter
import heapq
G_LIST = {'red': 0, 'green': 0, 'blue': 0, 'black': 0, 'white': 0, 'yellow': 0}
C_LIST = {'score': 0, 'red': 0, 'green': 0, 'blue': 0, 'black': 0, 'white': 0, 'yellow': 0}

class PriorityQueue:
    def __init__(self):
        self.queue_index = 0
        self.priority_queue = []

    def push(self, item, priority):
        heapq.heappush(self.priority_queue,(priority, self.queue_index, item))
        self.queue_index += 1

    def empty(self):
        return len(self.priority_queue) == 0

    def pop(self):
        return heapq.heappop(self.priority_queue)[-1]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.id = _id  
        self.eid = (self.id+1) %2
        self.gamerule = SplendorGameRule(2)

    def SelectAction(self, actions, state):
        start_time = time.time()
        priority_queue = PriorityQueue()
        board_state = self.check_board(state,self.id)

        for action in actions:
            if time.time() - start_time < 0.9:
                action_rewards = self.check_action(action)
                priority_queue.push(action, self.heuristic_func(board_state, action_rewards))
            else:
                break
            
        return priority_queue.pop() if not priority_queue.empty() else random.choice(actions)
    
    def check_board(self, state, agent_id):
        board = {"score": state.agents[agent_id].score, "my_gem": {}, "my_gemcard": {}, "noble": []}
        
        # Initialize gem and card counts
        my_gem = {color: 0 for color in G_LIST}
        my_gemcard = {color: 0 for color in G_LIST}

        # Aggregate gems and cards held by the agent
        agent_gems = state.agents[agent_id].gems
        agent_cards = state.agents[agent_id].cards

        for color in G_LIST:
            gem_count = agent_gems.get(color, 0)
            card_count = len(agent_cards.get(color, []))
            my_gem[color] = gem_count + card_count
            my_gemcard[color] = card_count

        board["my_gem"] = my_gem
        board["my_gemcard"] = my_gemcard

        # Process noble cards available on the board
        for noble in state.board.nobles:
            noble_requirements = noble[1]
            noble_card = {color: noble_requirements.get(color, 0) for color in G_LIST}
            noble_card['score'] = 3
            board["noble"].append(noble_card)

        return board
    
    def check_action(self, action):
        rewards = {
            "score_reward": 0,
            "gem_rewards": deepcopy(G_LIST),
            "gem_cards_rewards": deepcopy(C_LIST)
        }

        action_type = action["type"]
        if action_type == "reserve":
            rewards["gem_rewards"]["yellow"] = 1
        elif action_type in ['buy_available', 'buy_reserve']:
            rewards["score_reward"] += action['card'].points
            rewards["gem_cards_rewards"]["score"] = action['card'].points
            rewards["gem_cards_rewards"][action['card'].colour] = 1
            for color in rewards["gem_rewards"]:
                rewards["gem_rewards"][color] = -action['returned_gems'].get(color, 0)
        elif action_type in ["collect_same", "collect_diff"]:
            for color in rewards["gem_rewards"]:
                collected = action['collected_gems'].get(color, 0)
                returned = action['returned_gems'].get(color, 0)
                rewards["gem_rewards"][color] = collected - returned

        if action.get('noble'):
            rewards["score_reward"] += 3

        return rewards

    def heuristic_func(self, on_board, action_rewards):
        f1 = self.feature1(on_board, action_rewards)
        f2 = self.feature2(action_rewards)
        f3 = self.feature3(on_board)
        f4 = self.feature4(on_board)
        f5 = self.feature5(on_board, action_rewards)
        current_score = f1
        return self.phase_function(f1, f2, f3, f4, f5,current_score)

    def phase_function(self, f1, f2, f3, f4, f5, current_score):
        final_distance = f1
        gem_ecoaffect = f2
        card_diversity = f3
        gem_diversity = f4
        nobel_distance = f5
        return 20 * final_distance + 5 * gem_ecoaffect + 10 * card_diversity + 10 * gem_diversity + 25 * nobel_distance
        
        
    def feature1(self, on_board, action_rewards):
        current_score = on_board["score"] + action_rewards["score_reward"]
        max_score = 15
        score_to_win = max_score - current_score
        score_ratio = current_score / max_score
        urgency = 1 - score_ratio  # lower then closer to win
        return urgency
    
    
    def feature2(self, action_rewards):
        gem_cost = action_rewards["gem_rewards"].values()
        gem_change = sum(value if value > 0 else 6 * value for value in gem_cost)
        cardnum = sum(list(action_rewards["gem_cards_rewards"].values())[1:])
        card_income = action_rewards["gem_cards_rewards"]["score"]
        denominator_income = max(1, card_income + 1)
        denominator_cards = max(1, cardnum + 1)
        normalization_constant = 10
        f2score = -((gem_change / denominator_income) + (gem_change / denominator_cards)) / normalization_constant

        return f2score


    def feature3(self, on_board):
        gem_cards = on_board["my_gemcard"]
        diversity_score = 0
        # Calculate the card's diversity score, which is treated the same as gems
        for card in gem_cards:
            if gem_cards[card] > 2:
                diversity_score -= (gem_cards[card] - 2) 
            else:
                diversity_score += 1 

        return diversity_score
        
    def feature4(self, on_board):
        gem_values = on_board["my_gem"]
        diversity_score = 0

        # Calculate the gem's diversity score
        for gem in gem_values:
            if gem_values[gem] > 4:
                # more than 4 decrease
                diversity_score -= (gem_values[gem] - 4)
            else:
                # less than 4 increase
                diversity_score += 1
        return diversity_score


    # Evaluating the potential contribution of a given action toward attracting noble cards.
    def feature5(self, on_board, action_rewards):
        f4score = 0
        gem_cards = Counter(on_board["my_gemcard"])
        re = Counter({k: action_rewards['gem_cards_rewards'][k] for k in list(action_rewards['gem_cards_rewards'].keys())[1:]})
        this_gem_card = dict(re + gem_cards)
        nobles = on_board["noble"]
        for noble in nobles:
            f4score += abs(sum(dict(Counter(noble) - Counter(this_gem_card)).values()))
        f4score = f4score / 10
        return f4score