from template import Agent
from Splendor.splendor_model import SplendorGameRule
from copy import deepcopy
import numpy as np
import random
import json
import os
from func_timeout import func_timeout, FunctionTimedOut
from collections import Counter
import heapq
import time 

# Constants
LEARNING_TIME = 1750
DISCOUNT_FACTOR = 0.9
NUMBER_PLAYERS = 2
WEIGHT_FILE = "agents/t_090/weights.json"
LEARNING_RATE = 0.02
LEARNING_DECAY = 0.99
LEARNING_MIN = 0.01
EPSILON = 0.15
EPSILON_DECAY = 0.99
EPSILON_MIN = 0.01
END_GAME_THRESHOLD = 15
TIME_LIMIT = 0.99

gem = {'red': 0, 'green': 0, 'blue': 0, 'black': 0, 'white': 0, 'yellow': 0}
card = {'score': 0, 'red': 0, 'green': 0, 'blue': 0, 'black': 0, 'white': 0, 'yellow': 0}

# True if training, False if testing
IS_TRAINING = True

# Initialize game rule
game_rule = SplendorGameRule(NUMBER_PLAYERS)

# Priority Queue class
class PriorityQueue:
    def __init__(self):
        self.queue_index = 0
        self.priority_queue = []

    def push(self, item, priority):
        # Push item into the queue with priority
        heapq.heappush(self.priority_queue,(priority, self.queue_index, item))
        self.queue_index += 1

    def empty(self):
        return len(self.priority_queue) == 0

    def pop(self):
        # Pop the item with the highest priority
        return heapq.heappop(self.priority_queue)[-1]

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.weights = self.load_weights()
        self.epsilon = EPSILON
        self.learning_rate = LEARNING_RATE
        
    def load_weights(self):
        # Load weights from file if exists
        if os.path.exists(WEIGHT_FILE):
            try:
                with open(WEIGHT_FILE) as f:
                    return json.load(f)
            except Exception as e:
                print(e)
                return {}
        else:
            # Initialize weights if file does not exist
            return self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize weights
        try:
            with open(WEIGHT_FILE, "w") as f:
                json.dump({}, f)
        except Exception as e:
            print(e)
            return {}
        
    def SelectAction(self, actions, game_state):
        self.game_state = deepcopy(game_state)
        
        # If testing, select action based on policy
        if not IS_TRAINING:
            return self.select_action_from_policy(actions, game_state)
        else:
            try:
                def SelectActionTraining(actions, game_state):
                    # If there is only one action, return it
                    if len(actions) == 1:
                        return actions[0]
                    
                    # Select action based on epsilon greedy
                    if random.uniform(0, 1) < self.epsilon:
                        # Use heuristic function to select action
                        best_action = self.heuristic_select_action(actions, game_state)
                    else:
                        # Select action based on policy
                        best_action = self.select_action_from_policy(actions, game_state)
                        
                    # Update epsilon
                    self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)
                    
                    # Do the action and calculate reward
                    game_state = deepcopy(game_state)
                    new_state = game_rule.generateSuccessor(game_state, best_action, self.id)
                    reward = self.get_reward(best_action, new_state, self.id)
                    
                    # Update weights
                    self.update_weights(best_action, reward, game_state, new_state)
                    
                    # Return the best action
                    if best_action:
                        return best_action
                    else:
                        return self.heuristic_select_action(actions, game_state)
                
                return func_timeout(LEARNING_TIME, lambda: SelectActionTraining(actions, game_state))
            
            except FunctionTimedOut:
                # If time out, return the action with the highest priority
                return self.heuristic_select_action(actions, game_state)
            
    def select_action_from_policy(self, actions, game_state):
        # Select action based on policy
        best_action = self.heuristic_select_action(actions, game_state)
        best_value = -float("inf")
        
        for action in actions:
            # Select action with the highest Q value
            q_value = self.get_q_value(action, game_state)
            
            if q_value > best_value:
                best_action = action
                best_value = q_value
                
        return best_action
    
    def get_q_value(self, action, game_state):
        # Calculate Q value
        q_value = 0
        for feature, value in self.select_action_from_feature(game_state, action).items():
            q_value += self.weights.get(feature, 0) * value
            
        return q_value
    
    def update_weights(self, action, reward, game_state, new_state):
        # Normalization
        lambda_1 = 0.01
        lambda_2 = 0.01
        
        next_legal_actions = game_rule.getLegalActions(new_state, self.id)
        
        # If there are no legal actions, set the next Q value to -inf
        if not next_legal_actions:
            next_q_value_max = -float("inf")
        else:
            next_q_value_max = 0
            
        # Calculate the maximum Q value of the next state
        for next_action in next_legal_actions:
            next_q_value = self.get_q_value(next_action, new_state)
            next_q_value_max = max(next_q_value_max, next_q_value)
            
        # Calculate the delta
        delta = reward + DISCOUNT_FACTOR * next_q_value_max - self.get_q_value(action, game_state)
        
        # Update weights
        for feature, value in self.select_action_from_feature(game_state, action).items():
            old_weight = self.weights.get(feature, 0)
            
            # Gradient update with normalization
            gradient_update = LEARNING_RATE * (delta * value)
            gradient_update -= LEARNING_RATE * lambda_1 * (1 if old_weight > 0 else -1)
            gradient_update-= LEARNING_RATE * lambda_2 * old_weight
            
            new_weight = old_weight + gradient_update
            self.weights[feature] = new_weight

        # Update learning rate
        self.learning_rate = max(LEARNING_MIN, self.learning_rate * LEARNING_DECAY)
        
        # Save weights
        self.save_weights(self.weights, WEIGHT_FILE)
        
    def heuristic_select_action(self, actions, game_state):
        # Select action based on heuristic function
        priority_queue = PriorityQueue()
        board_state = self.check_board(game_state, self.id)
        start_time = time.time()

        for action in actions:
            if time.time() - start_time > TIME_LIMIT:
                break
            
            action_rewards = self.check_action(action)
            priority_queue.push(action, self.heuristic_func(board_state, action_rewards))
            
        # Return the action with the highest priority
        return priority_queue.pop() if not priority_queue.empty() else random.choice(actions)
    
    def heuristic_func(self, on_board, action_rewards):
        # Calculate the heuristic value of an action
        f1 = self.feature1(on_board, action_rewards)
        f2 = self.feature2(action_rewards)
        f3 = 0
        f4 = self.improved_feature4(on_board, action_rewards)
        f5 = self.feature5(on_board, action_rewards)
        current_score = on_board["score"]
        
        return self.phase_function(f1, f2, f3, f4, f5, current_score)

    def phase_function(self, f1, f2, f3, f4, f5, current_score):
        final_distance = f1
        gem_ecoaffect = f2
        card_diversity = f3
        gem_diversity = f4
        nobel_distance = f5
        
        # Late phase
        if current_score > 12:
            return 30 * final_distance + 2 * gem_ecoaffect + 10 * card_diversity + 10 * gem_diversity +  35 * nobel_distance 
        
        # Mid phase
        elif 6 < current_score <= 12:
            return 20 * final_distance + 2 * gem_ecoaffect + 15 * card_diversity + 15 * gem_diversity + 25 * nobel_distance
        
        # Early phase
        else:
            return 20 * final_distance + 2 * gem_ecoaffect + 10 * card_diversity + 10 * gem_diversity + 30 * nobel_distance       

    def feature1(self, on_board, action_rewards):
        # This feature measures the distance between the player and the winning score 
        # after performing a certain action.
        current_score = on_board["score"] + action_rewards["score_reward"]
        max_score = 15
        score_ratio = current_score / max_score # Simplify the calculation
        urgency = 1 - score_ratio
        
        return urgency
    
    def feature2(self, action_rewards):
        gem_cost = action_rewards["gem_rewards"].values()
        gem_change = sum(value if value > 0 else 8 * value for value in gem_cost)

        cardnum = sum(list(action_rewards["gem_cards_rewards"].values())[1:])
        card_income = action_rewards["gem_cards_rewards"]["score"]

        # Avoid division by zero errors and simplify the code
        denominator_income = max(1, card_income + 1)
        denominator_cards = max(1, cardnum + 1)

        # Use a constant to standardize the scores
        normalization_constant = 20
        f2score = -((gem_change / denominator_income) + (gem_change / denominator_cards)) / normalization_constant

        return f2score

    def feature3(self, on_board):
        gem_cards = on_board["my_gemcard"]
        diversity_score = 0
        # Calculate the diversity score of the card, and the processing here is the same as that of gems
        for card in gem_cards:
            if gem_cards[card] > 2:
                diversity_score -= (gem_cards[card] - 2) 
            else:
                diversity_score += 1 

        return diversity_score
        
    def feature4(self, on_board):
        gem_values = on_board["my_gem"]
        diversity_score = 0

        # Calculate the diversity score of gems
        for gem in gem_values:
            if gem_values[gem] > 4:
                # If there are more than 4 gems, each additional one will reduce a certain score
                diversity_score -= (gem_values[gem] - 4)
            else:
                # If less than or equal to 4, each one will increase a certain score
                diversity_score += 1
                
        return diversity_score
    
    def improved_feature4(self, on_board, action_rewards):
        # Calculate the diversity score of gems
        gem_values = list(on_board["my_gem"].values())[:-1]
        # Predict future gem card counts, excluding the last type of card reward
        gem_card_value = np.array(list(on_board["my_gemcard"].values()))
        card_re = np.array(list(action_rewards["gem_cards_rewards"].values())[1:])
        future_cardgems = gem_card_value + card_re
        # Calculate scores for each gem type when less than 4 gems are present
        scores = [4 - gem for gem in gem_values]
        total_score = sum(scores)  # Calculate total score
        normalized_score = total_score / (sum(future_cardgems[:-1]) + 1)
        last_gem_card_bonus = future_cardgems[-1] * 2
        # Final diversity score
        diversity_score = normalized_score + last_gem_card_bonus
        return diversity_score 

    def feature5(self, on_board, action_rewards):
        # Evaluate the contribution of a given action to attracting noble cards
        f4score = 0
        gem_cards = Counter(on_board["my_gemcard"])
        re = Counter({k: action_rewards['gem_cards_rewards'][k] for k in list(action_rewards['gem_cards_rewards'].keys())[1:]})
        this_gem_card = dict(re + gem_cards)
        nobles = on_board["noble"]
        for noble in nobles:
            f4score += abs(sum(dict(Counter(noble) - Counter(this_gem_card)).values()))
        f4score = f4score / 9
        
        return f4score
    
    def check_board(self, state, agent_id):
        board = {"score": state.agents[agent_id].score, "my_gem": {}, "my_gemcard": {}, "noble": []}
        
        # Initialize gem and card counts
        my_gem = {color: 0 for color in gem}
        my_gemcard = {color: 0 for color in gem}

        # Aggregate gems and cards held by the agent
        agent_gems = state.agents[agent_id].gems
        agent_cards = state.agents[agent_id].cards

        for color in gem:
            gem_count = agent_gems.get(color, 0)
            card_count = len(agent_cards.get(color, []))
            my_gem[color] = gem_count + card_count
            my_gemcard[color] = card_count

        board["my_gem"] = my_gem
        board["my_gemcard"] = my_gemcard

        # Process noble cards available on the board
        for noble in state.board.nobles:
            noble_requirements = noble[1]
            noble_card = {color: noble_requirements.get(color, 0) for color in gem}
            noble_card['score'] = 3
            board["noble"].append(noble_card)

        return board
    
    def check_action(self, action):
        # Check the rewards of a given action
        rewards = {
            "score_reward": 0,
            "gem_rewards": deepcopy(gem),
            "gem_cards_rewards": deepcopy(card)
        }

        action_type = action["type"]
        if action_type == "reserve":
            # Reserve a card
            rewards["gem_rewards"]["yellow"] = 1
            
        elif action_type in ['buy_available', 'buy_reserve']:
            # Buy a card
            rewards["score_reward"] += action['card'].points
            rewards["gem_cards_rewards"]["score"] = action['card'].points
            rewards["gem_cards_rewards"][action['card'].colour] = 1

            for color in rewards["gem_rewards"]:
                rewards["gem_rewards"][color] = -action['returned_gems'].get(color, 0)
                
        elif action_type in ["collect_same", "collect_diff"]:
            # Collect gems
            for color in rewards["gem_rewards"]:
                # Calculate the gem rewards
                collected = action['collected_gems'].get(color, 0)
                returned = action['returned_gems'].get(color, 0)
                rewards["gem_rewards"][color] = collected - returned
        
        # Check if the action leads to a noble card
        if action.get('noble'):
            rewards["score_reward"] += 3

        return rewards
        
    def select_action_from_feature(self, game_state, action):
        # Select action based on features
        features = {}
        on_board = self.check_board(game_state, self.id)
        action_rewards = self.check_action(action)
        
        # Calculate features
        features["f1"] = self.feature1(on_board, action_rewards)
        features["f2"] = self.feature2(action_rewards)
        features["f3"] = self.feature3(on_board)
        features["f4"] = self.improved_feature4(on_board, action_rewards)
        features["f5"] = self.feature5(on_board, action_rewards)
    
        return features
    
    def save_weights(self, weights, file):
        # Save weights to file
        try:
            with open(file, "w") as f:
                json.dump(weights, f)
        except Exception as e:
            print(e)
            
    def get_reward(self, action, game_state, agent_id):
        reward = 0
        opponent_id = 1 - agent_id
        game_state = deepcopy(game_state)
        useful_cards = self.CheckUsefulCard(game_state)

        if action['type'] == 'buy_available' or action['type'] == 'buy_reserve':
            card = action['card']
            
            # Check if the card is useful
            if card in useful_cards:
                reward += 3
            
            # Check if buying the card will help achieve a noble
            noble_probabilities = self.CheckNobleProbability(game_state, agent_id)
            game_state_after_buy = game_rule.generateSuccessor(game_state, action, agent_id)
            noble_probabilities_after_buy = self.CheckNobleProbability(game_state_after_buy, agent_id)
            for noble in range(min(len(noble_probabilities), len(noble_probabilities_after_buy))):
                if noble_probabilities_after_buy[noble] > noble_probabilities[noble]:
                    reward += 3
                    
            # Check if opponent has high probability to buy the card
            opponent_card_probability = self.CheckCardProbability(game_state, opponent_id, card, game_state.agents[opponent_id].gems.get('yellow', 0))
            if opponent_card_probability > 0.75:
                reward += 1.5
                
            # Check if the card is useful for buying other useful cards
            for useful_card in useful_cards:
                if useful_card.points > card.points:
                    card_probability = self.CheckCardProbability(game_state, agent_id, useful_card, game_state.agents[agent_id].gems.get('yellow', 0))
                    if card_probability > 0.5:
                        reward += 3

        elif action['type'] == 'reserve':
            card = action['card']
            
            # Check if the card is useful
            if card in useful_cards:
                reward += 3
                
            # Check if after reserving the card, the player can buy useful cards in the next turn
            yellow_gems = game_state.agents[agent_id].gems.get('yellow', 0) + 1
            game_state_after_reserve = game_rule.generateSuccessor(game_state, action, agent_id)
            useful_cards_after_reserve = self.CheckUsefulCard(game_state_after_reserve)
            for useful_card in useful_cards_after_reserve:
                card_probability = self.CheckCardProbability(game_state_after_reserve, agent_id, useful_card, yellow_gems)
                if card_probability > 0.5:
                    reward += 3
                    
            # Check if the opponent has high probability to buy the reserved card
            opponent_card_probability = self.CheckCardProbability(game_state, opponent_id, card, game_state.agents[opponent_id].gems.get('yellow', 0))
            if opponent_card_probability > 0.75:
                reward += 1
                if card.points >= 3:
                    reward += 2
                
            # Penalize if the player reserves a card with very low probability to buy
            card_probability = self.CheckCardProbability(game_state, agent_id, card, yellow_gems)
            if card_probability < 0.25:
                reward -= 3
                
            # Penalize for returning gems
            reward -= 1 * sum(action['returned_gems'].values())
                             
        elif action['type'] == 'collect_diff' or action['type'] == 'collect_same':
            yellow_gems = game_state.agents[agent_id].gems.get('yellow', 0)
            game_state_after_collect = game_rule.generateSuccessor(game_state, action, agent_id)
            
            # Check if the probability of buying useful cards increases after collecting gems
            cards_prob_before = []
            for useful_card in useful_cards:
                card_probability = self.CheckCardProbability(game_state, agent_id, useful_card, yellow_gems)
                cards_prob_before.append(card_probability)
                
            cards_prob_after = []
            for useful_card in useful_cards:
                card_probability = self.CheckCardProbability(game_state_after_collect, agent_id, useful_card, yellow_gems)
                cards_prob_after.append(card_probability)
                
            for i in range(len(cards_prob_before)):
                if cards_prob_after[i] > cards_prob_before[i]:
                    reward += 3
                    
            # Check if the probability of buying reserved cards increases after collecting gems
            reserved_cards = game_state.agents[agent_id].cards['yellow']
            cards_prob_before = []
            for reserved_card in reserved_cards:
                card_probability = self.CheckCardProbability(game_state, agent_id, reserved_card, yellow_gems)
                cards_prob_before.append(card_probability)
                
            cards_prob_after = []
            for reserved_card in reserved_cards:
                card_probability = self.CheckCardProbability(game_state_after_collect, agent_id, reserved_card, yellow_gems)
                cards_prob_after.append(card_probability)
                
            for i in range(len(cards_prob_before)):
                if cards_prob_after[i] > cards_prob_before[i]:
                    reward += 1.5
                
            # Penalize for returning gems
            reward -= 1 * sum(action['returned_gems'].values())
                
            # Big penalty for returning yellow gems
            if 'yellow' in action['returned_gems']:
                reward -= 10
            
        if action.get('noble') is not None:
            reward += 5
        
        return reward
    
    def CheckNobleProbability(self, game_state, agent_id):
        # Check the probability of achieving each noble
        noble_probabilities = []
       
        for noble in game_state.board.nobles:
            noble_requirements = noble[1]
            cards_prob = []
            for card, count in noble_requirements.items():
                cards_prob.append(len(game_state.agents[agent_id].cards.get(card, [])) / count)
                 
            card_prob = min(cards_prob)
            noble_probabilities.append(card_prob)
            
        return noble_probabilities
    
    def CheckCardProbability(self, game_state, agent_id, card, yellow_gems):
        # Check the probability of achieving a specific card
        card_probabilities = []
        card_requirements = card.cost
        
        for color, count in card_requirements.items():
            gem_on_card = len(game_state.agents[agent_id].cards.get(color, []))
            gem = game_state.agents[agent_id].gems.get(color, 0)
            if gem_on_card >= count or gem >= count or (gem_on_card + gem) >= count:
                card_probabilities.append(1)
            elif (gem_on_card + yellow_gems) >= count:
                card_probabilities.append(1)
                yellow_gems -= count - gem_on_card
            elif (gem + yellow_gems) >= count:
                card_probabilities.append(1)
                yellow_gems -= count - gem
            elif (gem_on_card + gem + yellow_gems) >= count:
                card_probabilities.append(1)
                yellow_gems -= count - gem_on_card - gem
            else:
                card_probabilities.append((gem_on_card + gem + yellow_gems) / count)
                yellow_gems = 0
               
        return min(card_probabilities)
    
    def CheckUsefulCard(self, game_state):
        # Check if there are useful cards available on the board
        useful_cards = []
        dealt_list = game_state.board.dealt_list()
        
        for card in dealt_list:
            if card.points == 5:
                useful_cards.append(card)
            elif card.points == 4:
                if len(card.cost) == 1:
                    useful_cards.append(card)
            elif card.points == 3:
                if len(card.cost) == 1:
                    useful_cards.append(card)
            elif card.points == 2:
                if len(card.cost) == 1 or len(card.cost) == 2:
                    useful_cards.append(card)
            elif card.points == 1:
                total = 0
                for color in card.cost:
                    total += card.cost[color]
                if len(card.cost) == 1 or total == 7:
                    useful_cards.append(card)
                    
        return useful_cards