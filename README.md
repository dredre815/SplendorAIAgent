# Splendor AI Agent with Monte Carlo Tree Search

## Overview

This project implements an AI agent for playing the board game Splendor using Monte Carlo Tree Search (MCTS) with heuristic-guided action selection. The agent was developed as part of COMP90054 AI Planning for Autonomy at the University of Melbourne.

## Features

- **Monte Carlo Tree Search Implementation**: Advanced MCTS algorithm with UCB1 formula for node selection
- **Heuristic-Based Action Selection**: Multi-feature heuristic function for pruning the search space
- **Phase-Based Strategy**: Different weightings for game phases (early, mid, late)
- **Efficient Resource Management**: Sophisticated gem and card evaluation system
- **Time-Aware Processing**: Implements strict time management to avoid penalties

## Technical Implementation

### Core Components

1. **MCTS Node Class**
```python
class MCTSNode:
def init(self, agent_id, game_state, parent, action):
self.agent_id = agent_id
self.game_state = deepcopy(game_state)
self.parent = parent
self.action = action
self.children = []
self.visits = 0
self.value = 0
self.untried_actions = game_rule.getLegalActions(game_state, agent_id)
```

2. **Priority Queue for Action Selection**
```python
class PriorityQueue:
    def __init__(self):
        self.queue_index = 0
        self.priority_queue = []
```

### Key Features

1. **Phase-Based Decision Making**
- Early game (score ≤ 6): Focus on resource collection
- Mid game (6 < score ≤ 12): Balanced approach
- Late game (score > 12): Emphasis on point acquisition

2. **Heuristic Features**
- Feature 1: Distance to winning score
- Feature 2: Gem economy impact
- Feature 3: Card diversity
- Feature 4: Gem diversity
- Feature 5: Noble distance evaluation

## Usage

1. **Setup**
```bash
# Install required dependencies
python -m pip install func-timeout

# Run a game with random agents
python general_game_runner.py -g Splendor

# Run in interactive mode
python general_game_runner.py -g Splendor --interactive
```

2. **Configuration Options**
- `-t`: Text-only display
- `-s`: Save game replay
- `-l`: Save game log
- `-p`: Enable printing
- `--interactive`: Enable interactive mode

## Game Rules Modifications

The implementation includes some modifications to the original Splendor rules for computational efficiency:

1. Yellow tokens are treated as special gems
2. Card reservation limited to table cards
3. Card replacement occurs at turn end
4. Modified gem collection/return rules
5. Maximum 7 cards per color limit

## Technical Details

- Time limit per move: 1 second
- Warning system: 3 warnings before forfeit
- Initial startup allowance: 15 seconds
- Python version requirement: ≥ 3.8

## Project Structure

- `myTeam.py`: Main agent implementation
- `splendor_model.py`: Game state and action management
- `splendor_displayer.py`: Game visualization
- `general_game_runner.py`: Game execution system

## License

This project is part of COMP90054 at the University of Melbourne. All rights reserved.

## Development Team
```
| Student ID | Name         | Email                              | GitHub      |
|------------|--------------|----------------------------------- |-------------|
| 1160040    | Zijun Zhang  | zijuzhang1@student.unimelb.edu.au  | dredre815   |
| 1208784    | Di Wu        | dww3@student.unimelb.edu.au        | WD1120      |
| 1224627    | Xiaoyu Pang  | xiapang@student.unimelb.edu.au     | Pang1229    |
```