from typing import List
import heapq
import torch
import torch.nn as nn
import numpy as np
import os

class Policy:
    def __init__(self, target: int, grid_size: int, discount_factor: float = 1.0):
        self.target = target 
        self.grid_size = grid_size
        self.discount_factor = discount_factor

    def act(self, current_location: int, edges: List[tuple[int, int]], congestion: np.ndarray, target: int = None) -> int:
        raise NotImplementedError

class BeelinePolicy(Policy):
    def __init__(self, target: int, grid_size: int, discount_factor: float = 1.0):
        super().__init__(target, grid_size, discount_factor)

    def act(self, current_location: int, congestion: np.ndarray, target: int = None, edges: List[tuple[int, int]]=None) -> int:
        # Use provided target or fall back to instance target
        actual_target = target if target is not None else self.target
        
        current_x = current_location % self.grid_size
        current_y = current_location // self.grid_size
        target_x = actual_target % self.grid_size
        target_y = actual_target // self.grid_size
        
        # Move towards target: prioritize the direction with larger distance to cover
        dx = target_x - current_x
        dy = target_y - current_y
        
        # Check which directions are valid based on available edges
        valid_directions = set()
        for edge in edges:
            if current_location + 1 in edge and dx > 0:  # Can go right and target is right
                valid_directions.add(2)
            if current_location - 1 in edge and dx < 0:  # Can go left and target is left
                valid_directions.add(0)
            if current_location + self.grid_size in edge and dy > 0:  # Can go down and target is down
                valid_directions.add(3)
            if current_location - self.grid_size in edge and dy < 0:  # Can go up and target is up
                valid_directions.add(1)
        
        # Prioritize direction with larger distance to travel
        if abs(dx) >= abs(dy):
            # Prefer horizontal movement
            if 2 in valid_directions and dx > 0:  # Right
                return 2
            elif 0 in valid_directions and dx < 0:  # Left
                return 0
            elif 3 in valid_directions and dy > 0:  # Down
                return 3
            elif 1 in valid_directions and dy < 0:  # Up
                return 1
        else:
            # Prefer vertical movement
            if 3 in valid_directions and dy > 0:  # Down
                return 3
            elif 1 in valid_directions and dy < 0:  # Up
                return 1
            elif 2 in valid_directions and dx > 0:  # Right
                return 2
            elif 0 in valid_directions and dx < 0:  # Left
                return 0
        
        # Fallback: return any valid direction towards target
        if valid_directions:
            return min(valid_directions)
        
        # Final fallback: return a valid direction (original logic)
        for edge in edges:
            if current_location + 1 in edge:
                return 2
            if current_location + self.grid_size in edge:
                return 3
        return 2

class A_star(Policy):
    def act(self, current_location: int, congestion: np.ndarray, target: int = None, edges: List[tuple[int, int]] = None) -> int:
        """
        A* pathfinding considering congestion as edge weights.
        Returns the next action to take towards the target.
        """
        # Use provided target or fall back to instance target
        actual_target = target if target is not None else self.target
        
        path = self._find_astar_path(current_location, congestion, actual_target)
        if len(path) > 1:
            next_location = path[1]
            return self._get_action_to_location(current_location, next_location)
        return 2  # Default to right if no path found
    
    def _find_astar_path(self, start: int, congestion: np.ndarray, target: int):
        """Find optimal path using A* with congestion-weighted edges."""
        open_set = [(0, start, [start])]  # (f_score, location, path)
        visited = set()
        
        while open_set:
            f_score, current, path = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == target:
                return path
            
            # Explore neighbors
            for action in range(4):
                if self._is_valid_action(current, action):
                    next_location = self._get_next_location(current, action)
                    if next_location not in visited:
                        # Get congestion cost for this action
                        congestion_cost = self._get_congestion_cost(congestion, action)
                        
                        # Discount future costs based on number of steps ahead
                        steps_ahead = len(path)  # How many steps into the future this cost occurs
                        discounted_congestion = congestion_cost * (self.discount_factor ** steps_ahead)
                        
                        g_score = len(path)  # Steps taken so far
                        h_score = self._manhattan_distance(next_location, target)
                        f_score = g_score + h_score + discounted_congestion
                        
                        new_path = path + [next_location]
                        heapq.heappush(open_set, (f_score, next_location, new_path))
        
        return [start]  # Return single element if no path found
    
    def _get_congestion_cost(self, congestion: np.ndarray, action: int) -> float:
        """Extract congestion cost for a given action."""
        if action < len(congestion) and not np.isinf(congestion[action]).all():
            # Sum both human and AV congestion
            return np.sum(congestion[action])
        return 0.0
    
    def _manhattan_distance(self, pos1: int, pos2: int) -> int:
        """Calculate Manhattan distance between two grid positions."""
        x1, y1 = pos1 % self.grid_size, pos1 // self.grid_size
        x2, y2 = pos2 % self.grid_size, pos2 // self.grid_size
        return abs(x1 - x2) + abs(y1 - y2)
    
    def _is_valid_action(self, location: int, action: int) -> bool:
        """Check if action is valid from current location."""
        if action == 0:  # left
            return (location % self.grid_size) != 0
        elif action == 1:  # up
            return (location - self.grid_size) >= 0
        elif action == 2:  # right
            return ((location + 1) % self.grid_size) != 0
        elif action == 3:  # down
            return (location + self.grid_size) < self.grid_size ** 2
        return False
    
    def _get_next_location(self, location: int, action: int) -> int:
        """Get next location given current location and action."""
        if action == 0:  # left
            return location - 1
        elif action == 1:  # up
            return location - self.grid_size
        elif action == 2:  # right
            return location + 1
        elif action == 3:  # down
            return location + self.grid_size
        return location
    
    def _get_action_to_location(self, current: int, target: int) -> int:
        """Get action needed to move from current to target location."""
        diff = target - current
        if diff == -1:
            return 0  # left
        elif diff == -self.grid_size:
            return 1  # up
        elif diff == 1:
            return 2  # right
        elif diff == self.grid_size:
            return 3  # down
        return 2  # default

class Greedy(Policy):
    def act(self, current_location: int, congestion: np.ndarray, target: int = None, edges: List[tuple[int, int]]=None) -> int:
        """
        Greedy policy: moves towards goal preferring less congested routes.
        If both right and down moves are towards the goal, picks the one with lower congestion.
        """
        # Use provided target or fall back to instance target
        actual_target = target if target is not None else self.target
        
        current_x = current_location % self.grid_size
        current_y = current_location // self.grid_size
        target_x = actual_target % self.grid_size
        target_y = actual_target // self.grid_size
        
        # Determine which directions move towards the target
        can_go_right = current_x < target_x and self._is_valid_action(current_location, 2)
        can_go_down = current_y < target_y and self._is_valid_action(current_location, 3)
        can_go_left = current_x > target_x and self._is_valid_action(current_location, 0)
        can_go_up = current_y > target_y and self._is_valid_action(current_location, 1)
        
        # Collect all valid directions towards target
        towards_target_actions = []
        if can_go_right:
            towards_target_actions.append((self._get_congestion_cost(congestion, 2), 2))
        if can_go_down:
            towards_target_actions.append((self._get_congestion_cost(congestion, 3), 3))
        if can_go_left:
            towards_target_actions.append((self._get_congestion_cost(congestion, 0), 0))
        if can_go_up:
            towards_target_actions.append((self._get_congestion_cost(congestion, 1), 1))
        
        # If we have actions towards target, pick the least congested one
        if towards_target_actions:
            towards_target_actions.sort(key=lambda x: x[0])
            return towards_target_actions[0][1]
        
        # If can't move towards target directly, try other valid moves
        # Prefer actions with lower congestion
        valid_actions = []
        for action in range(4):
            if self._is_valid_action(current_location, action):
                congestion_cost = self._get_congestion_cost(congestion, action)
                valid_actions.append((congestion_cost, action))
        
        if valid_actions:
            # Sort by congestion and pick the least congested
            valid_actions.sort(key=lambda x: x[0])
            return valid_actions[0][1]
        
        return 2  # Default fallback
    
    def _get_congestion_cost(self, congestion: np.ndarray, action: int) -> float:
        """Extract congestion cost for a given action."""
        if action < len(congestion) and not np.isinf(congestion[action]).all():
            # Sum both human and AV congestion
            return np.sum(congestion[action])
        return float('inf')  # High cost for invalid actions
    
    def _is_valid_action(self, location: int, action: int) -> bool:
        """Check if action is valid from current location."""
        if action == 0:  # left
            return (location % self.grid_size) != 0
        elif action == 1:  # up
            return (location - self.grid_size) >= 0
        elif action == 2:  # right
            return ((location + 1) % self.grid_size) != 0
        elif action == 3:  # down
            return (location + self.grid_size) < self.grid_size ** 2
        return False

class DQN(nn.Module):
    def __init__(self, state_size, action_size=2, hidden_size=64):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNPolicy(Policy):
    def __init__(self, target: int, grid_size: int, discount_factor: float = 1.0, model_path: str = "models/dqn_policy_selector.pth"):
        super().__init__(target, grid_size, discount_factor)
        
        # Initialize the underlying policies
        self.greedy_policy = Greedy(target, grid_size, discount_factor)
        self.astar_policy = A_star(target, grid_size, discount_factor)
        self.policies = [self.greedy_policy, self.astar_policy]
        
        # Load the DQN model
        self.state_size = 10
        self.dqn = DQN(self.state_size)
        
        if os.path.exists(model_path):
            self.dqn.load_state_dict(torch.load(model_path))
            self.dqn.eval()
            print(f"DQN model loaded from {model_path}")
        else:
            print(f"Warning: DQN model not found at {model_path}. Using random policy selection.")
            self.dqn = None
    
    def act(self, current_location: int, edges: List[tuple[int, int]], congestion: np.ndarray, target: int = None) -> int:
        # Use provided target or fall back to instance target
        actual_target = target if target is not None else self.target
        
        # Get state features for DQN
        state_features = self._get_state_features(current_location, actual_target, congestion)
        
        # Choose policy using DQN
        if self.dqn is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
                q_values = self.dqn(state_tensor)
                policy_choice = torch.argmax(q_values).item()
        else:
            # Fallback to random selection if model not loaded
            policy_choice = np.random.choice([0, 1])
        
        # Execute chosen policy
        chosen_policy = self.policies[policy_choice]
        
        # Call the chosen policy's act method
        if isinstance(chosen_policy, (Greedy, A_star)):
            return chosen_policy.act(current_location, congestion, target=actual_target)
        else:  # BeelinePolicy case (shouldn't happen but just in case)
            return chosen_policy.act(current_location, edges, congestion, target=actual_target)
    
    def _get_state_features(self, agent_pos: int, target_pos: int, congestion: np.ndarray):
        """Extract state features for DQN input (matches dqn.py implementation)"""
        # Position features
        pos_x = agent_pos % self.grid_size
        pos_y = agent_pos // self.grid_size
        target_x = target_pos % self.grid_size
        target_y = target_pos // self.grid_size
        
        # Distance features
        manhattan_dist = abs(pos_x - target_x) + abs(pos_y - target_y)
        euclidean_dist = np.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2)
        max_possible_distance = 2 * (self.grid_size - 1)  # Max Manhattan distance in grid
        
        # Congestion features (simplified)
        congestion_sum = np.sum(congestion)
        congestion_max = np.max(congestion) if len(congestion) > 0 else 0
        congestion_mean = np.mean(congestion) if len(congestion) > 0 else 0
        
        # Progress feature - how close are we to target compared to starting distance
        start_pos = 0 if target_pos == self.grid_size ** 2 - 1 else self.grid_size ** 2 - 1  # Assume opposite corners
        start_x = start_pos % self.grid_size
        start_y = start_pos // self.grid_size
        initial_dist = abs(start_x - target_x) + abs(start_y - target_y)
        progress = 1.0 - (manhattan_dist / max(initial_dist, 1))
        
        return np.array([
            pos_x / self.grid_size, pos_y / self.grid_size,  # Normalized position
            target_x / self.grid_size, target_y / self.grid_size,  # Normalized target
            manhattan_dist / max_possible_distance,  # Normalized Manhattan distance
            euclidean_dist / (self.grid_size * np.sqrt(2)),  # Normalized Euclidean distance
            min(congestion_sum / 50.0, 1.0),  # Normalized congestion sum (capped at 1)
            min(congestion_max / 20.0, 1.0),   # Normalized congestion max (capped at 1)
            min(congestion_mean / 20.0, 1.0),  # Normalized congestion mean (capped at 1)
            max(0.0, min(1.0, progress))  # Progress toward goal (clamped 0-1)
        ])
