from typing import List
import heapq

import numpy as np

class Policy:
    def __init__(self, target: int, grid_size: int, discount_factor: float = 1.0):
        self.target = target 
        self.grid_size = grid_size
        self.discount_factor = discount_factor

    def act(self, current_location: int, edges: List[tuple[int, int]], congestion: np.ndarray) -> int:
        raise NotImplementedError

class BeelinePolicy(Policy):
    def __init__(self, target: int, grid_size: int, discount_factor: float = 1.0):
        super().__init__(target, grid_size, discount_factor)

    def act(self, current_location: int, edges: List[tuple[int, int]], congestion: np.ndarray) -> int:
        for edge in edges:
            if current_location + 1 in edge:
                return 2
            if current_location + self.grid_size in edge:
                return 3
        return 2

class A_star(Policy):
    def act(self, current_location: int, congestion: np.ndarray) -> int:
        """
        A* pathfinding considering congestion as edge weights.
        Returns the next action to take towards the target.
        """
        path = self._find_astar_path(current_location, congestion)
        if len(path) > 1:
            next_location = path[1]
            return self._get_action_to_location(current_location, next_location)
        return 2  # Default to right if no path found
    
    def _find_astar_path(self, start: int, congestion: np.ndarray):
        """Find optimal path using A* with congestion-weighted edges."""
        open_set = [(0, start, [start])]  # (f_score, location, path)
        visited = set()
        
        while open_set:
            f_score, current, path = heapq.heappop(open_set)
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == self.target:
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
                        h_score = self._manhattan_distance(next_location, self.target)
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
    def act(self, current_location: int, congestion: np.ndarray) -> int:
        """
        Greedy policy: moves towards goal preferring less congested routes.
        If both right and down moves are towards the goal, picks the one with lower congestion.
        """
        current_x = current_location % self.grid_size
        current_y = current_location // self.grid_size
        target_x = self.target % self.grid_size
        target_y = self.target // self.grid_size
        
        # Determine which directions move towards the target
        can_go_right = current_x < target_x and self._is_valid_action(current_location, 2)
        can_go_down = current_y < target_y and self._is_valid_action(current_location, 3)
        
        # If both directions are possible, choose based on congestion
        if can_go_right and can_go_down:
            right_congestion = self._get_congestion_cost(congestion, 2)
            down_congestion = self._get_congestion_cost(congestion, 3)
            
            if right_congestion <= down_congestion:
                return 2  # right
            else:
                return 3  # down
        
        # If only one direction is possible
        elif can_go_right:
            return 2  # right
        elif can_go_down:
            return 3  # down
        
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
