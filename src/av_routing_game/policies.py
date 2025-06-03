from typing import List

import numpy as np

class Policy:
    def __init__(self, target: int, grid_size: int):
        self.target = target 
        self.grid_size = grid_size

    def act(self, current_location: int, edges: List[tuple[int, int]], congestion: np.ndarray) -> int:
        raise NotImplementedError

class BeelinePolicy(Policy):
    def __init__(self, target: int, grid_size: int):
        super().__init__(target, grid_size)

    def act(self, current_location: int, edges: List[tuple[int, int]], congestion: np.ndarray) -> int:
        for edge in edges:
            if current_location + 1 in edge:
                return 2
            if current_location + self.grid_size in edge:
                return 3
        return 2
class A_star(Policy):
    def act(self, current_location: int, congestion: np.ndarray) -> int:
        pass

class Manual(Policy):
    def act(self, current_location: int, congestion: np.ndarray) -> int:
        pass
