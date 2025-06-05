import functools

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx

class RoutingEnv(gym.Env):
    def __init__(self, render_mode=None, target=None, 
                 size=2, num_agents_mean=100, num_agents_std=20, discount_factor=0.9,
                 start_time_generator=None):
        if not target:
            self.target = size ** 2 - 1
        self.size = size
        self.num_agents_mean = num_agents_mean
        self.num_agents_std = num_agents_std
        self.num_agents = int(np.random.normal(self.num_agents_mean, self.num_agents_std))
        self.discount_factor = discount_factor
        self.current_agent = 0
        self.start_time_generator = lambda agent_id, num_agents: int(np.random.normal(num_agents // 2, num_agents // 10)) if start_time_generator is None else start_time_generator

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        observation_space = spaces.Dict(
            {
                "position": spaces.Discrete(self.size ** 2),
                "congestion": spaces.Box(low=-10, high=0, shape=(4, 1, 2), dtype=np.int32)
            }
        )
        return observation_space

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # We can seed the action space to make the environment deterministic.
        return spaces.Discrete(8)


    def reset(self):
        self.agent_locations = {i: 0 for i in range(self.num_agents)} 
          # Change to account for edges, not vertices
        # (manual_drivers, platoon_vehicles)

        self.road_network = self.build_road_network(self.size)
        self.vehicle_counts = {i: np.array([[0, 0]]) for i in self.road_network.edges()} #   
        self.traffic_params = {i: None for i in self.road_network.edges()}
        
        for edge in self.road_network.edges():
            human_param = np.random.random() + 1
            av_param = human_param - np.random.random()
            self.traffic_params[edge] = np.array([human_param, av_param])    
        
        self.rewards = {i: [] for i in range(self.num_agents)}
        self.dones = {i: False for i in range(self.num_agents)}
        self.past_action = {i: None for i in range(self.num_agents)}
        self.env_step = 0
        self.start_time = {i: self.start_time_generator(i, self.num_agents) for i in range(self.num_agents)}
        self.congestion_per_route = {i: [] for i in self.road_network.edges()}   
        self.observations = {
            i: {
            "position": 0,
            "congestion": self.observation_space(0)["congestion"].sample() 
            } for i in range(self.num_agents)
        }

        self.current_agent = 0

        info = {}

        self.action_counts = {i: {i: 0 for i in range(4)} for i in range(self.num_agents)}

        return self.observations, info

    def step(self, action):
        current_location = self.agent_locations[self.current_agent]

        # Same shape as congestion in observation
        # First test with fixed traffic
        congestion = np.random.random([4, 1, 2])

        observation = {
          "position": current_location,
          "congestion": congestion
        }

        if self.past_action[self.current_agent]:
            past_road = self.past_action[self.current_agent]["location"]
            past_action = self.past_action[self.current_agent]["action"]  
            if past_action // 4 == 0:
                self.vehicle_counts[past_road][0][0] -= 1 #max(self.vehicle_counts[past_road][0][0] - 1, 1)
            elif past_action // 4 == 1:
                # add extra edges
                self.vehicle_counts[past_road][0][1] -= 1 #max(self.vehicle_counts[past_road][0][1] - 1, 1) 
        
            self.past_action[self.current_agent] = None
        # Finish step without update if
            # The current agent is done
            # If the current step is less than the agent start time
            # If the action is invalid
        current_agent_done = self.dones[self.current_agent]
        agent_waiting= self.env_step < self.start_time[self.current_agent]
        invalid_action = not self.validate_action(current_location, action)
        if current_agent_done or agent_waiting or invalid_action:
            self.current_agent = (self.current_agent + 1) % self.num_agents
            
            self.action_counts[self.current_agent][action] += 1

            self.observations[self.current_agent] = observation

            return self.observations, 0, self.dones, None, {}   
        
        for route in self.congestion_per_route:
            congestion_at_edge = -1 * np.matmul(self.traffic_params[route], np.transpose(self.vehicle_counts[route])) 
            self.congestion_per_route[route].append(congestion_at_edge) # Might be better to move metric tracking to separate method  
        # Action space is split in 2:
        # [0, 3] - Manually driving
        # [4, 7] - Autonomous driving

        next_location = self.edge_transition(current_location, action)

        road_to_next_location = tuple(sorted([current_location, next_location]))
        if action // 4 == 0:
            self.vehicle_counts[road_to_next_location][0][0] += 1
        elif action // 4 == 1:
            # add extra edges
            self.vehicle_counts[road_to_next_location][0][1] += 1  

        #if self.traffic_params[current_location] > 0
        reward = -1 * np.matmul(self.traffic_params[road_to_next_location], np.transpose(self.vehicle_counts[road_to_next_location]))
        reward = reward[0].item()   

        info = {}
        truncated = None    
        self.past_action[self.current_agent] = {"location": road_to_next_location, "action": action} 
                
        if not self.dones[self.current_agent]:
            self.agent_locations[self.current_agent] = next_location
            congestion = self.route_congestions_at_intersection(next_location)

            self.observations[self.current_agent] = {
              "position": next_location,
              "congestion": congestion 
            }

            if next_location == self.target:
                self.dones[self.current_agent] = True

        self.rewards[self.current_agent].append(reward)
        self.current_agent = (self.current_agent + 1) % self.num_agents
 
        return self.observations, reward, self.dones, truncated, info 

    def build_road_network(self, size: int) -> nx.Graph:
        graph = nx.Graph()

        graph.add_nodes_from([i for i in range(size ** 2)])

        for node in range(size ** 2):
            if node % size != 0:
                # Has an edge to the left
                graph.add_edge(node - 1, node)
            if node - self.size >= 0:
                graph.add_edge(node - self.size, node)

        return graph
    

    def validate_action(self, current_location: int, action: int) -> bool:
        action_direction = action % 4
        if action_direction == 0 and (current_location % self.size) != 0:
            return True
        if action_direction == 1 and (current_location - self.size) >= 0:
            return True
        if action_direction == 2 and ((current_location + 1) % self.size) != 0:
            return True
        if action_direction == 3 and (current_location + self.size) < self.size ** 2:
            return True

        return False
    
    def edge_transition(self, location: int, action: int) -> int:
        action_direction = action % 4
 
        if action_direction == 0:
            return location - 1 

        if action_direction == 1:
            return location - self.size

        if action_direction == 2:
            return location + 1

        if action_direction == 3:
            return location + self.size
        
    def route_congestions_at_intersection(self, agent_location: int) -> np.ndarray:
        """
        Return congestion array of size (4, 1, 2) with congestion per outgoing route (edge)
        at an intersection     
        """
        congestion = np.full((4, 1, 2), fill_value=np.inf)
        num_actions = 4

        for edge in self.road_network.edges(agent_location):
            edge = edge if edge[0] < edge[1] else (edge[1], edge[0])
            edge_congestion = self.vehicle_counts[edge] * self.traffic_params[edge]
            
            for action_idx in range(num_actions):
                if self.edge_transition(agent_location, action_idx) in edge:
                    congestion[action_idx] = edge_congestion

        return congestion


