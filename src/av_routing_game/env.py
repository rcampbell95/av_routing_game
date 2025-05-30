import functools

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx


class RoutingEnv(gym.Env):
    def __init__(self, render_mode=None, size=2, num_agents=10):
        self.size = size
        self.target = self.size - 1
        self.num_agents = num_agents
        self.current_agent = 0
        self.target = size - 1

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        observation_space = spaces.Dict(
            {
                "position": spaces.Discrete(self.size ** 2),
                "congestion": spaces.Box(low=1, high=10.0, shape=(2, 1), dtype=np.int32)
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
      self.vehicle_counts = {i: np.array([[1, 1]]) for i in range(self.size)} #

      self.traffic_params = {}
      for i in range(self.size):
        human_param = np.random.random() + 1
        av_param = human_param - np.random.random()
        self.traffic_params[i] = np.array([human_param, av_param])

      self.rewards = {i: [] for i in range(self.num_agents)}
      self.dones = {i: False for i in range(self.num_agents)}
      self.past_action = {i: None for i in range(self.num_agents)}
      self.env_step = 0
      self.start_time = {i: int(np.random.normal(self.num_agents // 2, self.num_agents // 10)) for i in range(self.num_agents)}
      self.congestion_per_route = {i: [] for i in range(self.size)}

      self.observations = {
        i: {
          "position": 0,
          "congestion": self.observation_space(0)["congestion"].sample()
        } for i in range(self.num_agents)
      }

      self.current_agent = 0

      info = {}

      return self.observations, info

    def step(self, action):
      current_location = self.agent_locations[self.current_agent]

      observation = {
        "position": self.agent_locations[self.current_agent],
        "congestion": self.vehicle_counts[current_location]
      }

      if self.past_action[self.current_agent]:
        past_road = self.past_action[self.current_agent]["location"]
        past_action = self.past_action[self.current_agent]["action"]

        if past_action >= 0 and past_action <= 3:
          self.vehicle_counts[past_road][0][0] -= 1 #max(self.vehicle_counts[past_road][0][0] - 1, 1)
        elif past_action <= 7:
          # add extra edges
          self.vehicle_counts[past_road][0][1] -= 1 #max(self.vehicle_counts[past_road][0][1] - 1, 1)

      if self.dones[self.current_agent] or self.env_step < self.start_time[self.current_agent]:
        self.current_agent = (self.current_agent + 1) % self.num_agents
        self.env_step += 1

        return observation, 0, self.dones, None, {}


      #if self.traffic_params[current_location] > 0
      reward = -1 * np.matmul(self.traffic_params[current_location], np.transpose(self.vehicle_counts[current_location]))
      reward = reward[0].item()

      for route in self.congestion_per_route:
        congestion = -1 * np.matmul(self.traffic_params[route], np.transpose(self.vehicle_counts[route]))

        self.congestion_per_route[route].append(congestion) # Might be better to move metric tracking to separate method

      # Action space is split in 2:
      # [0, 3] - Manually driving
      # [4, 7] - Autonomous driving
      if action >= 0 and action <= 3:
        self.vehicle_counts[current_location][0][0] += 1
      elif action <= 7:
        # add extra edges
        self.vehicle_counts[current_location][0][1] += 1

      info = {}
      truncated = None

      self.past_action[self.current_agent] = {"location": current_location, "action": action}

      if self.agent_locations[self.current_agent] == self.target:
        self.dones[self.current_agent] = True
        self.past_action[self.current_agent] = None

      observation = {}

      if not self.dones[self.current_agent]:
        self.agent_locations[self.current_agent] += 1
        observation = {
          "position": self.agent_locations[self.current_agent],
          "congestion": self.vehicle_counts[current_location]

        }
      
      self.rewards[self.current_agent].append(reward)
      self.current_agent = (self.current_agent + 1) % self.num_agents
      self.env_step += 1

      return observation, reward, self.dones, truncated, info

    def build_road_network(self, size: int) -> nx.Graph:
      graph = nx.Graph()

      graph.add_nodes_from([i for i in range(size ** 2)])

      for node in range(size ** 2):
        if node % size != 0 // size and (node - 1) >= 0:
          # Has an edge to the left
          graph.add_edge(node - 1, node)
        if node - 5 >= 0:
          graph.add_edge(node - 5, node)
