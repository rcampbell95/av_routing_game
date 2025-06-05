"""
DQN Policy Selector for AV Routing Game

This script implements a Deep Q-Network that learns to choose between
Greedy and A* policies for autonomous vehicle routing.

Usage:
1. Train the DQN:
   python dqn.py

2. Use trained DQN in simulation:
   Set use_dqn=True in simulation.py

The DQN learns a mapping from environment state (position, target, congestion)
to policy choice (0=Greedy, 1=A*) to maximize cumulative reward.

State features:
- Normalized agent position (x, y)
- Normalized target position (x, y)  
- Normalized Manhattan distance to target
- Normalized Euclidean distance to target
- Congestion statistics (sum, max, mean)
- Progress toward goal (0-1)

Action space:
- 0: Use Greedy policy
- 1: Use A* policy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

from env import RoutingEnv
from policies import Greedy, A_star

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

class DQNAgent:
    def __init__(self, state_size, action_size=2, lr=2e-4, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        # Neural networks
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copy weights to target network
        self.update_target_network()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)
    
    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename))
        self.update_target_network()

def get_state_features(env, agent_id):
    """Extract state features for DQN input"""
    agent_pos = env.agent_locations[agent_id]
    target_pos = env.agent_targets[agent_id]
    
    # Position features
    pos_x = agent_pos % env.size
    pos_y = agent_pos // env.size
    target_x = target_pos % env.size
    target_y = target_pos // env.size
    
    # Distance features
    manhattan_dist = abs(pos_x - target_x) + abs(pos_y - target_y)
    euclidean_dist = np.sqrt((pos_x - target_x)**2 + (pos_y - target_y)**2)
    max_possible_distance = 2 * (env.size - 1)  # Max Manhattan distance in grid
    
    # Congestion features (simplified)
    observation = env.observations[agent_id]
    congestion_sum = np.sum(observation["congestion"])
    congestion_max = np.max(observation["congestion"]) if len(observation["congestion"]) > 0 else 0
    congestion_mean = np.mean(observation["congestion"]) if len(observation["congestion"]) > 0 else 0
    
    # Progress feature - how close are we to target compared to starting distance
    start_pos = 0 if target_pos == env.size ** 2 - 1 else env.size ** 2 - 1  # Assume opposite corners
    start_x = start_pos % env.size
    start_y = start_pos // env.size
    initial_dist = abs(start_x - target_x) + abs(start_y - target_y)
    progress = 1.0 - (manhattan_dist / max(initial_dist, 1))
    
    return np.array([
        pos_x / env.size, pos_y / env.size,  # Normalized position
        target_x / env.size, target_y / env.size,  # Normalized target
        manhattan_dist / max_possible_distance,  # Normalized Manhattan distance
        euclidean_dist / (env.size * np.sqrt(2)),  # Normalized Euclidean distance
        min(congestion_sum / 50.0, 1.0),  # Normalized congestion sum (capped at 1)
        min(congestion_max / 20.0, 1.0),   # Normalized congestion max (capped at 1)
        min(congestion_mean / 20.0, 1.0),  # Normalized congestion mean (capped at 1)
        max(0.0, min(1.0, progress))  # Progress toward goal (clamped 0-1)
    ])

def evaluate_policy_performance(env, policy, agent_id, target):
    """Evaluate a policy's performance for one step"""
    observation = env.observations[agent_id]
    agent_location = observation["position"]
    outgoing_edges = env.road_network.edges(agent_location)
    congestion = observation["congestion"]
    
    action = policy.act(current_location=agent_location, congestion=congestion, target=target)
    
    # Simulate the action to get reward (without actually taking it)
    if env.validate_action(agent_location, action):
        next_location = env.edge_transition(agent_location, action)
        road_to_next_location = tuple(sorted([agent_location, next_location]))
        
        # Calculate reward based on traffic params and vehicle counts
        if road_to_next_location in env.traffic_params:
            reward = -1 * np.matmul(env.traffic_params[road_to_next_location], 
                                  np.transpose(env.vehicle_counts[road_to_next_location]))
            return reward[0].item()
    
    return -10.0  # Heavy penalty for invalid actions

def train_dqn():
    print("Starting DQN training...")
    
    # Environment setup - reduce agents to get more meaningful episodes
    env = RoutingEnv(size=5, num_agents_mean=10, num_agents_std=3, 
                     discount_factor=0.9, flow_direction="both")
    
    # Policies
    greedy_policy = Greedy(target=24, grid_size=5, discount_factor=0.9)
    astar_policy = A_star(target=24, grid_size=5, discount_factor=0.9)
    policies = [greedy_policy, astar_policy]
    
    # DQN agent
    state_size = 10  # Features from get_state_features
    agent = DQNAgent(state_size)
    
    episodes = 1000
    max_steps_per_episode = 3000
    target_update_freq = 50
    
    scores = []
    experiences_collected = 0
    
    for episode in range(episodes):
        observations, _ = env.reset()
        
        # Set env_step high enough so all agents start immediately
        env.env_step = max(env.start_time.values()) if env.start_time else 0
        
        total_episode_delay = 0  # Track total delay (negative rewards) for all agents
        steps = 0
        episode_experiences = 0
        agent_states = {}  # Track state for each agent
        
        while not all(env.dones.values()) and steps < max_steps_per_episode:
            current_agent_id = env.current_agent
            
            # Skip if agent is done
            if env.dones[current_agent_id]:
                env.step(0)  # Take dummy action
                steps += 1
                continue
            
            # Get current state and location BEFORE step
            state = get_state_features(env, current_agent_id)
            target = env.agent_targets[current_agent_id]
            old_location = env.agent_locations[current_agent_id]
            
            # DQN chooses policy (0=Greedy, 1=A*)
            policy_choice = agent.act(state)
            chosen_policy = policies[policy_choice]
            
            # Execute chosen policy
            observation = env.observations[current_agent_id]
            agent_location = observation["position"]
            outgoing_edges = env.road_network.edges(agent_location)
            congestion = observation["congestion"]
            
            action = chosen_policy.act(current_location=agent_location, 
                                     congestion=congestion, target=target)
            
            # Validate action before taking step
            action_is_valid = env.validate_action(agent_location, action)
            
            # Take step in environment
            observations, reward, dones, truncated, info = env.step(action)
            
            # Debug output for first few steps
            if episode == 0 and steps < 10:
                print(f"Step {steps}: Agent {current_agent_id}, Action {action}, Valid: {action_is_valid}, Reward: {reward}, Done: {env.dones[current_agent_id]}")
            
            # Collect experience if this was a valid learning step
            # (agent was active, not waiting, and either took valid action or got meaningful reward)
            if action_is_valid and reward != 0:
                # Get next state
                next_state = get_state_features(env, current_agent_id)
                done = env.dones[current_agent_id]
                
                # Store experience and track
                agent.remember(state, policy_choice, reward, next_state, done)
                experiences_collected += 1
                episode_experiences += 1
                
                # Add to total episode delay (reward is negative delay)
                total_episode_delay += abs(reward)  # Convert to positive delay
            
            steps += 1
        
        # Episode score is total delay across all agents
        episode_score = -total_episode_delay  # Negative because we want to minimize delay
        scores.append(episode_score)
        
        # Train the agent multiple times per episode
        if episode_experiences > 0:
            training_iterations = max(1, episode_experiences // 5)  # Train more frequently
            avg_loss = np.zeros(training_iterations)
            for train_iter in range(training_iterations):
                avg_loss[train_iter] = agent.replay()
        
        # Force epsilon decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        if episode % 50 == 0:
            avg_score = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}, Experiences: {experiences_collected}, Loss: {avg_loss.mean()}")
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    agent.save("models/dqn_policy_selector.pth")
    print("Training completed! Model saved to models/dqn_policy_selector.pth")
    print(f"Total experiences collected: {experiences_collected}")

if __name__ == "__main__":
    train_dqn() 