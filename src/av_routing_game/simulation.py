from av_routing_game.env import RoutingEnv
from av_routing_game.policies import BeelinePolicy, A_star, Greedy, DQNPolicy

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    network_size = 5
    flow_direction = "both"  # Change this to "default" for single direction flow
    use_dqn = True  # Set to True to use trained DQN policy, False for manual policy selection
    
    MAX_EDGES_PER_INTERSECTION = 4
    MAX_STEPS = 100000
    EPISODES = 100

    total_reward_per_episode = np.zeros(EPISODES)
    for episode in tqdm(range(EPISODES)):

        env = RoutingEnv(size=network_size, num_agents_mean=1000, num_agents_std=20, 
                     discount_factor=0.9, flow_direction=flow_direction)

        observations, info = env.reset()

        # Create policy based on configuration
        if use_dqn:
            manual_policy = DQNPolicy(target=network_size ** 2 - 1, grid_size=network_size, discount_factor=env.discount_factor)
            print("Using DQN policy (chooses between Greedy and A*)")
        else:
            # Create policy with default target (will be overridden with agent-specific targets)
            manual_policy = BeelinePolicy(target=network_size ** 2 - 1, grid_size=network_size, discount_factor=env.discount_factor)
            print("Using manual policy selection")
    
        # Alternative policies:
        # manual_policy = Greedy(target=network_size ** 2 - 1, grid_size=network_size, discount_factor=env.discount_factor)
        # manual_policy = A_star(target=network_size ** 2 - 1, grid_size=network_size, discount_factor=env.discount_factor)
        
        dones = {0: False}
        actions = {0: 0, 1: 0}

        action_map = {0: "left", 1: "up", 2: "right", 3: "down"}

        while not all(dones.values()) and env.env_step < MAX_STEPS:
            current_agent = env.current_agent
            observation = observations[current_agent]
            agent_location = observation["position"]
            outgoing_edges = env.road_network.edges(agent_location)
            congestion = observation["congestion"]
            
            # Get agent-specific target
            agent_target = env.agent_targets[current_agent]

            action = manual_policy.act(current_location=agent_location, edges=outgoing_edges, 
                                    congestion=congestion, target=agent_target)

            observations, reward, dones, truncated, info = env.step(action)
            
            env.env_step += 1

            #done_count = 0
            #for agent, done in dones.items(): 
            #    if done:
            #        done_count += 1

        total_rewards = []
        for rewards in env.rewards.values():
            total_rewards.append(sum(rewards))

        total_reward_per_episode[episode] = sum(total_rewards)

    total_reward_per_episode = total_reward_per_episode / MAX_STEPS
    print("Reward mean:", np.mean(total_reward_per_episode))
    print("Reward std:", np.std(np.array(total_reward_per_episode)))

    print("Steps elapsed in simulation:", env.env_step)
    print(f"Flow direction: {flow_direction}")
    
    # Show distribution of agent targets
    target_counts = {}
    for target in env.agent_targets.values():
        target_counts[target] = target_counts.get(target, 0) + 1
    print(f"Target distribution: {target_counts}")
    
    # Show corner mappings for reference
    print(f"Grid corners (size {network_size}x{network_size}):")
    print(f"  Top-left: 0, Top-right: {network_size-1}")
    print(f"  Bottom-left: {(network_size-1)*network_size}, Bottom-right: {network_size**2-1}")

    plt.figure(figsize=(10, 10))
    plt.bar(env.rewards.keys(), sorted(total_rewards, reverse=True))
    plt.savefig("total_reward.png")

    plt.figure(figsize=(10, 10))
    plt.scatter(env.start_time.values(), total_rewards)
    plt.xlabel("start time")
    plt.ylabel('total reward')
    plt.savefig("total_reward_wrt_start_time.png")

    plt.figure(figsize=(10, 10))
    for route, congestion_per_route in env.congestion_per_route.items():
        #if np.mean(congestion_per_route) < -10:
        if np.min(congestion_per_route) != np.max(congestion_per_route):
            plt.plot([i for i in range(len(congestion_per_route))], congestion_per_route, label = f"{route}")

    plt.legend()
    plt.ylabel('Congestion')
    plt.xlabel("Time")
    plt.title("Congestion per link over time")
    plt.savefig("congestion_per_link.png")
    plt.show()


