from av_routing_game.env import RoutingEnv
from av_routing_game.policies import BeelinePolicy

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    network_size = 5
    env = RoutingEnv(size=network_size, num_agents=100)

    MAX_EDGES_PER_INTERSECTION = 4
    MAX_STEPS = 100000

    observations, info = env.reset()

    observation = observations[0]
    dones = {0: False}
    actions = {0: 0, 1: 0}

    action_map = {0: "left", 1: "up", 2: "right", 3: "down"}

    manual_policy = BeelinePolicy(target=network_size ** 2 - 1, grid_size=network_size)

    while not all(dones.values()) and env.env_step < MAX_STEPS:
        current_agent = env.current_agent
        observation = observations[current_agent]
        agent_location = observation["position"]
        outgoing_edges = env.road_network.edges(agent_location)
        congestion = observation["congestion"]

        action = manual_policy.act(current_location=agent_location, edges=outgoing_edges, congestion=congestion)

        observations, reward, dones, truncated, info = env.step(action)
        
        env.env_step += 1

        #done_count = 0
        #for agent, done in dones.items(): 
        #    if done:
        #        done_count += 1

    total_rewards = []
    for rewards in env.rewards.values():
        total_rewards.append(sum(rewards))

    print("Steps elapsed in simulation:", env.env_step)

    plt.bar(env.rewards.keys(), sorted(total_rewards, reverse=True))
    plt.savefig("total_reward.png")
    plt.show()

    plt.scatter(env.start_time.values(), total_rewards)
    plt.xlabel("start time")
    plt.ylabel('total reward')
    plt.savefig("total_reward_wrt_start_time.png")
    plt.show()  

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


