from av_routing_game.env import RoutingEnv
from av_routing_game.policies import BeelinePolicy

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    network_size = 3
    env = RoutingEnv(size=network_size, num_agents=200)

    MAX_EDGES_PER_INTERSECTION = 4
    MAX_STEPS = 100000

    observation, info = env.reset()

    dones = {0: False}
    actions = {0: 0, 1: 0}

    step = 0
    action_map = {0: "left", 1: "up", 2: "right", 3: "down"}

    manual_policy = BeelinePolicy(target=network_size ** 2 - 1, grid_size=network_size)

    while not all(dones.values()) and step < MAX_STEPS:
        current_agent = env.current_agent
        agent_location = observation["position"]
        outgoing_edges = env.road_network.edges(agent_location)
        congestion = observation["congestion"]


        action = manual_policy.act(current_location=agent_location, edges=outgoing_edges, congestion=congestion)

        observation, reward, dones, truncated, info = env.step(action)
        step = env.env_step

    total_rewards = []
    for rewards in env.rewards.values():
        total_rewards.append(sum(rewards))


    plt.bar(env.rewards.keys(), sorted(total_rewards, reverse=True))
    plt.savefig("total_reward.png")
    plt.show()

    plt.scatter(env.start_time.values(), total_rewards)
    plt.xlabel("start time")
    plt.ylabel('total reward')
    plt.savefig("total_reward_wrt_start_time.png")
    plt.show()  

    for route, congestion_per_route in env.congestion_per_route.items():

        plt.plot([i for i in range(len(congestion_per_route))], congestion_per_route, label = f"{route}")

    plt.legend()
    plt.ylabel('Congestion')
    plt.xlabel("Time")
    plt.title("Congestion per link over time")
    plt.savefig("congestion_per_link.png")
    plt.show()


