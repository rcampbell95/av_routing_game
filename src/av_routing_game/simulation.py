from av_routing_game.env import RoutingEnv

import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    env = RoutingEnv(size=2, num_agents=1)

    MAX_EDGES_PER_INTERSECTION = 4

    observation, info = env.reset()

    dones = {0: False}
    actions = {0: 0, 1: 0}

    while not all(dones.values()):
        action = np.random.randint(0, MAX_EDGES_PER_INTERSECTION * 2)
        if "congestion" in observation:
            direction = np.random.randint(0, 4)
            vehicle_type = np.argmax(observation["congestion"][direction])
            action = MAX_EDGES_PER_INTERSECTION * vehicle_type + direction

            actions[vehicle_type] += 1

        observation, reward, dones, truncated, info = env.step(action)

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


