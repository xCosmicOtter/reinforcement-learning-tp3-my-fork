"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        agent.update(s,a,r,next_s)
        s = next_s
        total_reward += r
        if done:
            break

    return total_reward


rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

# assert np.mean(rewards[-100:]) > 0.0
# DONE: créer des vidéos de l'agent en action
def record_agent_performance(agent, env, name_prefix = "agent",number_generation: int = 5, video_folder: str = "videos"):
    env = RecordVideo(env, video_folder=video_folder, name_prefix=name_prefix, episode_trigger=lambda x: True)
    env = RecordEpisodeStatistics(env)

    for _ in range(number_generation):
        s, _ = env.reset()
        episode_over = False
        while not episode_over:
            action = agent.get_best_action(s)
            s, reward, terminated, truncated, _ = env.step(action)
            episode_over = terminated or truncated

    env.close()
record_agent_performance(agent,env,"QLearningAgent",1)
#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
record_agent_performance(agent,env,"QLearningAgentEpsScheduling",1)


####################
# 3. Play with SARSA
####################


agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)), epsilon=0.25)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))
record_agent_performance(agent,env,"SARSA",1)



print("\n ==== STATISTIC ====")
import matplotlib.pyplot as plt
import time

def run_trials(agent, n_trials=50, n_episodes=100):
    rewards_history = []
    learning_times = []

    for _ in range(n_trials):
        rewards = []
        start_time = time.time()

        for _ in range(n_episodes):
            rewards.append(play_and_train(env, agent))

        learning_time = time.time() - start_time
        rewards_history.append(rewards)
        learning_times.append(learning_time)

    return rewards_history, learning_times


def compare_agents():
    agents = [QLearningAgent(learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))), 
              QLearningAgentEpsScheduling(learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))),
                SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)), epsilon=0.25)]
    agents_name = ["QLearningAgent", "QLearningAgentEpsScheduling", "SarsaAgent"]
    colors = ['r', 'g', 'b']
    all_rewards = []
    all_times = []

    for agent in agents:
        rewards, time = run_trials(agent)
        all_rewards.append(rewards)
        all_times.append(time)

    plt.figure(figsize=(12, 6))
    for i, (_, rewards) in enumerate(zip(agents, all_rewards)):
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)
        episodes = range(1, len(mean_rewards) + 1)
        plt.plot(episodes, mean_rewards, label=agents_name[i], color=colors[i])
        plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, color=colors[i])

    plt.xlabel('Epochs')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward by Epochs for Different Agents')
    plt.legend()
    plt.savefig('figures/comparison_curves.png')
    plt.close()

    print("\nPerformance Statistics:")
    for i, _ in enumerate(agents):
        final_rewards = [r[-100:] for r in all_rewards[i]]
        mean_final_reward = np.mean(final_rewards)
        std_final_reward = np.std(final_rewards)
        mean_learning_time = np.mean(all_times[i])
        std_learning_time = np.std(all_times[i])

        print(f"\n{agents_name[i]}:")
        print(f"-> Final Mean Reward: {mean_final_reward:.5f} ± {std_final_reward:.5f}")
        print(f"-> Learning Time: {mean_learning_time:.5f}s ± {std_learning_time:.5f}s")

compare_agents()