from unityagents import UnityEnvironment
from double_dqn_agent import DoubleDQNAgent

import torch
from collections import namedtuple, deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

env = UnityEnvironment(file_name="../Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)

def train_agent(agent, filename, num_episodes = 2000, max_iter = 300, epsilon_start = 1.0, epsilon_decay = 0.995, epsilon_min = 0.01):
    scores = []
    scores_window = deque(maxlen=100)
    epsilon = epsilon_start
    best_mean_score = 13
    for episode in range(1, num_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        for i in range(max_iter):
            action = agent.act(state, epsilon)             # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break
        epsilon = max(epsilon_decay*epsilon, epsilon_min)
        scores_window.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        if len(scores) >= 100:
            mean_score = np.mean(scores[-100:])
            if mean_score > best_mean_score:
                torch.save(agent.qnetwork_local.state_dict(), filename)
                best_mean_score = mean_score
    return scores

hidden_layers = [256, 128, 64, 32]
filename = '../Resources/double_dqn_trained_weights_256x128x64x32_.pth'
agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=0, hidden_layers=hidden_layers)
scores = train_agent(agent, filename=filename, num_episodes = 1200)

# Saving plot of average scores over 100 episodes
fig = plt.figure(figsize=(8,6))
avg_100 = pd.Series([np.mean(scores[i-100:i]) for i in range(100, len(scores)+1)])
avg_100.index += 100
avg_100.plot()
plt.title('Tracking performance of RL agent')
plt.xlabel('Episode #')
plt.ylabel('Average score over 100 episodes')
plt.grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
plt.axhline(y=13.0, linestyle='--', linewidth=1.0, color='red')
plt.savefig('../Resources/My_Trained_Agent.jpg', bbox_inches='tight')

# Game play
play = input('Would you like to play the trained agent (Y/N)? ')

while play in ["y", "Y"]:
    trained_weights = torch.load(filename)
    agent.qnetwork_local.load_state_dict(trained_weights)
    agent.qnetwork_target.load_state_dict(trained_weights)
    env_info = env.reset(train_mode=False)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        agent.step(state, action, reward, next_state, done)
        score += reward
        state = next_state
        if done:
            break
    print('Score =', score)
    play = input('Play again (Y/N)? ')
    if play not in ["y", "Y"]:
        break

env.close()
