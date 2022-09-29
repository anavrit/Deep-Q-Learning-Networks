from unityagents import UnityEnvironment
from double_dqn_agent import DoubleDQNAgent
from dqn_agent import DQNAgent

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

def train_agent(agent, filename, num_episodes = 2000, max_iter = 1000, epsilon_start = 1.0, epsilon_decay = 0.995, epsilon_min = 0.01):
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

architectures = [[64, 64], [128, 64], [64, 64, 64], [128, 64, 32], [256, 128, 64, 32]]
doubledqn_agent_scores = []
dqn_agent_scores = []
column_names = ['64x64', '128x64', '64x64x64', '128x64x32', '256x128x64x32']
for i, hidden_layers in enumerate(architectures):
    dqn_agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, hidden_layers=hidden_layers)
    dqn_agent_scores.append(train_agent(dqn_agent, filename=f'../Trained_Weights/dqn_trained_weights_{column_names[i]}_.pth'))
    double_dqn_agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=0, hidden_layers=hidden_layers)
    doubledqn_agent_scores.append(train_agent(double_dqn_agent, filename=f'../Trained_Weights/double_dqn_trained_weights_{column_names[i]}_.pth'))

dqn = pd.DataFrame({col: dqn_agent_scores[i] for i, col in enumerate(column_names)})
dqn.to_csv('../Scores/dqn_scores.csv', index=False)
doubledqn = pd.DataFrame({col: doubledqn_agent_scores[i] for i, col in enumerate(column_names)})
doubledqn.to_csv('../Scores/doubledqn_scores.csv', index=False)

"""
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('Training a DoubleDQN Agent')
plt.savefig('../Charts/doubledqn.png', transparent=True)
"""

env.close()
