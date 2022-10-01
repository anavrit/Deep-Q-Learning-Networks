import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

dqn = pd.read_csv('../Scores/dqn_scores.csv', header=0)
ddqn = pd.read_csv('../Scores/doubledqn_scores.csv', header=0)

fig, ax = plt.subplots(1, 2, figsize=(18, 9), sharey=True)
for col in range(len(dqn.columns)):
    dqn_100 = pd.Series([np.mean(dqn.iloc[i-100:i, col]) for i, _ in dqn.iloc[100:].iterrows()])
    ddqn_100 = pd.Series([np.mean(ddqn.iloc[i-100:i, col]) for i, _ in ddqn.iloc[100:].iterrows()])
    dqn_100.index += 100
    ddqn_100.index += 100
    ax[0].plot(dqn_100, label=dqn.columns[col])
    ax[1].plot(ddqn_100, label=ddqn.columns[col])
ax[0].set_title('Performance of different architectures of DQN')
ax[0].set_ylabel('Average score over 100 episodes')
ax[1].set_title('Performance of different architectures of Double DQN')
for x in ax.flat:
    x.set(xlabel='Episode #')
    x.grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
    x.axhline(y = 13.0, color = 'red', linestyle = '--', linewidth=1)
plt.subplots_adjust(wspace=0)
plt.legend(loc='lower right')
plt.savefig('../Resources/Comparing_All_Architectures.jpg', bbox_inches='tight')

col = 4
fig = plt.figure(figsize=(8,6))
dqn_100 = pd.Series([np.mean(dqn.iloc[i-100:i, col]) for i, _ in dqn.iloc[100:].iterrows()])
ddqn_100 = pd.Series([np.mean(ddqn.iloc[i-100:i, col]) for i, _ in ddqn.iloc[100:].iterrows()])
dqn_100.index += 100
ddqn_100.index += 100
dqn_100.plot(label='DQN')
ddqn_100.plot(label='DDQN')
plt.title('Comparing DQN and Double DQN over 256x128x64x32')
plt.xlabel('Episode #')
plt.ylabel('Average score over 100 episodes')
plt.legend(loc='lower right')
plt.grid(color='gray', linestyle='--', linewidth=0.5, axis='y')
plt.axhline(y=13.0, linestyle='--', linewidth=1, color='red')
plt.savefig('../Resources/Comparing_Best_DQN_DoubleDQN.jpg', bbox_inches='tight')
