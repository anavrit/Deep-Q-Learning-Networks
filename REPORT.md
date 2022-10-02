[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project #1: REPORT

### Learning Algorithm

For this project, I have trained an agent to navigate and collect bananas in a large, square world using deep reinforcement learning algorithms. Two reinforcement learning algorithms with varied deep learning architectures and hyperparameters were tested using the starter code provided by Udacity.

![Trained Agent][image1]

#### 1. Deep Q-Network (DQN)
The [Deep Q-Network] (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) was introduced by Mnih et al. in a 2015 paper titled 'Human-level control through deep reinforcement learning' in Nature. As in Markov Decision Processes (MDPs) the goal of the agent is to interact with the environment by selecting actions in a way that maximizes future rewards. At each time-step the agent selects an action from the set of legal game actions and receives a future state and reward from the environment. Using Q-learning with a deep neural network as a nonlinear function approximator, the following modifications are made: (a) *Experience replay* that stores experiences and randomizes over the data, thereby removing correlations in the observation sequence and smoothing over changes in the data distribution; and (b) *Iterative updates* that adjusts the action-values towards target values that are only periodically updated, thereby reducing correlations with the target.

The algorithm in full is shown below:

<img src="/Resources/dqn.jpg" alt="drawing" style="width:350px;"/>

#### 2. Double Deep Q-Network
The [Double Deep Q-Network] (https://arxiv.org/pdf/1509.06461.pdf) is a specific adaptation to the DQN algorithm to reduce the observed overestimations of action values in DQN and lead to better performance. The max operator in DQN uses the same values both to select and to evaluate an action. This makes it more likely to select overestimated values, resulting in overoptimistic value estimates. To prevent this, we can decouple the selection from the evaluation i.e. continue estimating the value of the greedy policy according to the current values but use the second set of weights from the target network to fairly evaluate the value of this policy. The key update step for the target is shown below:

Y<sub>t</sub><sup>DoubleDQN</sup> = R<sub>t+1</sub> + γQ(S<sub>t+1</sub>, argmax<sub>a</sub>Q(S<sub>t+1</sub>, a; θ<sub>t</sub>), θ<sub>t</sub><sup>-</sup>)

**Network Architecture**

The input to the network is equivalent to the state size of the game. The final network architecture selected, after testing five different architectures (see Plot of Rewards), is a double DQN with four fully-connected hidden layers of 256, 128, 64 and 32 units, in sequence. All these layers are separated by Rectifier Linear Units (ReLu). Finally, a fully-connected linear layer projects to the output of the network, i.e., the Q-values for each of the four actions. The optimization employed to train the network is Adam.

**Hyper-parameters**

After testing hyperparameters for batch size and learning rate, the following set of hyperparameters were used for all architectures of deep neural networks, including the selected network:

BUFFER_SIZE = int(1e5)  # replay buffer size <br>
BATCH_SIZE = 64         # minibatch size <br>
GAMMA = 0.99            # discount factor <br>
TAU = 1e-3              # for soft update of target parameters <br>
LR = 1e-4               # learning rate <br>
UPDATE_EVERY = 4        # how often to update the network <br>

### Plot of Rewards

**Comparing all architectures tested**

![Comparing all architectures](/Resources/Comparing_All_Architectures.jpg)

**Comparing the best architecture for DQN and Double DQN**

![Comparing DQN and Double DQN](/Resources/Comparing_Best_DQN_DoubleDQN.jpg)

### Ideas for Future Work

A number of ideas have been proposed to improve on the performance of Double DQN. A couple of key ideas for future work are:

1. [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf) improves on experience replay by prioritizing experience, so as to replay important transitions more frequently, and therefore learn more efficiently. <br><br>
2. [Dueling Network Architectures](https://arxiv.org/pdf/1511.06581.pdf) represents two separate estimators: one for the state value function and one for the state-dependent action advantage function. The main benefit of this factoring is to generalize learning across actions without imposing any change to the underlying reinforcement learning algorithm.
