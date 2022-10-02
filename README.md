[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Project Details

For this project, I trained an agent to navigate and collect bananas in a large, square world.  The code is written in PyTorch and Python 3.6. I trained the agent on a iMac Pro (2017) with a 3.2 GHz 8-Core Intel Xeon W including 64 GB RAM. I did not use the GPU for training as it is not CUDA compatible.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Create and activate a new environment with Python 3.6

  **Linux or Mac:**<br>
  `conda create --name drlnd python=3.6` <br>
  `source activate drlnd`

  **Windows:**<br>
  `conda create --name drlnd python=3.6`<br>
  `activate drlnd`    

2. Install OpenAI gym in the environment:

  `pip install gym`

3. Clone the following repository and install the additional dependencies:

  `git clone https://github.com/anavrit/Deep-Q-Learning-Networks.git`<br>
  `cd Deep-Q-Learning-Networks`<br>
  `pip install .`

4. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

5. Move or copy the downloaded environment to the root directory of Deep-Q-Learning-Networks; and unzip the file to get `Banana.app`.

### Instructions

A brief description of files in the `Python` folder: <br>
- `dqn_agent.py`: defines a DQN agent
- `double_dqn_agent.py`: defines a Double DQN agent
- `replaybuffer.py`: class that buffers the experiences for both agents
- `model.py`: deep neural network model architecture
- `train_agent.py`: code used to train DQN and Double DQN agents over multiple architectures
- `charts.py`: code for charts showing results from `train_agent.py`
- `train_and_play_agent.py`: code for user to train and optionally play the trained agent.

#### Train agent <br>

1. Navigate to the Python directory

  `cd Python`

2. Train agent with the following command:

  `python train_and_play_agent.py`<br>

  User will be given an option for showing how well the agent plays the game.

#### Resources <br>

The following key resources can be found in the `Resources` folder:

1. `double_dqn_trained_weights_256x128x64x32_.pth`: trained weights of the best Q network
2. `My_Trained_Agent`: tracking progress of the trained agent
