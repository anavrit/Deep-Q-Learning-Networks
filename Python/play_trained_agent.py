from unityagents import UnityEnvironment
from double_dqn_agent import DoubleDQNAgent
import torch

env = UnityEnvironment(file_name = '../Banana.app', no_graphics=False)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name]
action_size = brain.vector_action_space_size
state_size = brain.vector_observation_space_size

agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=0, hidden_layers=[64, 64, 64])

# Load trained weights
trained_weights = torch.load('../Trained_Weights/doubledqn_trained_weights_64x64x64_.pth')
agent.qnetwork_local.load_state_dict(trained_weights)
agent.qnetwork_target.load_state_dict(trained_weights)

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
env.close()
