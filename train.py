import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from source.environment import TakeItEasy
from source.buffer import ReplayBuffer
from source.agents.dqn import DQN

seed = 42
batch_size = 32
buffer_size = 1000000
transitions = 10000000
show_every = 50000
train_every = 4
train_start_iter = 20000

env = TakeItEasy(seed=seed, intermediate_reward=True)
agent = DQN(env.obs_space, env.action_space, seed=seed)
buffer = ReplayBuffer(env.obs_space, buffer_size, batch_size, seed=seed)

all_rewards = []
rewards = []
values = []
done = True

for iter in tqdm(range(transitions)):
    # Reset if environment is done
    if done:
        state, available, reward, done = env.reset()

    # obtain action
    action, value = agent.policy(state, available)

    # step in environment
    next_state, occupied, reward, done = env.step(action)

    # add to buffer
    buffer.add(state, action, reward, done, next_state)

    # now next state is the new state
    state = next_state

    # update the agent periodically after initially populating the buffer
    if (iter+1) % train_every == 0 and iter > train_start_iter:
        agent.train(buffer)

    rewards.append(env.evaluate())
    values.append(value)

    if (iter+1) % show_every == 0:
        print(f"reward: "
              f"(mean) {round(np.mean(rewards), 2)} , "
              f"(min) {int(np.min(rewards))} , "
              f"(max) {int(np.max(rewards))} | "
              f"value : (mean){round(np.nanmean(values), 2):.02f} , "
              f"(min) {round(np.nanmin(values),2)} , "
              f"(max) {round(np.nanmax(values),2)}"
        )
        all_rewards.append(np.mean(rewards))
        rewards = []
        values = []


plt.figure(figsize=(8, 6))
plt.ylabel("Reward")
plt.xlabel("Training Steps")
plt.plot(range(len(all_rewards)), all_rewards)
plt.show()