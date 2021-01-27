import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from source.environment import TakeItEasy
from duelling_dqn_agent import DDQNAgent

# utility to crate observation space
def create_obs(current_card, gamestate):
    gamestate = gamestate.flatten()
    obs = np.concatenate((current_card, gamestate)).astype(float)
    # Normalize
    obs -= 5.0
    obs /= 2.581988897471611
    return obs


env = Environment()
agent = DDQNAgent(60, 19)

epochs = 10000
show_every = 10


done = False
env.reset()
rewards = []
values = []
entropies = []

for ep in tqdm(range(epochs)):
    current_card, gamestate, occupied, reward, done = env.step()
    obs = create_obs(current_card, gamestate)

    while not done:
        # obtain action
        action, value, entropy = agent.policy(obs, occupied)

        #to see if we have diverging value estimations
        if value != None:
            values.append(value)
            entropies.append(entropy)

        # push action to environment
        env.set_card(action, current_card)

        # step in environment
        current_card, gamestate, occupied, reward, done = env.step()
        new_obs = create_obs(current_card, gamestate)

        # update agent
        agent.update(new_obs, action, reward, obs, done)
        obs = new_obs

    rewards.append(env.evaluate())
    if ep % show_every == 0:
        print(f"reward in epoch {ep} ({round(agent.determinacy(), 2)}%) : "
              f"(mean) {int(round(np.mean(rewards[-show_every:]), 0))} , "
              f"(min) {int(round(np.min(rewards[-show_every:]), 0))} , "
              f"(max) {int(round(np.max(rewards[-show_every:]), 0))} | "
              f"mean value: {round(np.mean(values), 2):.02f} | "
              f"mean entropy: {round(np.mean(entropies), 2):.02f}")
        values = []
        entropies = []

    # do not reset last game
    if ep < epochs - 1:
        env.reset()


plt.figure(figsize=(12, 6))
plt.plot(range(len(rewards)), rewards)
plt.show()

env.show_game_state()