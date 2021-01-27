from source.environment import TakeItEasy
from source.agents.random import Random

seed=42
# create and reset env
env = TakeItEasy(seed=seed)
# Random Policy
agent = Random(env.obs_space, env.action_space, seed=seed)


state, available, reward, done = env.reset()

i = 0

print("Step ", i)
print(state)
print(available)
print(reward)
print(done)
print("-"*100)

while not done:
    # obtain action
    action = agent.policy(state, available)

    # step in environment
    next_state, occupied, reward, done = env.step(action)

    # here, one would update the agent
    # agent.train(None)

    # after training, this is the new state
    state = next_state

    print("Step ", i+1)
    print(state)
    print(action)
    print(available, int(available.sum()))
    print(reward)
    print(done)
    print("-" * 100)

    i += 1


# print final reward
print("Reward for random agent: ", reward)
# show final game state
env.show_game_state()