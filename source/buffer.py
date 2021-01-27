import numpy as np
import torch


class ReplayBuffer():

    def __init__(self, obs_space, buffer_size, batch_size, seed=None):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rng = np.random.default_rng(seed=seed)

        self.idx = 0
        self.current_size = 0

        self.state = np.zeros((self.buffer_size, obs_space), dtype=np.uint8)
        self.next_state = np.zeros((self.buffer_size, obs_space), dtype=np.uint8)
        self.action = np.zeros((self.buffer_size, 1), dtype=np.uint8)
        self.reward = np.zeros((self.buffer_size, 1))
        self.not_done = np.zeros((self.buffer_size, 1))

    def add(self, state, action, reward, done, next_state):

        self.state[self.idx] = state
        self.next_state[self.idx] = next_state
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.not_done[self.idx] = 1. - float(done)

        self.idx = (self.idx + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self):
        ind = self.rng.integers(0, self.current_size, size=self.batch_size)

        return (torch.ByteTensor(self.state[ind]).to(self.device).float(),
                torch.LongTensor(self.action[ind]).to(self.device),
                torch.ByteTensor(self.next_state[ind]).to(self.device).float(),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
                )

