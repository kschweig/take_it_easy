import numpy as np
from tkinter import font
import tkinter as tk


class TakeItEasy:

    def __init__(self, intermediate_reward=False, seed=None):

        self.coordinates = [(400, 82),
                            (262, 162), (538, 162),
                            (124, 242), (400, 242), (676, 242),
                            (262, 322), (538, 322),
                            (124, 402), (400, 402), (676, 402),
                            (262, 482), (538, 482),
                            (124, 562), (400, 562), (676, 562),
                            (262, 642), (538, 642),
                            (400, 722)]
        self.colors = ['lightgray', 'peachpuff', 'hotpink', 'deepskyblue', 'mediumturquoise', 'crimson', 'lightgreen', 'orange', 'yellow']
        self.intermediate_reward = intermediate_reward
        self.rng = np.random.default_rng(seed)
        self.action_space = 19
        self.obs_space = 60
        self.reset_env()

    def reset_env(self):
        self.possible = np.asarray(
            [[1, 2, 3], [1, 2, 4], [1, 2, 8], [1, 6, 3], [1, 6, 4], [1, 6, 8], [1, 7, 3], [1, 7, 4], [1, 7, 8],
             [5, 2, 3], [5, 2, 4], [5, 2, 8], [5, 6, 3], [5, 6, 4], [5, 6, 8], [5, 7, 3], [5, 7, 4], [5, 7, 8],
             [9, 2, 3], [9, 2, 4], [9, 2, 8], [9, 6, 3], [9, 6, 4], [9, 6, 8], [9, 7, 3], [9, 7, 4], [9, 7, 8]], dtype=np.uint8)
        self.gamestate = np.zeros((self.action_space, 3), dtype=np.uint8)
        self.indices = [i for i in range(27)]
        self.available = np.ones((self.action_space))
        self.old_reward = 0
        self.current_card = None
        self.done = False

    def reset(self):
        # reset env and return first state
        self.reset_env()
        self.current_card = self.possible[self.rng.choice(self.indices)]
        return create_obs(self.current_card, self.gamestate), self.available, 0, self.done

    def step(self, action):

        assert np.allclose(self.gamestate[action], 0), "Can't set cards on already occupied places!"
        self.gamestate[action] = self.current_card
        self.available[action] = 0
        if self.available.sum() == 0:
            self.done = True

        # draw next card to place
        idx = self.rng.choice(self.indices)
        self.indices.remove(idx)
        self.current_card = self.possible[idx]

        # return intermediate reward or not
        if self.intermediate_reward:
            reward = self.calc_reward()
        else:
            reward = self.evaluate() if self.done else 0

        return create_obs(self.current_card, self.gamestate), self.available, reward, self.done


    def calc_reward(self):
        reward = 0
        # first direction
        reward += self.sum_of_equals([self.gamestate[3][0], self.gamestate[8][0], self.gamestate[13][0]])
        reward += self.sum_of_equals([self.gamestate[1][0], self.gamestate[6][0], self.gamestate[11][0], self.gamestate[16][0]])
        reward += self.sum_of_equals([self.gamestate[0][0], self.gamestate[5][0], self.gamestate[9][0], self.gamestate[14][0], self.gamestate[18][0]])
        reward += self.sum_of_equals([self.gamestate[2][0], self.gamestate[7][0], self.gamestate[12][0], self.gamestate[17][0]])
        reward += self.sum_of_equals([self.gamestate[5][0], self.gamestate[10][0], self.gamestate[15][0]])
        # second direction
        reward += self.sum_of_equals([self.gamestate[0][1], self.gamestate[1][1], self.gamestate[3][1]])
        reward += self.sum_of_equals([self.gamestate[2][1], self.gamestate[4][1], self.gamestate[6][1], self.gamestate[8][1]])
        reward += self.sum_of_equals([self.gamestate[5][1], self.gamestate[7][1], self.gamestate[9][1], self.gamestate[11][1],self.gamestate[13][1]])
        reward += self.sum_of_equals([self.gamestate[10][1], self.gamestate[12][1], self.gamestate[14][1], self.gamestate[16][1]])
        reward += self.sum_of_equals([self.gamestate[15][1], self.gamestate[17][1], self.gamestate[18][1]])
        # third direction
        reward += self.sum_of_equals([self.gamestate[0][2], self.gamestate[2][2], self.gamestate[5][2]])
        reward += self.sum_of_equals([self.gamestate[1][2], self.gamestate[4][2], self.gamestate[7][2], self.gamestate[10][2]])
        reward += self.sum_of_equals([self.gamestate[3][2], self.gamestate[6][2], self.gamestate[9][2], self.gamestate[12][2],self.gamestate[15][2]])
        reward += self.sum_of_equals([self.gamestate[8][2], self.gamestate[11][2], self.gamestate[14][2], self.gamestate[17][2]])
        reward += self.sum_of_equals([self.gamestate[13][2], self.gamestate[16][2], self.gamestate[18][2]])

        ret = reward - self.old_reward
        self.old_reward = reward

        return ret

    def sum_of_equals(self, values):
        reward = 0
        first = 0
        i = 0
        for val in values:
            if val == 0:
                continue
            else:
                if first == 0:
                    first = val
                elif first != val:
                    return 0
                reward += val
                i += 1

        if i == 1:
            return 0

        return reward

    def evaluate(self):
        reward = 0
        #first direction
        if self.gamestate[3][0] == self.gamestate[8][0] and self.gamestate[3][0] == self.gamestate[13][0]:
            reward += self.gamestate[3][0] * 3
        if self.gamestate[1][0] == self.gamestate[6][0] and self.gamestate[1][0] == self.gamestate[11][0] and self.gamestate[1][0] == self.gamestate[16][0]:
            reward += self.gamestate[1][0] * 4
        if self.gamestate[0][0] == self.gamestate[4][0] and self.gamestate[0][0] == self.gamestate[9][0] and self.gamestate[0][0] == self.gamestate[14][0] and self.gamestate[0][0] == self.gamestate[18][0]:
            reward += self.gamestate[0][0] * 5
        if self.gamestate[2][0] == self.gamestate[7][0] and self.gamestate[2][0] == self.gamestate[12][0] and self.gamestate[2][0] == self.gamestate[17][0]:
            reward += self.gamestate[2][0] * 4
        if self.gamestate[5][0] == self.gamestate[10][0] and self.gamestate[5][0] == self.gamestate[15][0]:
            reward += self.gamestate[5][0] * 3
        # second direction
        if self.gamestate[0][1] == self.gamestate[1][1] and self.gamestate[0][1] == self.gamestate[3][1]:
            reward += self.gamestate[0][1] * 3
        if self.gamestate[2][1] == self.gamestate[4][1] and self.gamestate[2][1] == self.gamestate[6][1] and self.gamestate[2][1] == self.gamestate[8][1]:
            reward += self.gamestate[2][1] * 4
        if self.gamestate[5][1] == self.gamestate[7][1] and self.gamestate[5][1] == self.gamestate[9][1] and self.gamestate[5][1] == self.gamestate[11][1] and self.gamestate[5][1] == self.gamestate[13][1]:
            reward += self.gamestate[5][1] * 5
        if self.gamestate[10][1] == self.gamestate[12][1] and self.gamestate[10][1] == self.gamestate[14][1] and self.gamestate[10][1] == self.gamestate[16][1]:
            reward += self.gamestate[10][1] * 4
        if self.gamestate[15][1] == self.gamestate[17][1] and self.gamestate[15][1] == self.gamestate[18][1]:
            reward += self.gamestate[15][1] * 3
        # third direction
        if self.gamestate[0][2] == self.gamestate[2][2] and self.gamestate[0][2] == self.gamestate[5][2]:
            reward += self.gamestate[0][2] * 3
        if self.gamestate[1][2] == self.gamestate[4][2] and self.gamestate[1][2] == self.gamestate[7][2] and self.gamestate[1][2] == self.gamestate[10][2]:
            reward += self.gamestate[1][2] * 4
        if self.gamestate[3][2] == self.gamestate[6][2] and self.gamestate[3][2] == self.gamestate[9][2] and self.gamestate[3][2] == self.gamestate[12][2] and self.gamestate[3][2] == self.gamestate[15][2]:
            reward += self.gamestate[3][2] * 5
        if self.gamestate[8][2] == self.gamestate[11][2] and self.gamestate[8][2] == self.gamestate[14][2] and self.gamestate[8][2] == self.gamestate[17][2]:
            reward += self.gamestate[8][2] * 4
        if self.gamestate[13][2] == self.gamestate[16][2] and self.gamestate[13][2] == self.gamestate[18][2]:
            reward += self.gamestate[13][2] * 3

        return reward

    def set_state(self, gamestate, available, indices):
        self.gamestate = gamestate
        self.available = available
        self.indices = indices

    def get_state(self):
        return self.gamestate, self.available, self.indices

    def show_game_state(self):
        root = tk.Tk()
        root.title("Take it Easy")
        root.minsize(300, 300)
        root.geometry("800x804")
        w = tk.Canvas(root, width=800, height=804)
        w.pack()

        helv20 = font.Font(family="Helvetica", size=20, weight="bold")

        for i, state in enumerate(self.gamestate):

            fp = self.coordinates[i]  # fixpoint
            points = [fp[0] - 46, fp[1] - 80,
                      fp[0] - 92, fp[1],
                      fp[0] - 46, fp[1] + 80,
                      fp[0] + 46, fp[1] + 80,
                      fp[0] + 92, fp[1],
                      fp[0] + 46, fp[1] - 80]

            if state[0] == 0:
                w.create_polygon(points, outline='black',
                                 fill='white', width=2)
            else:
                w.create_polygon(points, outline='black',
                                 fill='white', width=2)
                stripe3 = [fp[0] + 64, fp[1] + 49, fp[0] + 74, fp[1] + 31, fp[0] - 64, fp[1] - 49, fp[0] - 74,
                           fp[1] - 31]
                w.create_polygon(stripe3, fill=self.colors[state[2]-1])
                stripe2 = [fp[0] + 64, fp[1] - 49, fp[0] + 74, fp[1] - 31, fp[0] - 64, fp[1] + 49, fp[0] - 74,
                           fp[1] + 31]
                w.create_polygon(stripe2, fill=self.colors[state[1]-1])
                stripe1 = [fp[0]-10, fp[1]-80, fp[0]+10, fp[1]-80, fp[0]+10, fp[1]+80, fp[0]-10, fp[1]+80]
                w.create_polygon(stripe1, fill=self.colors[state[0]-1])
                w.create_text(fp[0], fp[1]-40, text=f"{state[0]}", font=helv20)
                w.create_text(fp[0]-46, fp[1]+28, text=f"{state[1]}", font=helv20)
                w.create_text(fp[0]+46, fp[1]+28, text=f"{state[2]}", font=helv20)

        root.mainloop()


def create_obs(current_card, gamestate):
    """
    Concatenate current card to first position and normalize for Neural Net usage
    :param current_card:
    :param gamestate:
    :return:
    """
    gamestate = gamestate.flatten()
    obs = np.concatenate((current_card.flatten(), gamestate), axis=0).astype(float)
    return obs

def normalize(obs):
    obs -= np.mean(np.arange(1, 10))
    obs /= np.var(np.arange(1, 10))
    return obs