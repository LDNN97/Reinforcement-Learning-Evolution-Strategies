import numpy as np
import time
import sys
import tkinter as tk

UNIT = 40
MAZE_H = 4
MAZE_W = 4


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze,self).__init__()
        self.action_space = ['u', 'r', 'd', 'l']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

        # create grids
        for r in range(0, MAZE_H):
            x0, y0, x1, y1 = r * UNIT, 0, r * UNIT, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for c in range(0, MAZE_W):
            x0, y0, x1, y1 = 0, c * UNIT, MAZE_H * UNIT, c * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell1
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black'
        )
        # hell2
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black'
        )

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow'
        )

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red'
        )
        self.render()
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 1:
            if s[1] < (MAZE_W - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:
            if s[0] < (MAZE_H - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:
            if s[1] > UNIT:
                base_action[1] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])

        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        self.render()
        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def test():
    s = env.reset()
    for i in range(10):
        a = 1
        s, r, done = env.step(a)
        if done:
            break


if __name__ == '__main__':
    env = Maze()
    env.after(1000, test)
    env.mainloop()
