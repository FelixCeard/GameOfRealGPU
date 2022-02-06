import numpy
from vispy import gloo, app

from displaySimulation import GameOfRealSimulation
from plot import Canvas
import cupy as np
import time


class DisplayStart(Canvas):
    def __init__(self, W, H):

        path = "./configuations/" + "59455be4-09a5-49bb-a72e-88333ca544f3__64__3542.242670601356"
        starting_board = np.load(path)

        self.sl = GameOfRealSimulation(W, H, starting_board)

        self.is_generating = False
        self.n = 0
        super().__init__(W, H)
        self.t = time.time()
        self._fps = 1

    def step(self):
        if (not self.is_generating or self.n < 4) and time.time() - self.t > 1/self._fps:
            self.t = time.time()
            self.is_generating = True
            self.texture.set_data(numpy.float32((self.sl.sl.step()).get()))

    def on_draw(self, event):

        gloo.clear(color=True, depth=True)
        self.step()
        self.program.draw('triangle_strip')

if __name__ == '__main__':
    display = DisplayStart(1000, 1000)
    app.run()