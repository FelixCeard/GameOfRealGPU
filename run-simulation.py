import cupy as np
import numpy
from vispy import gloo, app

from SmoothLife import SmoothLife
from plot import Canvas
import time


class GameOfReal(Canvas):
    def __init__(self, W, H):
        self.sl = SmoothLife(W, H)
        self.sl.add_speckles()

        # self.sl.addCenteredRectangle()
        # self.texture.set_data(np.float32(self.sl.step()))

        self.is_generating = False
        super().__init__(W, H)


    def step(self):


        if not self.is_generating:
            start = time.time()
            self.is_generating = True
            # self.is_generating += 1

            # self.texture.set_data(np.float32(self.sl.field))
            # print((self.sl.step()))

            self.texture.set_data(numpy.float32((self.sl.step())))
            # print("updated")
            end = time.time()
            print(int((end - start)*1000), 'ms')
            self.is_generating = False


    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.step()
        self.program.draw('triangle_strip')

if __name__ == '__main__':
    GoR = GameOfReal(1000, 1000)
    app.run()