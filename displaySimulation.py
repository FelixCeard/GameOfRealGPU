import cupy as np
import numpy
from vispy import gloo, app

from SmoothLife import SmoothLife
from plot import Canvas
import time


class GameOfReal(Canvas):
    def __init__(self, W, H):
        self.sl = SmoothLife(W, H)
        # self.sl.add_speckles()

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

            self.texture.set_data(numpy.float32((self.sl.step()).get()))
            # print("updated")
            end = time.time()
            print(int((end - start) * 1000), 'ms')
            self.is_generating = False

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        self.step()
        self.program.draw('triangle_strip')


class GameOfRealSimulation:
    def __init__(self, W, H, input_field, steps=100):
        self.input_field = input_field
        self.sl = SmoothLife(W, H)

        self.width = W
        self.height = H

        self.steps = steps

        self.addInputField()

    def getBoard(self):
        return self.sl.field.get()

    def addInputField(self):
        size = self.input_field.shape

        center_x = self.width // 2
        center_y = self.height // 2

        self.sl.field[(center_x - size[0] // 2): (center_x + size[0] // 2),
        (center_y - size[1] // 2): (center_y + size[1] // 2)] = self.input_field

    def run(self):
        for _ in range(self.steps):
            self.sl.step()

    def evalField(self):
        # return self._evalSum()
        # return self._evalSpanX()
        # return self._evalArea()
        self.score =  self._evalArea() + self._evalSum() + self._evalSpanX()
        return self.score

    def _evalSum(self):
        self.score = self.sl.field.sum()
        return self.score

    def _evalSpanX(self):
        """
        Give the range of the most left and the most right Item
        :return:
        """
        threshold = 0.1
        arr = self.sl.field.get()

        x1 = -1
        x2 = -2

        x = 0
        for col in arr.transpose():
            for e in col:
                if e > threshold:
                    if x1 == -1:
                        x1 = x
                    else:
                        x2 = x
            x += 1

        self.score = x2 - x1 + 1
        return self.score

    def _evalSpanY(self):
        """
        Give the range of the most left and the most right Item
        :return:
        """
        threshold = 0.1
        arr = self.sl.field.get()

        x1 = -1
        x2 = -2

        y = 0
        for row in arr:
            for e in row:
                if e > threshold:
                    if x1 == -1:
                        x1 = y
                    else:
                        x2 = y
            y += 1

        self.score = x2 - x1 + 1
        return self.score

    def _evalArea(self):
        """
        computes the bounding box and then returns the area of the box
        :return:
        """

        # length
        l = self._evalSpanX()

        # height
        h = self._evalSpanY()

        area = l * h
        self.score = area
        return area


if __name__ == '__main__':
    inputs = np.random.random((10, 10))
    sim = GameOfRealSimulation(20, 20, inputs)

    sim.evalField()

# if __name__ == '__main__':
#     GoR = GameOfReal(3000, 2000)
#     app.run()
