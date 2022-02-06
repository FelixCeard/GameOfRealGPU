import cupy as np
import numpy

from vispy.util.transforms import ortho
from vispy import gloo
from vispy import app
import time

VERT_SHADER = """
// Uniforms
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_antialias;

// Attributes
attribute vec2 a_position;
attribute vec2 a_texcoord;

// Varyings
varying vec2 v_texcoord;

// Main
void main (void)
{
    v_texcoord = a_texcoord;
    gl_Position = u_projection * u_view * u_model * vec4(a_position,0.0,1.0);
}
"""

FRAG_SHADER = """
uniform sampler2D u_texture;
varying vec2 v_texcoord;
void main()
{
    gl_FragColor = texture2D(u_texture, v_texcoord);
    gl_FragColor.a = 1.0;
}

"""


class Canvas(app.Canvas):

    def __init__(self, W, H):
        self.width = W
        self.height = H

        # A simple texture quad
        self.data = numpy.zeros(4, dtype=[('a_position', numpy.float32, 2), ('a_texcoord', numpy.float32, 2)])
        self.data['a_position'] = numpy.array([[0, 0], [W, 0], [0, H], [W, H]])
        self.data['a_texcoord'] = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # Image to be displayed
        self.img_array = np.random.uniform(0, 1, (W, H)).astype(np.float32)

        app.Canvas.__init__(self, keys='interactive', size=((W * 5), (H * 5)))

        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.texture = gloo.Texture2D(self.img_array.get(), interpolation='linear')

        self.program['u_texture'] = self.texture
        self.program.bind(gloo.VertexBuffer(self.data))

        self.view = numpy.eye(4, dtype=np.float32)
        self.model = numpy.eye(4, dtype=np.float32)
        self.projection = numpy.eye(4, dtype=np.float32)

        self.program['u_model'] = self.model
        self.program['u_view'] = self.view
        self.projection = ortho(0, W, 0, H, -1, 1)
        self.program['u_projection'] = self.projection

        gloo.set_clear_color('white')

        self._timer = app.Timer('auto', connect=self.update, start=True)

        self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)
        self.projection = ortho(0, width, 0, height, -100, 100)
        self.program['u_projection'] = self.projection

        # Compute thje new size of the quad
        r = width / float(height)
        R = self.width / float(self.height)
        if r < R:
            w, h = width, width / R
            x, y = 0, int((height - h) / 2)
        else:
            w, h = height * R, height
            x, y = int((width - w) / 2), 0
        self.data['a_position'] = numpy.array(
            [[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
        self.program.bind(gloo.VertexBuffer(self.data))

    def setData(self, data):
        self.img_array = data
        self.texture.set_data(self.img_array)

    def on_draw(self, event):
        gloo.clear(color=True, depth=True)
        # self.img_array[...] = np.random.uniform(0, 1, (self.width, self.height)).astype(np.float32)
        self.program.draw('triangle_strip')


if __name__ == '__main__':
    canvas = Canvas(100, 100)
    app.run()
