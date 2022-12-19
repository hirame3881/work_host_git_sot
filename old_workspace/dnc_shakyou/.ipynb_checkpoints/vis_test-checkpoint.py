import visdom
import numpy as np
vis = visdom.Visdom()
#vis.text('Hello, world!')
vis.image(np.ones((10, 10)))