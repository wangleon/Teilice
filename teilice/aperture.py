import numpy as np

class Aperture(object):
    def __init__(self, name):
        self.name = name
        if name=='1x1':
            self.shape = (1, 1)
            self.data = np.array([
                [1],
                ])
            self.center = (0, 0)
        if name=='3x3':
            self.shape = (3, 3)
            self.data = np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                ])
            self.center = (1, 1)
        elif name=='5x5':
            self.shape = (5, 5)
            self.data = np.array([
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                ])
            self.center = (2, 2)
        elif name=='5x5-4':
            self.shape = (5, 5)
            self.data = np.array([
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                ])
            self.center = (2, 2)
