#! /usr/bin/env python2

from gridData import Grid
import numpy as np

ones = np.ones([4,4,4])
zeros = np.zeros([4,4,4])

opts = dict(delta=40,origin=-20*np.array((1,1,1)))
g = Grid(ones,**opts)
g.export('BrownDyn.diffusion.POT.dx')
# g.export('diff-B.dx')
g = Grid(zeros,**opts)
# g.export('pmf.dx')
