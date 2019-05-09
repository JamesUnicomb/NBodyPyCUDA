import numpy as np
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent

blocks     = 2**4
block_size = 2**10
n          = blocks * block_size
timestep   = 1e-4

code   = open('nbody.cu', 'r').read()
mod    = SourceModule(code)
func   = mod.get_function('update')

# initialise random normal(0,1) starting position and 
# chi-square(1) mass.
X0    =  np.random.randn(n,3)
X0    *= 1.0 / np.linalg.norm(X0, axis=1).reshape(-1,1)
rho   =  np.power(np.random.uniform(0.0, 1.0, size=(n,1)), 1.0 / 3.0)
 
pos_  = np.concatenate([rho * X0,
                        np.square(np.random.randn(n,1))], axis=1).astype(np.float32)

# initialise random normal(0,1) starting velocity. 
vel_ = np.zeros((n,3)).astype(np.float32)

# initalise update vectors
pos = np.zeros((n,4)).astype(np.float32)
vel = np.zeros((n,3)).astype(np.float32)


# initialise the opengl app
app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.showMaximized()

sp2 = gl.GLScatterPlotItem(pos=pos_[:,:3], size=pos_[:,3])
w.addItem(sp2)

def update():
    global pos, vel, pos_, vel_
    func(drv.Out(pos), drv.Out(vel), drv.In(pos_), drv.In(vel_), np.int32(n), np.float32(timestep), grid=(blocks,1), block=(block_size,1,1))

    pos_ = pos
    vel_ = vel

    sp2.setData(pos=pos_[:,:3])

# initialise the qt timer
t = QtCore.QTimer()
t.timeout.connect(update)
t.start(50)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

