import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import matplotlib.cm as cm

norm = Normalize(vmin=0, vmax=2*np.pi)
colormap = cm.hsv

n = 10

X = range(0,n)
Y = range(0,n)
X, Y = np.meshgrid(X,Y)

Theta = np.arctan2(Y,X)
t = Theta + np.pi/2
t %= 2*np.pi

def pol2vec(s):
    return np.cos(s), np.sin(s)

U,V = pol2vec(t)

fig, ax = plt.subplots()
Q = ax.quiver(X,Y,U,V,norm(t), angles='uv', pivot='mid', cmap=colormap)

def update(frame,Q,t):
	t += 0.1
	t %= 2*np.pi
	U,V = pol2vec(t)
	C = norm(t)
	Q.set_UVC(U,V,C)

ani = FuncAnimation(fig, update, interval=1, frames=range(100), repeat=True, blit=False, fargs=(Q,t))
plt.show()