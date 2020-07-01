import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

r_x1x2 = 0.2
mean3 = [0, 0, 0]
cov3 = [[1, r_x1x2, 0.8], [r_x1x2, 1, 0.7], [0.8, 0.7, 1]]

x1 = []
x2 = []
y = []
xs = []
ys = []
zs = []

for i in range(100):
    data = np.random.multivariate_normal(mean3, cov3, 100)
    x1.append(data[:,0].flatten())
    x2.append(data[:,1].flatten())
    y.append(data[:,2].flatten())
    model = LinearRegression(fit_intercept=True)
    model.fit(data[:,0:2], data[:,2])
    coefs = model.coef_
    intercept = model.intercept_
    xp = np.tile(np.linspace(-3,3), (50,1)).flatten()
    yp = np.tile(np.linspace(-3,3), (50,1)).T.flatten()
    zp = xp*coefs[0]+yp*coefs[1]+intercept
    xs.append(xp)
    ys.append(yp)
    zs.append(zp)

fig = plt.figure(figsize=(6, 5))
ax = Axes3D(fig)
# ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")

scat1, = ax.plot([],[],[],lw=0,marker="o",color="blue")
scat2, = ax.plot([],[],[],lw=0,marker="o",color="red",alpha=0.1)


def plot2(t, x1, x2, y, xs, ys, zs):
    scat1.set_data(x1[t], x2[t])
    scat1.set_3d_properties(y[t])
    scat2.set_data(xs[t], ys[t])
    scat2.set_3d_properties(zs[t])

ani = animation.FuncAnimation(fig, plot2,
                              fargs=(x1, x2, y, xs, ys, zs),
                              interval=100)

plt.show()