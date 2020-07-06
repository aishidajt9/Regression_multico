import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
import matplotlib.animation as animation

# fixing beta as 0.8
sigma_x = 0.01
mean = [0, 0]
cov = [[sigma_x, 0.8 * sigma_x], [0.8 * sigma_x, 1]]

fig = plt.figure(figsize=(5, 5))
plt.xlabel("x")
plt.ylabel("y")

def plot(t):
    plt.cla()
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    model = LinearRegression(fit_intercept=True)
    model.fit(x[:, np.newaxis], y)
    xfit = np.linspace(-3, 3, 1000)
    yfit = model.predict(xfit[:, np.newaxis])
    im1 = plt.scatter(x, y)
    im2 = plt.plot(xfit, yfit, c="red")
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

ani = animation.FuncAnimation(fig, plot, interval=100)
plt.show()
# w = animation.PillowWriter(fps=5)
# ani.save('Regression_multico_single.gif', writer=w)