import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import multivariate_normal

#
#

#Parameters to set
mu_x = 0
variance_x = 2

mu_y = 0
variance_y = 3

x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X,Y = np.meshgrid(x,y)
cov = [-0.99, -0.5, 0.5, 0.99]
pos = np.array([X.flatten(),Y.flatten()]).T

print (pos)
t = cov[0]
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 2*t], [2*t, variance_y]])


fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)
ax0.contour(rv.pdf(pos).reshape(500,500))



plt.show()

#
#
# N = 100000
#
# mean_a = [0, 0]
# cov_a = [[2, 1], [1, 2]]
#
# Xa = np.random.multivariate_normal(mean_a, cov_a, N)
# fig, ax3 = plt.subplots(nrows=1,ncols=1,figsize=(15,8))
#
# (counts, x_bins, y_bins) = np.histogram2d(Xa[:, 0], Xa[:, 1], bins=100)
# ax3.contourf(counts, extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]])
# print (x_bins)
# plt.show()