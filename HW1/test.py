import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import multivariate_normal
#
# X,Y = np.mgrid[-10:10.1:0.1, -10:10.1:0.1]
#
# xy = np.vstack((X.flatten(), Y.flatten())).T
# xlist = np.linspace(-3.0, 3.0, 100)
# print (xlist)
# import sys
#
# x = np.arange(1, 10)
# y = x.reshape(-1, 1)
# h = x * y
#
# cs = plt.contour(h, levels=[10, 30, 50],
#     colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
# cs.cmap.set_over('red')
# cs.cmap.set_under('blue')
# cs.changed()
# plt.show()
# sys.exit(0)
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
xlist = np.linspace(-10, 10, 1000)
ylist = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(xlist, ylist)

Z = np.sqrt(X**2 + Y**2)
def d_squared_function(x_1, x_2, matrix):
    a=2
    inverse_matrix =  inv(matrix)




#
# contour = plt.contour(X, Y, Z, levels=[1,3,4])
# plt.gca().set_aspect("equal")


X,Y = np.mgrid[-10:10.1:0.1, -10:10.1:0.1]
xy = np.vstack((X.flatten(), Y.flatten())).T

# matrix = np.array([[2,0],[0,2]])
# a = xy
import sys


# a = np.array([[1,2],[0,0],[5,5]])
Sigma = np.array([[1,0],[0,1]])
mu = np.array([0,0])# np.array([2,3])
def d_squared (xy, sigma, mu):
    matrix = inv(sigma)
    a = xy - mu
    t = np.matmul(a,matrix)
    # print (np.matmul(t, a.T))
    return np.matmul(t, a.T)

# d2_squared_values = np.apply_along_axis(func1d = d_squared, axis= 1, arr = xy, sigma=Sigma, mu = np.array([2,3]))
# plt.contour(X, Y, d2_squared_values)
# plt.show()
# print (d2_squared_values)

# def f(x, y):
#     return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
# x = np.linspace(0, 5, 50)
# y = np.linspace(0, 5, 40)
#
# X, Y = np.meshgrid(x, y)
#
# Z = f(X, Y)
Z = np.apply_along_axis(func1d = d_squared, axis= 1, arr = xy, sigma=Sigma, mu = mu)
Z = np.reshape(Z, (len(X), -1))
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.contour(X,Y,Z, levels=[4,9,16])
plt.show()

