import numpy as np
import matplotlib.pyplot as plt
a = np.array([2,4])
b = np.array([7,8])

x = np.arange(-10, 10, 0.1)
y = np.zeros(len(x))
dist = a- b
dist = np.dot(dist,dist)
print (dist)
a1=a[0]
a2 = a[1]
b1 = b[0]
b2 = b[1]


for i in range(0,  len(x)):
   y[i]= a2+ (1/(b2-a2))*((dist/2)-(b1-a1)*(x[i]-a1))

plt.plot(x,y)
plt.scatter(a[0],a[1],c='g')
plt.scatter(b[0],b[1],c='b')
plt.gca().set_aspect('equal')
plt.show()

# print (a[:,2])
# print (b)