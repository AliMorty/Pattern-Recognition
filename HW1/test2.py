import matplotlib.pyplot as plt
import numpy as np

a = np.array([[1,2],[3,4]])
b = np.array([1,2])
c = np.array([[1],[2]])

print (np.matmul(b,a))
print(np.matmul(c.T,a))
# fig, ax = plt.subplots()
#
# # The "clip_on" here specifies that we _don't_ want to clip the line
# # to the extent of the axes
# ax.plot([0,1],[0,2])
#
# plt.show()