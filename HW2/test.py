import numpy as np
import matplotlib.pyplot as plt
a = np.array([2,4])
b = np.array([7,8])
a = np.arange(9).reshape(3,3)
# print (a)
# b = a[:,[1,2]]
# print (b)
for i in range(0, 3):
    for j in range(0,3):
        if j==1:
            break
    print ("Hello")
