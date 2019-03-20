
# coding: utf-8

# ## Pattern Recognition, Question 3
# ### Ali Mortazavi
# ### 96131044

#     Importing Libraries

# In[1]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns


# For each pair of features, we will answer to the question a, and for each features we will answer to b,c, and d.  <br>
# > a) Which features are good, and which ones are bad?
# Explain your reasons. <br>
# b) Investigate each feature in terms of linear/non-linear
# separability. <br>
# c) Investigate each feature in terms of correlation among
# the samples. <br>
# d) Investigate each feature in terms of modality between
# the samples. <br>
# 

# In[2]:


iris = datasets.load_iris()
X = iris.data 
y = iris.target
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


# In[3]:


i=0
j=1
plt.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Set3)
plt.xlabel(feature_names[i])
plt.ylabel(feature_names[j])
plt.show() 
print ("features:", feature_names[i], ",", feature_names[j])


# a) This pair seperates blue class from the others but it poorly seperates green and yellow classes. <br>
# b) Blue class is linearly seperable, while the other classes are not.<br>
# c) For blue class there is a strong positive correlation between first and second features (sepal-width and sepal-length). For the green and yellow classes, there is a slight positive correlation.
# 

# In[4]:


i=0
j=2
plt.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Set3)
plt.xlabel(feature_names[i])
plt.ylabel(feature_names[j])
plt.show() 
print ("features:", feature_names[i], ",", feature_names[j])


# a) Using this pair, Blue class is seperates completely while the other classes are also seperated roughly. <br>
# b) It is linearly separable.<br>
# c) Blue class: No correlation, green: slight positive correlation, yellow: positive correlation
# 

# In[5]:


i=0
j=3
plt.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Set3)
plt.xlabel(feature_names[i])
plt.ylabel(feature_names[j])
plt.show() 
print ("features:", feature_names[i], ",", feature_names[j])


# a) Using this pair, Blue class is seperates completely while the other classes are also seperated roughly. <br>
# b) It is linearly separable.<br>
# c) No correlation for any of the classes

# In[6]:


i=1
j=2
plt.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Set3)
plt.xlabel(feature_names[i])
plt.ylabel(feature_names[j])
plt.show() 
print ("features:", feature_names[i], ",", feature_names[j])


# a) Using this pair, Blue class is seperates completely while the other classes are also seperated roughly. <br>
# b) It is linearly separable.<br>
# c) Blue: No correlation, Green and Yellow: slight correlation

# In[7]:


i=1
j=3
plt.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Set3)
plt.xlabel(feature_names[i])
plt.ylabel(feature_names[j])
plt.show() 
print ("features:", feature_names[i], ",", feature_names[j])


# a) Using this pair, Blue class is seperates completely while the other classes are also seperated roughly. <br>
# b) It is linearly separable.<br>
# c) Blue: No correlation, Green and Yellow: slight correlation

# In[8]:


i=2
j=3
plt.scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Set3)
plt.xlabel(feature_names[i])
plt.ylabel(feature_names[j])
plt.show() 
print ("features:", feature_names[i], ",", feature_names[j])


# a) All classes are separeted with high accuracy. <br>
# b) It is linearly separable.<br>
# c) Positive correlation for all classes

# Now we will use histogram to determine whether the classes are unimodal or not. 

# In[9]:


bins = 30
A = X[y==0]
B = X[y==1]
C = X[y==2]


# #### For sepal_length

# In[10]:


i=0
print (feature_names[i])
sns.distplot(A[:, i], color="blue", bins=bins)
plt.show()
print ("Unimodal for class 0")


#     

# In[11]:


sns.distplot(B[:, i], color="orange", bins=bins)
plt.show()
print ("Multimodal for class 1")


# In[12]:


sns.distplot(C[:, i], color="green", bins=bins)
plt.show()
print ("Multimodal for class 2")


# #### For sepal_width

# In[13]:


i=1
print (feature_names[i])
sns.distplot(A[:, i], color="blue", bins=bins)
plt.show()
print ("Multimodal for class 0")


#     

# In[14]:


sns.distplot(B[:, i], color="orange", bins=bins)
plt.show()
print ("Multimodal for class 1")


# In[15]:


sns.distplot(C[:, i], color="green", bins=bins)
plt.show()
print ("Multimodal for class 2")


# #### For petal_length

# In[16]:


i=2
print (feature_names[i])
sns.distplot(A[:, i], color="blue", bins=bins)
plt.show()
print ("Unimodal for class 0")


#     

# In[17]:


sns.distplot(B[:, i], color="orange", bins=bins)
plt.show()
print ("Multimodal for class 1")


# In[18]:


sns.distplot(C[:, i], color="green", bins=bins)
plt.show()
print ("Multimodal for class 2")


# #### For petal_width

# In[19]:


i=3
print (feature_names[i])
sns.distplot(A[:, i], color="blue", bins=bins)
plt.show()
print ("Multimodal for class 0")


#     

# In[20]:


sns.distplot(B[:, i], color="orange", bins=bins)
plt.show()
print ("Multimodal for class 1")


# In[21]:


sns.distplot(C[:, i], color="green", bins=bins)
plt.show()
print ("Multimodal for class 2")


# ### Conclusion
# Based on the figure associated with pair (Petal-Lentgh, Petal-Width) I think this pair is the best one for the classification task. Because we can draw two lines to separate points accurately in different classes. 
