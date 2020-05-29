#!/usr/bin/env python
# coding: utf-8

# # Email Spam filter
# 
# ### Abstract
# 
# ### Dependencies
# 
# ### DataSet
# This project used the Dataset downloaded from "https://archive.ics.uci.edu/ml/datasets/Spambase" which refers to sample Project 3 in MSA project AI & Advanced Analysis (A ML model to identify emails are spam or not - Dataset)

# In[1]:


import pandas as pd
import numpy as np

result = {
    'Model': [],
    'Accuracy': []
}

# loading dataset files
dataset = pd.read_csv("dataset/spambase.data", header = None)


# In[2]:


dataset.head()


# In[3]:


dataset.shape


# In[4]:


# checking for missing values
dataset.isnull().sum()


# The output of above cell is all zero, which indicates that the dataset has no missing values and do not need data cleaning.

# Next, I am spliting the dataset into train dataset and test dataset. I've set the train test ratio to be 80/20, which is a very commonly used ratio number.

# In[5]:


from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(dataset.drop(dataset.columns[-1],1), dataset[57], train_size = 0.8, test_size = 0.2, random_state = 4)


# Since the index is random after splitting, I need to reset all the indexes.

# In[6]:


train_X = train_X.reset_index(drop = True)
train_Y = train_Y.reset_index(drop = True)
test_X = test_X.reset_index(drop = True)
test_Y = test_Y.reset_index(drop = True)


# After resetting index, the training and testing set are ready to use for SVM and Naive-Bayes models

# Training and testing SVM model using three different kernel

# In[7]:


from sklearn import svm

# svm model using linear model
svm_linear = svm.SVC(kernel = "linear", gamma = 'auto')
svm_linear.fit(train_X, train_Y)
svm_linear_acc = svm_linear.score(test_X, test_Y)
result['Model'].append("linear")
result["Accuracy"].append(svm_linear_acc)

# In[ ]:


# svm model using polynomial kernel
svm_poly = svm.SVC(kernel = "poly")
svm_poly.fit(train_X, train_Y)
svm_poly_acc = svm_poly.score(test_X, test_Y)
result['Model'].append("poly")
result["Accuracy"].append(svm_poly_acc)

# In[ ]:


# svm model using sigmodi model
svm_sigmoid = svm.SVC(kernel = "sigmoid")
svm_sigmoid.fit(train_X, train_Y)
svm_sigmoid = svm_sigmoid.score(test_X, test_Y)
result['Model'].append("sigmoid")
result["Accuracy"].append(svm_sigmoid)


# Training and testing two different Naive-Bayes model

# In[9]:


from sklearn.naive_bayes import MultinomialNB

# Multinomial Naive-Bayes model
multi_nb = MultinomialNB().fit(train_X, train_Y)
multi_nb_acc = multi_nb.score(test_X, test_Y)
result['Model'].append("multinomial")
result["Accuracy"].append(multi_nb_acc)

# In[ ]:


from sklearn.naive_bayes import GaussianNB

# Gaussian Naive-Bayes model
gaussian_nb = GaussianNB().fit(train_X, train_Y)
gaussian_nb_acc = gaussian_nb.score(test_X, test_Y)
result['Model'].append("gaussian")
result["Accuracy"].append(gaussian_nb_acc)


# Before training, I need to further preprocess dataset (including normalization and categorization)

# Normalization: I need to rescale these three attributes "capital_run_length_average", "capital_run_length_longest", "capital_run_length_total" to be ranged from [1, ...] to [0, 100] (same formate as other attributes)

# In[8]:


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
for i in [54, 55, 56]:
    train_X[i] = 100 * NormalizeData(train_X[i])
    test_X[i] = 100 * NormalizeData(test_X[i])


# Categorization: Since I will be using "categorical_crossentropy" loss function in later neural network, I need train_Y and test_Y to be binary matrix representation of the {0, 1}

# In[ ]:


from keras.utils import to_categorical
# reformatting outputs to categorical values
train_Y = to_categorical(train_Y)
test_Y = to_categorical(test_Y)


# Checking the data is now what I expected.

# In[9]:


# print(train_X.head())
# print(train_X.describe())


# In[10]:


# print(train_Y.head())
# print(train_Y.describe())


# In[11]:


# print(test_X.head())
# print(test_X.describe())


# In[12]:


# print(test_Y.head())
# print(test_Y.describe())


# Constructing layers. This neural network includes 1 input layer(57 Neurons), 2 hidden layers(each with 16 neurons) and 1 output layer(2 neurons).

# In[6]:


import tensorflow as tf
import keras

model = keras.models.Sequential()

structure = [57, 16, 16, 2]

# Input layer + hidden layer 1
model.add(keras.layers.Dense(units=structure[1], input_dim = structure[0], activation = 'relu'))

# Hidden layer 2
model.add(keras.layers.Dense(units=structure[2], activation = 'relu'))

# Output layer - note that the activation function is softmax
model.add(keras.layers.Dense(units=structure[3], activation = tf.nn.softmax))


# In[7]:


model.compile(loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# Time to fit the model
print('Starting training')

training_stats = model.fit(train_X, train_Y, batch_size = 4, epochs = 5)

print('Training finished')
print('Training Evaluation: loss = %0.3f, accuracy = %0.2f%%'
      %(training_stats.history['loss'][-1], 100 * training_stats.history['accuracy'][-1]))


# In[8]:


import matplotlib.pyplot as graph
accuracy, = graph.plot(training_stats.history['accuracy'],label = 'Accuracy')
training_loss, = graph.plot(training_stats.history['loss'],label = 'Training Loss')


graph.legend(handles = [accuracy,training_loss])
loss = np.array(training_stats.history['loss'])
xp = np.linspace(0, loss.shape[0], 10 * loss.shape[0])
graph.plot(xp, np.full(xp.shape, 1), c = 'k', linestyle = ':', alpha = 0.5)
graph.plot(xp, np.full(xp.shape, 0), c = 'k', linestyle = ':', alpha = 0.5)
graph.savefig("image/neural_network_loss&acc.jpg")


# In[9]:


evaluation = model.evaluate(test_X, test_Y, verbose=0)
###

print('Test Set Evaluation: loss = %0.6f, accuracy = %0.2f%%' %(evaluation[0], 100*evaluation[1]))

result['Model'].append("Neural Network")
result["Accuracy"].append(evaluation[1])

print(result)

data = pd.DataFrame.from_dict(result)

print(data)

data.plot.bar(x = 'Model', y = 'Accuracy', rot = 10)
# df = pd.DataFrame({'lab':['A', 'B', 'C'], 'val':[10, 30, 20]})
# ax = df.plot.bar(x='lab', y='val', rot=0)
import matplotlib.pyplot as plt

plt.savefig("image/accuracy.jpg")
# In[ ]:




