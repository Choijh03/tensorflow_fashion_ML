#!/usr/bin/env python
# coding: utf-8

# In[56]:


import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
print(keras.__version__)


# In[57]:


fashion_mnist = keras.datasets.fashion_mnist


# In[58]:


(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[59]:


X_train_full.shape


# In[60]:


X_train_full.dtype


# ## pre-processing

# In[61]:


X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000]/255.0, y_train_full[5000:]
X_test = X_test/255.0


# In[62]:


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
              "Ankle Boot"]


# In[63]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(500, activation = 'relu'))

model.add(keras.layers.Dense(500, activation = 'sigmoid'))

model.add(keras.layers.Dense(10, activation = 'softmax'))


# In[64]:


model.summary()


# In[65]:


model.layers


# In[66]:


model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "nadam", metrics = ["accuracy"])


# In[67]:


history = model.fit(X_train, y_train, epochs=50, validation_data = (X_valid, y_valid))


# In[68]:


import matplotlib.pyplot as plt
import pandas as pd

pd.DataFrame(history.history).plot(figsize =(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[69]:


model.evaluate(X_test, y_test)


# In[70]:


X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(3)


# In[ ]:




