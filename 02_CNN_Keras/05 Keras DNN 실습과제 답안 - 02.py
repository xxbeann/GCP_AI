#!/usr/bin/env python
# coding: utf-8

# ## Keras DNN 실습과제 답안 - 02
# ### Logistic Regression , 'caesarian.csv'

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
tf.random.set_seed(5)
np.random.seed(5)


# In[2]:


xy = np.loadtxt('caesarian.csv',delimiter=',',dtype=np.float32)

# train data set
x_data = xy[0:56, 0:-1 ]
y_data = xy[0:56, [-1] ]

x_train = np.array(x_data,dtype=np.float32)
y_train = np.array(y_data,dtype=np.float32)
print(x_train.shape)  # (56,5)
print(y_train.shape)  # (56,1)


# In[3]:


# 모델 구현
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=20, input_shape=[5],activation='relu'),
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


model.summary()


# In[4]:


# 학습
history = model.fit(x_train, y_train, epochs=700)


# In[5]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[6]:


# 평가
x_test = xy[56:, :-1 ]
y_test = xy[56:, [-1] ]

accuracy = model.evaluate(x_test, y_test)

print('Accuracy : ',accuracy[1])


# ## 모델 개선 : 3 layers

# In[7]:


tf.keras.backend.clear_session()
tf.random.set_seed(5)
np.random.seed(5)


# In[8]:


# 모델 구현
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=20, input_shape=[5],activation='relu'),
    tf.keras.layers.Dense(units=2, activation='relu'),
    tf.keras.layers.Dense(units=1,activation='sigmoid') 
])

model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

model.summary()


# In[9]:


# 학습
history = model.fit(x_train, y_train, epochs=700)


# In[10]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[11]:


# 평가
x_test = xy[56:, :-1 ]
y_test = xy[56:, [-1] ]

accuracy = model.evaluate(x_test, y_test)
print('Accuracy : ',accuracy[1])


# In[ ]:





# In[ ]:




