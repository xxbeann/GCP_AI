#!/usr/bin/env python
# coding: utf-8

# ### [1] 단항 회귀_linear regression

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow import keras


# In[2]:


xs = np.array([-1.0, 0.0, 1.0,2.0,3.0,4.0],dtype=float)
ys = np.array([-3.0, -1.0, 1.0,3.0,5.0,7.0],dtype=float)


# In[3]:


model = tf.keras.Sequential([
    keras.layers.Dense(units=1,input_shape=[1])
])
model.summary()


# In[4]:


model.compile(optimizer='sgd',loss='mse')


# In[5]:


model.fit(xs,ys,epochs=500)


# In[6]:


print(model.predict([10.0]))


# In[7]:


# weight 과 bias값 접근
layer = keras.layers.Dense(units=1,input_shape=[1])
layer(xs.reshape(-1,1)) # x : (1,1)

# weight과 bias값을 random값으로 초기화한다, 실행시 마다 값이 다름
w = layer.weights[0].numpy()
b = layer.weights[1].numpy()
print('weight', w, w.shape) # W : (1,1)
print('bias', b, b.shape) # b : (1,)


# In[8]:


# units=3, 뉴런의 개수가 3개
layer = keras.layers.Dense(units=3,input_shape=[1])
layer(xs.reshape(-1,1)) # x : (1,1)

# weight과 bias값을 random값으로 초기화한다, 실행시 마다 값이 다름
w = layer.weights[0].numpy()
b = layer.weights[1].numpy()
print('weight', w, w.shape) # W : (1,3)
print('bias', b, b.shape) # b : (3,)


# ### [2] 다항 회귀

# In[9]:


xs = np.array([[-1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
               [0.1,  0.2, 0.4, 0.6, 0.7, 0.9],
               [-2.0, 1.0, 2.0, 3.0, 4.0, 5.0]],dtype=float).T  # T: tranpose,전치행렬
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
print(xs)
print(xs.shape)


# In[10]:


model = tf.keras.Sequential([
    keras.layers.Dense(units=1,input_shape=[3])
])
model.summary()


# In[11]:


model.compile(optimizer='sgd',loss='mse')


# In[12]:


model.fit(xs,ys,epochs=500)


# In[13]:


print(model.predict([[10.1,7,8.0]]))


# In[14]:


model.fit(xs,ys,epochs=500)


# ### [3] 학습된 모델 저장

# In[15]:


# 학습된 모델 저장
model.save('mymodel.h5')


# In[16]:


get_ipython().system('dir')


# In[17]:


# 저장된 모델 불러오기
new_model = tf.keras.models.load_model('mymodel.h5')
new_model.summary()


# In[18]:


# load 된 모델로 예측
print(new_model.predict([[10.1,7,8.0]]))


# In[ ]:




