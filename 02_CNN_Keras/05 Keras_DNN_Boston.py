#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(5)


# In[2]:


#데이터 불러오기
train_xy = np.loadtxt('Tensorflow기본_데이터셋/boston_train.csv',delimiter=',',skiprows=1, dtype=np.float32)
test_xy = np.loadtxt('Tensorflow기본_데이터셋/boston_test.csv',delimiter=',',skiprows=1, dtype=np.float32)

# index slicing 
x_train = train_xy[:,:-1] # x, 마지막 컬럼을 제외
y_train = train_xy[:,[-1]] # y, 마지막 컬럼만 2차원으로 추출
x_test = test_xy[:,:-1]
y_test = test_xy[:,[-1]]

print(x_train.shape, y_train.shape)


# In[3]:


# Dense Layer 구현 [2층 신경망]
model = tf.keras.Sequential([
    # (400,9) * (9,20) = (400,20)   , W:9*20=180, b=20 --> Params : 200
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(9,)) ,
    # (4,2) * (2,1) = (4,1)   , W:2*1, b=1 --> Params : 3
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='mean_squared_error')

model.summary()


# In[4]:


# 학습 verbose=1, 메세지를 출력
history = model.fit(x_train,y_train,epochs=700,batch_size=1,verbose=1) 


# In[5]:


history.history


# In[6]:


# 시각화_loss
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training MSE'])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()


# In[7]:


# 예측
preds = model.predict(x_test)
np.round(preds)


# In[8]:


# 평가
model.evaluate(x_test,y_test)


# In[19]:


# Dense Layer 구현 [3층 신경망]
model = tf.keras.Sequential([
    # (400,9) * (9,20) = (400,20)   , W:9*20=180, b=20 --> Params : 200
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(9,)) ,
    # (400,20) * (20,10) = (400,10)   , W:20*10, b=10 --> Params : 210
    tf.keras.layers.Dense(units=10,activation='relu'),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='mean_squared_error')

model.summary()


# In[20]:


# 학습 verbose=1, 메세지를 출력
history = model.fit(x_train,y_train,epochs=700,batch_size=1,verbose=1) 


# In[21]:


# 시각화_loss
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training MSE'])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.show()


# In[22]:


# 예측
preds = model.predict(x_test)
np.round(preds)


# In[23]:


# 평가
model.evaluate(x_test,y_test)


# In[ ]:




