#!/usr/bin/env python
# coding: utf-8

# ### 04 Keras_basic
# 
# ### Sequential Layer 의 주요 레이어 정리 
# #### [1]  Dense Layer : Fully Connected Layer, layer의 입력과 출력 사이에 있는 모든 뉴런이 서로 연결 되는 Layer
# #### [2]  Flatten Layer : 다차원(4차원)을 2차원으로 축소하여 FC Layer에 전달한다
# #### [3]  Conv2D Layer : 이미지 특징을 추출하는 Convolution Layer 
# #### [4]  MaxPool2D Layer : 중요 데이터(값이 큰 데이터)를 subsampling 하는 Layer
# #### [5]  Dropout Layer : 학습시 신경망의 과적합을 막기위해 일부 뉴런을 제거하는 Layer

# ### [1] Dense Layer

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(5)


# In[2]:


# # keras 를 이용한 XOR 네트워크

# train data set 
x_data = [[0,0],
          [0,1],
          [1,0],
          [1,1]]   # (4,2)

y_data = [[0],
          [1],
          [1],
          [0]]     # (4,1)

x_train = np.array(x_data,dtype=np.float32)
y_train = np.array(y_data,dtype=np.float32)


# In[3]:


# Dense Layer 구현
model = tf.keras.Sequential([
    # (4,2) * (2,2) = (4,2)   , W:2*2=4, b=2 --> Params : 6
    tf.keras.layers.Dense(units=4,activation='sigmoid',input_shape=(2,)) ,
    # (4,2) * (2,1) = (4,1)   , W:2*1, b=1 --> Params : 3
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[4]:


# 학습
history = model.fit(x_train,y_train,epochs=1000,batch_size=1,verbose=1) # verbose=1, 메세지를 출력


# In[16]:


# 시각화_loss
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[18]:


# 시각화_accuracy
epoch_count = range(1, len(history.history['accuracy']) + 1)
plt.plot(epoch_count, history.history['accuracy'], 'r-')
plt.legend(['Training Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# In[5]:


# 예측
preds = model.predict(x_train)
np.round(preds)


# In[6]:


# 평가
model.evaluate(x_train,y_train)


# In[7]:


# X,Y input data 
x_train = np.array([[1.0,   2.0,  3.0,  4.0,  5.0],
                    [6.0,   7.0,  8.0,  9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0]])  # (3,5)
                    
                    
y_train = np.array([[1.1,   2.2,  3.3,  4.4,  5.5],
                    [6.1,   7.2,  8.3,  9.4, 10.5],
                    [11.1, 12.2, 13.3, 14.4, 15.5]]) # (3,5)

model = tf.keras.Sequential([
    # (3,1) * (1,100) = (3,100)   , Param :200 (W:1*100, b:100), (1 + 1) * 100 = 200
    # tf.keras.layers.Dense(units=100, activation='relu', input_shape=(1,)), # input data의 shape을 (?,1) 

    # (3,5) * (5,100) = (3,100)   , Param :600 (W:5*100, b:100), (5 + 1) * (100) = 600
    tf.keras.layers.Dense(units=100, activation='relu', input_shape=(5,)), # input data의 shape을 (?,5) 
    
    # (3,100) * (100,50) = (3,50)   , Param : 5050 (W:100*50, b:50), (100+1)*50 = 5050
    tf.keras.layers.Dense(units=50, activation='relu' ), 

    # (3,50) * (50,25) = (3,25)   , Param : 1275 (W:50*25, b:25), (50 + 1)* 25 = 1275
    tf.keras.layers.Dense(units=25, activation='relu' ),
    
    # (3,25) * (25,5) = (3,5)   , Param : 130 (W:25*5, b:5), (25 + 1) * 5 = 130
    tf.keras.layers.Dense(units=5 , activation='relu' ),
    
#     # (3,5) * (5,1) = (3,1)   , Pram : 6  (W:5*1, b:1), (5 + 1) *1 = 6
#     tf.keras.layers.Dense(units=1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')    
model.summary()    


# In[8]:


# 학습
model.fit(x_train,y_train,epochs=100)


# In[9]:


# 예측
model.predict(x_train)


# In[10]:


# 평가
model.evaluate(x_train,y_train)


# ### [2] Flatten Layer

# In[11]:


# 4차원 데이터를 2차원으로 축소하여 FC Layer에 전달
data = np.arange(1000*28*28*10).reshape(-1,28,28,10)
print(data.shape) #(None, 28, 28, 10)
layer =tf.keras.layers.Flatten()
#print(type(layer))
#dir(layer)
output = layer(data) #layer은 함수처럼 사용가능
print(output.shape) #(None, 28*28*10) = (none, 7840)


# In[12]:


# 3차원 데이터를 2차원으로 축소하여 FC Layer에 전달
data = np.arange(1000*28*28).reshape(1000,28,28).astype(np.float32)
print(data.shape) #(1000,28,28)
layer =tf.keras.layers.Flatten()
#print(type(layer))
#dir(layer)
output = layer(data) #layer은 함수처럼 사용가능
print(output.shape) #(None, 28*28) = (none, 784)


# ### [3] Dropout Layer

# In[13]:


# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
# The Dropout layer randomly sets input units to 0 with a frequency of rate 
# at each step during training time, which helps prevent overfitting. 
# Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.

# Note that the Dropout layer only applies when training is set to True such that
# no values are dropped during inference. When using model.fit, training will be
# appropriately set


# In[14]:


tf.random.set_seed(0)
data = np.arange(10).reshape(5,2).astype(np.float32)
print(data)

layer = tf.keras.layers.Dropout(0.3,input_shape=(2,))
outputs = layer(data,training=True)
# model.fit()호출 시는  학습 모드로 training=True가 되어 dropuout 적용
# model.evaluate() 호출 시는 예측 모드로 False가 되어 dropuout이 수행되지 않음
print(outputs) #데이터의 30%는 0으로 나머지 데이터는 1/(1-0.3)으로 곱하여 scaled up 된다.


# In[ ]:




