#!/usr/bin/env python
# coding: utf-8

# ### 02 RNN_레이어종류

# ### * RNN 주요 레이어 종류
# #### (1) SimpleRNN :가장 간단한 형태의 RNN레이어, 활성화 함수로 tanh가 사용됨(tanh: -1 ~ 1 사이의 값을 반환)
# #### (2) LSTM(Long short Term Memory) : 입력 데이터와 출력 사이의 거리가 멀어질수로 연관 관계가 적어진다(Long Term Dependency,장기의존성 문제), LSTM은 장기 의존성 문제를 해결하기 위해 출력값외에 셀상태(cell state)값을 출력함, 활성화 함수로 tanh외에 sigmoid가 사용됨
# #### (3) GRU(Gated Recurent Unit) : 뉴욕대 조경현 교수 등이 제안, LSTM보다 구조가 간단하고 성능이 우수함

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


# Sequence data
X = np.array([[0,1,2,3],
              [1,2,3,4],
              [2,3,4,5],
              [3,4,5,6],
              [4,5,6,7],
              [5,6,7,8]],dtype=np.float32)

x_data = tf.reshape(X,(-1,4,1))  # (6,4,1)

y_data = np.array([4,5,6,7,8,9],dtype=np.float32)

print(x_data.shape,y_data.shape)
# print(type(x_data),type(y_data))
x_data


# ### [1] SimpleRNN
# #### 가장 간단한 형태의 RNN

# ### tanh() 함수 : Hyperbolic Tangent(tanh)

# In[3]:


# Hyperbolic Tangent 함수는 확장 된 시그모이드 함수이다
# tanh와 Sigmoid의 차이점은 Sigmoid의 출력 범위가 0에서 1 사이인 반면 tanh와 출력 범위는 -1에서 1사이라는 점이다
# Sigmoid와 비교하여 tanh와는 출력 범위가 더 넓고 경사면이 큰 범위가 더 크기 때문에 더 빠르게 수렴하여 학습하는 특성이 있다
# http://taewan.kim/post/tanh_diff/
import matplotlib.pyplot as plt

x = np.arange(-200,200)
x = x/10
y = np.tanh(x)

plt.plot(x,y)
plt.grid(True)
print(np.tanh(19))
print(np.tanh(20))


# ### RNN 순환 신경망 구현  : SimpleRNN

# In[11]:


# Rnn 순환 신경만 구현 : simplernn
model = tf.keras.Sequential([
    # X:(N,T,D):(None,4,1) --> (N,T,H):(None,4,300)
    # X:(None,1) , Wx:(1,300), Wh:(300,300), b:(300,), 1*300+300*300+300=90600
    tf.keras.layers.SimpleRNN(units=300, return_sequences=True, input_shape=[4,1]),
    # X:(None,300) , Wx:(300,300), Wh:(300,300), b:(300,), 300*300+300*300+300=180300
    tf.keras.layers.SimpleRNN(units=300),
    
    # X:(None,300), W:(300,1),b(1,)
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')
model.summary()


# In[12]:


# 학습
model.fit(x_data,y_data,epochs=100)


# In[13]:


# 예측
model.predict(x_data)


# In[14]:


# 학습되지 않은 입력 데이터에 대한 예측 결과
print(model.predict(np.array([[[6.],[7.],[8.],[9.]]])))
print(model.predict(np.array([[[-1.],[0.],[1.],[2.]]])))


# In[16]:


# 평가
model.evaluate(x_data,y_data)


# ### [2] LSTM(Long short Term Memory)
# #### 입력 데이터와 출력 사이의 거리가 멀어질수로 연관 관계가 적어진다(Long Term Dependency,장기의존성 문제)
# #### LSTM은 장기 의존성 문제를 해결하기 위해 출력값외에 셀상태(cell state)값을 출력함

# In[18]:


# RNN 순환 신경망 구현  : LSTM

model = tf.keras.Sequential([
    # X:(N,T,D):(None,4,1) --> (N,T,H):(None,4,300)
    # X:(None,1) , Wx:(1,4*300), Wh:(300,4*300), b:(4*300,), 1*4*300+300*4*300+4*300=362400
    tf.keras.layers.LSTM(units=300, return_sequences=True, input_shape=[4,1]),
    tf.keras.layers.LSTM(units=300),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')
model.summary()


# In[21]:


# 학습
model.fit(x_data,y_data,epochs=100)


# In[22]:


# 예측
model.predict(x_data)


# In[23]:


# 평가
model.evaluate(x_data,y_data)


# ### [3] GRU(Gated Recurrent Unit)
# #### 뉴욕대 조경현 교수 등이 제안, LSTM보다 구조가 간단하고 성능이 우수

# In[24]:


# RNN 순환 신경망 구현  : GRU

model = tf.keras.Sequential([
    tf.keras.layers.GRU(units=300, return_sequences=True, input_shape=[4,1]),
    tf.keras.layers.GRU(units=300),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',loss='mse')
model.summary()


# In[25]:


# 학습
model.fit(x_data,y_data,epochs=100)


# In[ ]:




