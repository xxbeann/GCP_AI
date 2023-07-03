#!/usr/bin/env python
# coding: utf-8

# ### 13 XOR_DNN

# In[1]:


# XOR : 2 layer
# 2진 분류 : Logistic Regression
# 활성화 함수 : sigmoid 함수 사용


# In[2]:


import tensorflow as tf
import numpy as np
tf.random.set_seed(5)


# In[3]:


# train data set 
x_data = [[0,0],
          [0,1],
          [1,0],
          [1,1]]

y_data = [[0],
          [1],
          [1],
          [0]]

x_train = np.array(x_data,dtype=np.float32)
y_train = np.array(y_data,dtype=np.float32)


# In[4]:


# Layer 1 (은닉층_hidden layer) // 얼마로 나갈지 결정이 안되어있음
# 신경망을 만들때는 deep and wide
# 입력층 -> 은닉층 -> 출력층으로 이루어짐
# (m,n) * (n,L) = (m,L) : 행렬의 내적 곱셈
# (4,2) * (2,2) = (4,2)
# L값이 커질수록 학습이 잘됨

W1 = tf.Variable(tf.random.normal([2,2]),name='weigh1')
b1 = tf.Variable(tf.random.normal([2]),name='bias1')

def layer1(X):
    return tf.sigmoid(tf.matmul(X, W1) + b1) # 0과 1사이의 값이 출력


# In[5]:


# Layer 2 (출력층_output layer)
# (4,2) * (2,1) = (4,1)

W2 = tf.Variable(tf.random.normal([2,1]),name='weigh2')
b2 = tf.Variable(tf.random.normal([1]),name='bias2')

# 예측 함수(hypothesis) : H(x) = sigmoid(x*w+b)

def hypothesis(X):
    return tf.sigmoid(tf.matmul(layer1(X), W2) + b2) # 0과 1사이의 값이 출력


# In[6]:


# 비용함수 : logloss, 2진분류모델
# C(H(x),y)=-ylog(H(X))-(1-y)long(1-H(x))
# in 회귀, 예측함수와 실제값의 차이
# in 분류, 아무리커도 오차가 1이 넘지않음, 경사하강법X

def cost_func():
    cost = -tf.reduce_mean(y_train*tf.math.log(hypothesis(x_train)) + 
                          ((1-y_train)*tf.math.log(1-hypothesis(x_train))))
    return cost


# In[7]:


# 경사 하강법
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# 좀 더 발전된 경사 하강법
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
# 학습률 0.01로 옵티마이저 객체 생성
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# In[8]:


# 학습 시작(W값을 계속 갱신)
print('**** Start learning!!')
for step in range(10001):
    optimizer.minimize(cost_func,var_list =[W1,b1,W2,b2]) # 비용함수, weigh, bias
    if step % 1000 == 0:
        print('%04d'%step, 'cost:[',cost_func().numpy() ,']')
print('**** Learning Finished!!')


# In[13]:


# weight 과 bias 출력
print('Weight1:', W1.numpy())
print("bias1:", b1.numpy())
print('Weight2:', W2.numpy())
print("bias2:", b2.numpy())


# In[12]:


# 평가 : 정확도_accuracy
# sigmoid는 값이 0 혹은 1로 나오지 않으므로 변환이 필요.
def predict(x):
    return tf.cast(hypothesis(x) > 0.5, dtype=tf.float32)
    # cast함수 : 형변환

# 학습데이터가 검증데이터
x_test = x_train
y_test = y_train

# 예측
# preds = hypothesis(x_test)
preds = predict(x_test)
print(preds)

# 정확도
accuracy = tf.reduce_mean(tf.cast(tf.equal(preds,y_test), dtype=tf.float32))
print('Accuracy = ', accuracy.numpy()) # Accuracy = 1.0

# 2층 신경망으로는 해결


# In[ ]:




