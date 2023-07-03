#!/usr/bin/env python
# coding: utf-8

# ### 13 XOR_DNN

# In[1]:


# XOR : 1 layer
# 2진 분류 : Logistic Regression
# 활성화 함수 : sigmoid 함수 사용


# In[3]:


import tensorflow as tf
import numpy as np
tf.random.set_seed(5)


# In[4]:


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


# In[5]:


# 변수 초기화 : weigh, bias
# (m,n) * (n,L) = (m,L) : 행렬의 내적 곱셈
# 바이어스는 웨이의 끝과 맞춰줌
# (4,2) * (2,1) = (4,1)

W = tf.Variable(tf.random.normal([2,1]),name='weigh')
b = tf.Variable(tf.random.normal([1]),name='bias')
print(W)
print(b)


# In[6]:


# 예측 함수(hypothesis) : H(x) = sigmoid(x*w+b)

def hypothesis(X):
    return tf.sigmoid(tf.matmul(X, W) + b) # 0과 1사이의 값이 출력


# In[7]:


# 비용함수 : logloss, 2진분류모델
# C(H(x),y)=-ylog(H(X))-(1-y)long(1-H(x))
# in 회귀, 예측함수와 실제값의 차이
# in 분류, 아무리커도 오차가 1이 넘지않음, 경사하강법X

def cost_func():
    cost = -tf.reduce_mean(y_train*tf.math.log(hypothesis(x_train)) + 
                          ((1-y_train)*tf.math.log(1-hypothesis(x_train))))
    return cost


# In[8]:


# 경사 하강법
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# 좀 더 발전된 경사 하강법
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
# 학습률 0.01로 옵티마이저 객체 생성
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# In[9]:


# 학습 시작(W값을 계속 갱신)
print('**** Start learning!!')
for step in range(10001):
    optimizer.minimize(cost_func,var_list =[W,b]) # 비용함수, weigh, bias
    if step % 1000 == 0:
        print('%04d'%step, 'cost:[',cost_func().numpy() ,']','W:',W.numpy(), 'b:', b.numpy())
print('**** Learning Finished!!')


# In[10]:


# weight 과 bias 출력
print('Weight:', W.numpy())
print("bias:", b.numpy())


# In[11]:


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
print('Accuracy = ', accuracy.numpy()) # Accuracy =  0.5

# 단층 신경망으로는 XOR Problem 해결 불가


# In[ ]:




