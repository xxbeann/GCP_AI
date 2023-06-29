#!/usr/bin/env python
# coding: utf-8

# ### 09 Logistic_regression_Caesarian_Using_DNN

# In[1]:


# 09 Logistic_regression_Caesarian
# 제왕절개 진단 2진 분류 모델

import tensorflow as tf
import numpy as np
tf.random.set_seed(5)


# In[2]:


# 데이터 불러오기
xy = np.loadtxt('Tensorflow기본_데이터셋/caesarian.csv',delimiter=',', dtype=np.float32)

# 학습용 데이터 70% 56개
x_train = xy[0:56,:-1] # x, 마지막 컬럼을 제외
y_train = xy[0:56,[-1]] # y, 마지막 컬럼만 2차원으로 추출

# 검증용 데이터 30% 24개
x_test = xy[56:,:-1] #[531:,:-1]
y_test = xy[56:,[-1]]

x_train.shape


# In[3]:


# Layer 1 (은닉층_hidden layer) // 얼마로 나갈지 결정이 안되어있음
# 변수 초기화 : weigh, bias
# (56,5) * (5,3) = (56,3)

W1 = tf.Variable(tf.random.normal([5,3]),name='weigh1')
b1 = tf.Variable(tf.random.normal([3]),name='bias1')

def layer1(X):
    return tf.nn.relu(tf.matmul(X, W1) + b1)


# In[4]:


# Layer 2 (은닉층_hidden layer)
# 변수 초기화 : weigh, bias
# (56,3) * (3,2) = (56,2)

W2 = tf.Variable(tf.random.normal([3,2]),name='weigh2')
b2 = tf.Variable(tf.random.normal([2]),name='bias2')

def layer2(X):
    return tf.nn.relu(tf.matmul(layer1(X) ,W2) + b2)


# In[5]:


# Layer 3 (출력층_output layer)
# 변수 초기화 : weigh, bias
# (56,2) * (2,1) = (56,1)

W3 = tf.Variable(tf.random.normal([2,1]),name='weigh3')
b3 = tf.Variable(tf.random.normal([1]),name='bias3')

def hypothesis(X):
    return tf.sigmoid(tf.matmul(layer2(X), W3) + b3)


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
for step in range(5001):
    optimizer.minimize(cost_func,var_list =[W1,b1,W2,b2,W3,b3]) # 비용함수, weigh, bias
    if step % 1000 == 0:
        print('%04d'%step, 'cost:[',cost_func().numpy() ,']')
print('**** Learning Finished!!')


# In[9]:


# weight 과 bias 출력
print('Weight:', W3.numpy())
print("bias:", b3.numpy())


# In[10]:


# 평가 : 정확도_accuracy
# sigmoid는 값이 0 혹은 1로 나오지 않으므로 변환이 필요.
def predict(x):
    return tf.cast(hypothesis(x) > 0.5, dtype=tf.float32)
    # cast함수 : 형변환

# 예측
# preds = hypothesis(x_test)
preds = predict(x_test)
# print(preds)

# 정확도
accuracy = tf.reduce_mean(tf.cast(tf.equal(preds,y_test), dtype=tf.float32))
print('Accuracy = ', accuracy.numpy())


# In[ ]:




