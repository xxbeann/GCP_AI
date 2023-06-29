#!/usr/bin/env python
# coding: utf-8

# ### 10 Softmax_multi_classification

# In[2]:


import tensorflow as tf
import numpy as np
tf.random.set_seed(5)


# In[3]:


# train data set :
# x_data :  [N,4]  --> [8,4]
x_data = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]

# y_data : [N,3] --> [8,3]
#해당 인덱스의 값을 1로 지정
y_data = [[0,0,1],  # [2]
          [0,0,1],  # [2]
          [0,0,1],  # [2]
          [0,1,0],  # [1]
          [0,1,0],  # [1]
          [0,1,0],  # [1]
          [1,0,0],  # [0]
          [1,0,0]]  # [0]

x_train = np.array(x_data,dtype=np.float32)
y_train = np.array(y_data,dtype=np.float32)


# In[4]:


nb_classes = 3
# 변수 초기화 : weigh, bias
# (m,n) * (n,L) = (m,L) : 행렬의 내적 곱셈
# (8,4) * (4,3) = (8,3)
# bias가 1에서 3으로 변화

W = tf.Variable(tf.random.normal([4,nb_classes]),name='weigh')
b = tf.Variable(tf.random.normal([nb_classes]),name='bias')
print(W)
print(b)


# In[5]:


# 예측 함수(hypothesis) : H(x) = softmax(x*w+b)

def logits(X):
    return tf.matmul(X,W) + b
    # 회귀의 예측함수
    
def hypothesis(X):
    return tf.nn.softmax(logits(X)) # 0과 1사이의 확률


# In[6]:


# # 비용 함수 구현 방법 1: log함수를 사용하여 수식을 직접 표현
# Sigma [y(log(H(x)))]
# in 회귀, 예측함수와 실제값의 차이
# in 분류, 아무리커도 오차가 1이 넘지않음, 경사하강법X

# def cost_func():
#     cost = tf.reduce_mean(-tf.reduce_sum(y_train*tf.math.log(hypothesis(x_train)),
#                                          axis=1))
#     return cost


# In[7]:


#  비용함수 구현 방법 2 : tf.nn.softmax_cross_entropy_with_logits() 함수 사용

def cost_func():
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits(x_train),
                                                    labels=y_train)
    cost = tf.reduce_mean(cost_i)
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
for step in range(5001):
    optimizer.minimize(cost_func,var_list =[W,b]) # 비용함수, weigh, bias
    if step % 1000 == 0:
        print('%04d'%step, 'cost:[',cost_func().numpy() ,']','W:',W.numpy(), 'b:', b.numpy())
print('**** Learning Finished!!')


# In[10]:


# weight 과 bias 출력
print('Weight:', W.numpy())
print("bias:", b.numpy())


# In[17]:


# 예측
def predict(x):
    return tf.argmax(hypothesis(x), axis=1)

# 학습 데이터 = 검증 데이터
x_test = x_train
y_test = y_train

preds = predict(x_test)
print(preds.numpy())
print(hypothesis(x_test).numpy())
print(tf.argmax(y_test, axis=1).numpy())


# In[ ]:




