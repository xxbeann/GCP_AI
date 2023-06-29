#!/usr/bin/env python
# coding: utf-8

# ### 11 Softmax_zoo_multi_classification

# In[1]:


# multi-classification
# multi-nomial classification (다중 분류) : Y값의 범주가 3개 이상인 분류
# 활성화 함수(Activation function) 으로 softmax함수 가 사용된다

import tensorflow as tf
import numpy as np
tf.random.set_seed(5)


# In[2]:


# 데이터 불러오기
xy = np.loadtxt('Tensorflow기본_데이터셋/data-04-zoo.csv',delimiter=',', dtype=np.float32)
# print(xy.shape) # (101,17)

# 학습용 데이터 70% 70개
x_train = xy[0:70,:-1] # x, 마지막 컬럼을 제외
y_train = xy[0:70,[-1]] # y, 마지막 컬럼만 2차원으로 추출

# 검증용 데이터 30% 31개
x_test = xy[70:,:-1] #[531:,:-1]
y_test = xy[70:,[-1]]

y_train.shape


# In[15]:


# one - hot encoding
# 0: [1 0 0 0 0 0 0]
# 1: [0 1 0 0 0 0 0]
# 2: [0 0 1 0 0 0 0]
# ...
# 6: [0 0 0 0 0 0 1]

#class 범주 개수 (0,1,2,3,4,5,6)
nb_classes = 7

y_one_hot = tf.one_hot(y_train, nb_classes) #3차원 (70, 1, 7), Rank = 3
y_one_hot = tf.reshape(y_one_hot,[-1,nb_classes]) # 3차원에서 2차원으로 변환

# y_test도 one-hot encoding
y_test_one_hot = tf.one_hot(y_test, nb_classes)
y_test_one_hot = tf.reshape(y_test_one_hot,[-1,nb_classes])
print(y_test_one_hot)


# In[4]:


# 변수 초기화 : weigh, bias
# (m,n) * (n,L) = (m,L) : 행렬의 내적 곱셈
# (70,16) * (16,7) = (70,7)

W = tf.Variable(tf.random.normal([16,nb_classes]),name='weigh')
b = tf.Variable(tf.random.normal([nb_classes]),name='bias')


# In[5]:


# 예측 함수(hypothesis) : H(x) = softmax(x*w+b)

def logits(X):
    return tf.matmul(X,W) + b
    # 회귀의 예측함수
    
def hypothesis(X):
    return tf.nn.softmax(logits(X)) # 0과 1사이의 확률


# In[6]:


#  비용함수 구현 방법 2 : tf.nn.softmax_cross_entropy_with_logits() 함수 사용

def cost_func():
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits(x_train),
                                                    labels=y_one_hot)
    cost = tf.reduce_mean(cost_i)
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
    optimizer.minimize(cost_func,var_list =[W,b]) # 비용함수, weigh, bias
    if step % 1000 == 0:
        print('%04d'%step, 'cost:[',cost_func().numpy() ,']')
print('**** Learning Finished!!')


# In[9]:


# weight 과 bias 출력
print('Weight:', W.numpy())
print("bias:", b.numpy())


# In[18]:


# 정확도 측정(평가)

def predict(x):
    return tf.argmax(hypothesis(x), axis=1)
    #argmax 값이 1인값의 인덱스를 반환
    
# y_test_one_hot을 argamx형식으로 묶어줌
correct_predict = tf.equal(predict(x_test),tf.argmax(y_test_one_hot,1))
# 형변환 타입 맞추기
accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32)) 
print('Accuracy:', accuracy.numpy())

