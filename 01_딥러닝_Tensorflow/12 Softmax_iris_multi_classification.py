#!/usr/bin/env python
# coding: utf-8

# ### 12 Softmax_iris_multi_classification

# In[1]:


import tensorflow as tf
import numpy as np
tf.random.set_seed(5)


# In[2]:


'''
하 피땀눈물,,,
각각 데이터 받아서 슬라이싱
y_train 데이터는 스트링 형식이므로 int 형식으로 변환시켜주기 위해
species_list를 만들어서 list.index로 각각 매칭시켜줌
매칭시켜준후 2차원 배열로 나온 y_train을
다시 one_hot_encoding을 통해 분류가 가능하게 한다.
이를 y_test에도 똑같이 적용한다.
'''


# In[3]:


# train 데이터 만들기

species_list =['"setosa"','"versicolor"','"virginica"']

xy = np.loadtxt('Tensorflow기본_데이터셋/iris.csv',delimiter=',', skiprows=1, dtype=np.str)
xy.shape

# index slicing
x_train = np.float32(xy[:35,1:-1])
x_train = np.append(x_train , np.float32(xy[50:85,1:-1]), 0)
x_train = np.append(x_train , np.float32(xy[100:135,1:-1]), 0)

y_train = xy[:35, [-1]]
y_train = np.append(y_train, xy[50:85, [-1]], 0)
y_train = np.append(y_train, xy[100:135, [-1]], 0)

# after slicing - str을 indexnumber로 매칭시켜줌

for i in range(105):
    y_train[i,-1] = np.int32(species_list.index(y_train[i,-1]))

print(x_train.shape)
print(y_train.shape)


# In[4]:


# test 데이터 만들기

x_test = np.float32(xy[35:50,1:-1])
x_test = np.append(x_test , np.float32(xy[85:100 ,1:-1]), 0)
x_test = np.append(x_test , np.float32(xy[135:150,1:-1]), 0)

y_test = xy[35:50, [-1]]
y_test = np.append(y_test, xy[85:100, [-1]], 0)
y_test = np.append(y_test, xy[135:150, [-1]], 0)

for i in range(45):
    y_test[i,-1] = np.int32(species_list.index(y_test[i,-1]))

# print(x_test)    
# print(y_test)


# In[5]:


# one_hot_encoding
nb_classes = 3

y_one_hot = tf.one_hot(y_train, nb_classes) #3차원 (70, 1, 7), Rank = 3
y_one_hot = tf.reshape(y_one_hot,[-1,nb_classes]) # 3차원에서 2차원으로 변환

print(y_one_hot.shape)

# test data -> one_hot_encoding
y_test_one_hot = tf.one_hot(y_test, nb_classes) #3차원 (70, 1, 7), Rank = 3
y_test_one_hot = tf.reshape(y_test_one_hot,[-1,nb_classes]) # 3차원에서 2차원으로 변환

print(y_test_one_hot.shape)


# In[6]:


# 변수 초기화 : weigh, bias
# (m,n) * (n,L) = (m,L) : 행렬의 내적 곱셈
# (105,4) * (4,3) = (105,3)

W = tf.Variable(tf.random.normal([4,nb_classes]),name='weigh')
b = tf.Variable(tf.random.normal([nb_classes]),name='bias')


# In[7]:


# 예측 함수(hypothesis) : H(x) = softmax(x*w+b)

def logits(X):
    return tf.matmul(X,W) + b
    # 회귀의 예측함수
    
def hypothesis(X):
    return tf.nn.softmax(logits(X)) # 0과 1사이의 확률


# In[8]:


#  비용함수 구현 방법 2 : tf.nn.softmax_cross_entropy_with_logits() 함수 사용

def cost_func():
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits(x_train),
                                                    labels=y_one_hot)
    cost = tf.reduce_mean(cost_i)
    return cost
    


# In[9]:


# 경사 하강법
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# 좀 더 발전된 경사 하강법
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
# 학습률 0.01로 옵티마이저 객체 생성
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# In[10]:


# 학습 시작(W값을 계속 갱신)
print('**** Start learning!!')
for step in range(5001):
    optimizer.minimize(cost_func,var_list =[W,b]) # 비용함수, weigh, bias
    if step % 1000 == 0:
        print('%04d'%step, 'cost:[',cost_func().numpy() ,']')
print('**** Learning Finished!!')


# In[11]:


# weight 과 bias 출력
print('Weight:', W.numpy())
print("bias:", b.numpy())


# In[12]:


# 정확도 측정(평가)

def predict(x):
    return tf.argmax(hypothesis(x), axis=1)
    #argmax : 1차원 배열에서 가장 큰 값을 찾아 인덱스를 리턴
    
# y_test_one_hot을 argamx형식으로 묶어줌
correct_predict = tf.equal(predict(x_test),tf.argmax(y_test_one_hot,1))
# 형변환 타입 맞추기
accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32)) 
print('Accuracy:', accuracy.numpy())


# In[ ]:




