#!/usr/bin/env python
# coding: utf-8

# ### 06 Boston_dataset_linear_regression_Using_DNN

# In[1]:


# 3층 신경망 RMSE 비교


# In[2]:


import tensorflow as tf
import numpy as np
tf.random.set_seed(5)


# In[3]:


#데이터 불러오기
train_xy = np.loadtxt('Tensorflow기본_데이터셋/boston_train.csv',delimiter=',',skiprows=1, dtype=np.float32)
test_xy = np.loadtxt('Tensorflow기본_데이터셋/boston_test.csv',delimiter=',',skiprows=1, dtype=np.float32)

# index slicing 
x_train = train_xy[:,:-1] # x, 마지막 컬럼을 제외
y_train = train_xy[:,[-1]] # y, 마지막 컬럼만 2차원으로 추출
x_test = test_xy[:,:-1]
y_test = test_xy[:,[-1]]

print(x_train.shape, y_train.shape)


# In[4]:


# Layer 1 (은닉층_hidden layer) // 얼마로 나갈지 결정이 안되어있음
# 변수 초기화 : weigh, bias
# (400,9) * (9,9) = (400,9)

W1 = tf.Variable(tf.random.normal([9,9]),name='weigh1')
b1 = tf.Variable(tf.random.normal([9]),name='bias1')

def layer1(X):
    return tf.nn.relu(tf.matmul(X, W1) + b1)


# In[5]:


# Layer 2 (은닉층_hidden layer)
# 변수 초기화 : weigh, bias
# (400,9) * (9,5) = (400,5)

W2 = tf.Variable(tf.random.normal([9,5]),name='weigh2')
b2 = tf.Variable(tf.random.normal([5]),name='bias2')

def layer2(X):
    return tf.nn.relu(tf.matmul(layer1(X) ,W2) + b2)


# In[6]:


# Layer 3 (출력층_output layer)
# 변수 초기화 : weigh, bias
# (400,5) * (5,1) = (400,1)

W3 = tf.Variable(tf.random.normal([5,1]),name='weigh3')
b3 = tf.Variable(tf.random.normal([1]),name='bias3')

def hypothesis(X):
    return tf.matmul(layer2(X), W3) + b3


# In[7]:


# 비용함수 : (H(x)-y)^2의 평균
# tf.square() : 제곱
# tf.reduce_mean() : 평균

def cost_func():
    cost = tf.reduce_mean(tf.square(hypothesis(x_train) - y_train))
    return cost


# In[8]:


# 경사 하강법
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# 좀 더 발전된 경사 하강법
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# In[9]:


# 학습 시작(W값을 계속 갱신)
print('**** Start learning!!')
for step in range(8001):
    optimizer.minimize(cost_func,var_list =[W1,b1,W2,b2,W3,b3]) # 비용함수, weigh, bias
    if step % 1000 == 0:
        print('%04d'%step, 'cost:[',cost_func().numpy() ,']')
print('**** Learning Finished!!')


# In[10]:


# weight 과 bias 출력
print('Weight:', W3.numpy())
print("bias:", b3.numpy())


# In[11]:


# 예측
print('***** Predict')
print(hypothesis(x_test).numpy())


# In[12]:


# 정확도 측정 : RMSE Root mean squared error
# 회귀모델의 평가
def get_rmse(y_test,preds):
    squared_error = 0
    for k,_ in enumerate(y_test):
        squared_error += (preds[k] - y_test[k])**2
    mse = squared_error/len(y_test)  
    rmse = np.sqrt(mse)
    return rmse[0]

preds = hypothesis(x_test).numpy()
print('RMSE:',get_rmse(y_test,preds))


# In[ ]:




