#!/usr/bin/env python
# coding: utf-8

# ### 14 Mnist_softmax

# In[1]:


# softmax: 다중분류
# sigmoid: 2진분류
# 회귀 - no action function

# mnist_softmax
# MNIST(Modified National Institute of Standard Technology) Dataset
# https://ko.wikipedia.org/wiki/MNIST
# label : 0 ~ 9 , 손글씨체 이미지  28*28(784 byte) , gray scale
# Train : 60000개 , Test : 10000개

# mini batch : 큰 데이터를 쪼개어 1회에 작은 단위로 가져다가 학습, next_batch()
# epoch : batch를 반복하여 전체 데이터가 모두 소진되었을 때를 1 epoch
# Vanishing Gradient  : 신경망이 깊어 질수록 입력신호가 사라진다(줄어든다), sigmoid 사용시
# Relu  : Rectified Linear Unit, DNN(deep neural net) 구현시 sigmoid 대신 사용됨
# dropout : 전체 신경망의 일부를 사용하지 않고 학습, 예측시는 전체를


# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(5)


# In[3]:


# mnist 데이터 가져오기
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
print(x_train.shape)


# In[4]:


# 이미지로 출력 (시각화)
def show_one_image(n):
    print(type(x_train),x_train.shape) # (60000, 28, 28)
    print(type(y_train),y_train.shape) # (60000,)
    print(type(x_test),x_test.shape)   # (10000, 28, 28)
    print(type(y_test),y_test.shape)   # (10000,)
    
    image = x_train[n]
    print(y_train[n])
    
    plt.imshow(image,cmap='Greys')
    plt.show()

show_one_image(0)


# In[5]:


# X값의 shape을 2차원으로 변환
# -1은 자동계산, 6만개 채워주기, -1 technique
x_train = x_train.reshape(-1,28*28) 
x_test = x_test.reshape(-1,28*28)

# X값의 형변환 float32 // 텐서플로의 데이터 타입은 float32여야 함.
# 위에서는 numpy array 였지만 밑에서 텐서 객체로 바뀜
x_train = tf.cast(x_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)

print(x_train.shape,x_train.dtype)
print(x_test.shape,x_test.dtype)


# In[6]:


# one - hot encoding
# (60000, 10) Rank=2
# keras에서는 무조건 2차원?
nb_classes = 10

y_one_hot = tf.one_hot(y_train, nb_classes) 
print(y_one_hot.shape)


# In[7]:


# 변수 초기화 : weigh, bias
# (m,n) * (n,L) = (m,L) : 행렬의 내적 곱셈
# (60000,784) * (784,10) = (60000,10)

W = tf.Variable(tf.random.normal([784,nb_classes]),name='weigh')
b = tf.Variable(tf.random.normal([nb_classes]),name='bias')


# In[8]:


# 예측 함수(hypothesis) : H(x) = softmax(x*w+b)

def logits(X):
    return tf.matmul(X,W) + b
    # 회귀의 예측함수
    
def hypothesis(X):
    return tf.nn.softmax(logits(X)) # 0과 1사이의 확률


# In[9]:


#  비용함수 구현 방법 2 : tf.nn.softmax_cross_entropy_with_logits() 함수 사용

def cost_func():
    cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits(x_train),
                                                    labels=y_one_hot)
    cost = tf.reduce_mean(cost_i)
    return cost
    


# In[10]:


# 경사 하강법
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
# 좀 더 발전된 경사 하강법
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
# 학습률 0.01로 옵티마이저 객체 생성
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# In[11]:


# 방법1, 전체데이터를 한번에 학습, 비효율적이며 학습에 장시간 소요
# 학습 시작(W값을 계속 갱신) #비효율적 GPU를 안쓸때
# print('**** Start learning!!')
# for step in range(2001):
#     optimizer.minimize(cost_func,var_list =[W,b]) # 비용함수, weigh, bias
#     if step % 1000 == 0:
#         print('%04d'%step, 'cost:[',cost_func().numpy() ,']')
# print('**** Learning Finished!!')


# In[12]:


# 방법2, batch 사이즈로 나누어 학습, 효율적이며 학습시간 단축
# 학습시작

training_epoch = 25
batch_size = 600

print('**** Start learning!!')
for epoch in range(training_epoch): #25회
    
    avg_cost = 0
    
    # 60000/600 = 100
    total_batch = int(x_train.shape[0]/batch_size)
    for k in range(total_batch): #100회
        batch_xs = x_train[0+k*batch_size:batch_size+k*batch_size] #index slicing, 600개의 x 데이터
        batch_ys = y_one_hot[0+k*batch_size:batch_size+k*batch_size] #index slicing, 600개의 y 데이터
        
        # 비용함수
        def cost_func_batch():
            cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits(batch_xs),labels=batch_ys)
            cost = tf.reduce_mean(cost_i)
            return cost
        
        optimizer.minimize(cost_func_batch,var_list =[W,b]) # 비용함수, weigh, bias
        avg_cost += cost_func_batch().numpy()/total_batch
        

    print('Epoch: ''%04d'%(epoch+1), 'cost:[',cost_func().numpy() ,']')
print('**** Learning Finished!!')


# In[15]:


# 정확도 측정(평가)

y_test_one_hot = tf.one_hot(y_test, nb_classes) 

def predict(x):
    return tf.argmax(hypothesis(x), axis=1)
    #argmax : 1차원 배열에서 가장 큰 값을 찾아 인덱스를 리턴
    
# y_test_one_hot을 argamx형식으로 묶어줌
correct_predict = tf.equal(predict(x_test),tf.argmax(y_test_one_hot,1))
# 형변환 타입 맞추기
accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32)) 
print('Accuracy:', accuracy.numpy())


# In[21]:


# 예측
print('***** Predict')
pred = predict(x_test).numpy()
print('',pred[:100],'\n',y_test[:100])


# In[20]:


r = np.random.randint(0,x_test.shape[0] - 1) # 0 to 9999 random int number
# r = 1411   # Label: 0, Prediction :  [9]

print('random = ',r, 'Label:',y_test[r])

print('Prediction : ',predict(x_test[r:r+1]).numpy())

image = tf.reshape(x_test[r],(28,28))
plt.imshow(image,cmap='Greys')
plt.show()


# In[ ]:




