#!/usr/bin/env python
# coding: utf-8

# ### 06_Fashion_mnist_keras_callback

# In[1]:


# mnist cnn keras callback CNN과 callback구현

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# mnist 데이터 가져오기
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

# X값의 형변환 float32 // 텐서플로의 데이터 타입은 float32여야 함.
# 위에서는 numpy array 였지만 밑에서 텐서 객체로 바뀜
# x_train = tf.cast(x_train, dtype=tf.float32)
# x_test = tf.cast(x_test, dtype=tf.float32)

print(x_train.shape,x_train.dtype)
print(x_test.shape,x_test.dtype)


# In[3]:


# 이미지 데이터 정보 및 시각화
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(x_train.shape,y_train.shape)  # (60000, 28, 28) (60000,)
print(x_test.shape,y_test.shape)    # (10000, 28, 28) (10000,)
print(x_train[0].shape)             # (28, 28)
# print(x_train[0])
print(y_train[:30])

plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.imshow(x_train[0],cmap='gray')
plt.colorbar()


# In[4]:


# 이미지 정규화(normalization) : 0 to 255 ==> 0 to 1
# z = (x-min())/(max()-min())
x_train = x_train / 255.0
x_test = x_test / 255.0

# print(x_train[0])


# In[5]:


# 정규화 함수 직접 구현할 경우(여기서는 불필요)
# Z = (X-min())/(max()-min())
# 머신러닝 데이터 전처리
def normalizer(data):
    result = (data - np.min(data,axis=0))/(np.max(data,axis=0) - np.min(data,axis=0))
    return result
    
# print(np.min(x_train,axis=0))   # 0  ...
# print(np.max(x_train,axis=0))   # 255 ... 
# x_train = normalizer(x_train) 
# x_test = nomalizer(x_test)


# In[6]:


# 4차원으로 변환
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
print(x_train.shape,x_test.shape)


# In[7]:


# CNN 모델 구현

# Callback 클래스 구현
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.89):
            print('\nReached 87% accuracy so cancelling training!')
            self.model.stop_training = True
            
callbacks = myCallback()  # 클래스의 인스턴스 생성

model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)), 

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128,activation='relu'),    
    tf.keras.layers.Dense(units=10,activation='softmax')
])

# no one hot encoding - sparse
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()


# In[8]:


# 학습
history = model.fit(x_train,y_train,epochs=10,callbacks=[callbacks]) 


# In[9]:


# 평가

model.evaluate(x_test,y_test)


# In[ ]:




