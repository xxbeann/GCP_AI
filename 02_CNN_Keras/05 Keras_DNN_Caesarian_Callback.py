#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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

print(x_train.shape)
print(y_train.shape)


# In[3]:


# Callback 클래스 구현
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss') < 0.53):
            print('\nReached under 53% loss so cancelling training!')
            self.model.stop_training = True
            
callbacks = myCallback()  # 클래스의 인스턴스 생성

# Dense Layer 구현 [2층 신경망]
model = tf.keras.Sequential([
    # (56,5) * (5,20) = (56,20)   , W:5*20=100, b=20 --> Params : 120
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(5,)) ,
    # (4,2) * (2,1) = (4,1)   , W:2*1, b=1 --> Params : 3
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[4]:


# 학습 verbose=1, 메세지를 출력
history = model.fit(x_train,y_train,epochs=700,callbacks=[callbacks]) 


# In[5]:


history.history


# In[6]:


# 시각화_loss
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()


# In[7]:


# 예측
model.predict(x_test)


# In[8]:


# 평가
model.evaluate(x_test,y_test)


# In[9]:


# Callback 클래스 구현
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('loss') < 0.53):
            print('\nReached under 53% loss so cancelling training!')
            self.model.stop_training = True
            
callbacks = myCallback()  # 클래스의 인스턴스 생성

# Dense Layer 구현 [3층 신경망]
model = tf.keras.Sequential([
    # (56,5) * (5,20) = (56,20)   , W:5*20=100, b=20 --> Params : 120
    tf.keras.layers.Dense(units=40,activation='relu',input_shape=(5,)) ,
    # (4,2) * (2,1) = (4,1)   , W:2*1, b=1 --> Params : 3
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[10]:


# 학습 verbose=1, 메세지를 출력
history = model.fit(x_train,y_train,epochs=700,callbacks=[callbacks]) 


# In[11]:


# 시각화_accuracy
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training loss'])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.show()


# In[12]:


# 예측
model.predict(x_test)


# In[13]:


# 평가
model.evaluate(x_test,y_test)


# In[ ]:




