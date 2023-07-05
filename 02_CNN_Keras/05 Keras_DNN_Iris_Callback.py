#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.random.set_seed(5)


# In[2]:


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


# In[3]:


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


# In[4]:


# # one_hot_encoding
# nb_classes = 3

# y_one_hot = tf.one_hot(y_train, nb_classes) #3차원 (70, 1, 7), Rank = 3
# y_one_hot = tf.reshape(y_one_hot,[-1,nb_classes]) # 3차원에서 2차원으로 변환

# print(y_one_hot.shape)
# print(y_one_hot)

# # test data -> one_hot_encoding
# y_test_one_hot = tf.one_hot(y_test, nb_classes) #3차원 (70, 1, 7), Rank = 3
# y_test_one_hot = tf.reshape(y_test_one_hot,[-1,nb_classes]) # 3차원에서 2차원으로 변환

# print(y_test_one_hot.shape)
# print(y_test_one_hot)


# In[5]:


y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)
# print(y_train.shape)
# print(y_train)
# print(y_test.shape)
# print(y_test)


# In[6]:


# Callback 클래스 구현
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.9):
            print('\nReached 90% accuracy so cancelling training!')
            self.model.stop_training = True
            
callbacks = myCallback()  # 클래스의 인스턴스 생성

# Dense Layer 구현 [2층 신경망]
model = tf.keras.Sequential([
    # (105,4) * (4,20) = (105,20)   , W:4*20=80, b=20 --> Params : 100
    tf.keras.layers.Dense(units=20,activation='relu',input_shape=(4,)) ,
    # (4,2) * (2,1) = (4,1)   , W:2*1, b=1 --> Params : 3
    tf.keras.layers.Dense(units=3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[7]:


# 학습 verbose=1, 메세지를 출력
history = model.fit(x_train,y_train,epochs=700,batch_size=1,verbose=1,callbacks=[callbacks]) 


# In[8]:


history.history


# In[9]:


# 시각화_accuracy
epoch_count = range(1, len(history.history['accuracy']) + 1)
plt.plot(epoch_count, history.history['accuracy'], 'r-')
plt.legend(['Training Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# In[10]:


# 예측
model.predict(x_test)


# In[11]:


# 평가
model.evaluate(x_test,y_test)


# In[12]:


# Callback 클래스 구현
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.9):
            print('\nReached 90% accuracy so cancelling training!')
            self.model.stop_training = True
            
callbacks = myCallback()  # 클래스의 인스턴스 생성

# Dense Layer 구현 [3층 신경망]
model = tf.keras.Sequential([
    # (105,4) * (4,20) = (105,20)   , W:4*20=80, b=20 --> Params : 100
    tf.keras.layers.Dense(units=40,activation='relu',input_shape=(4,)) ,
    # (4,2) * (2,1) = (4,1)   , W:2*1, b=1 --> Params : 3
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[13]:


# 학습 verbose=1, 메세지를 출력
history = model.fit(x_train,y_train,epochs=700,batch_size=1,verbose=1,callbacks=[callbacks]) 


# In[14]:


history.history


# In[15]:


# 시각화_accuracy
epoch_count = range(1, len(history.history['accuracy']) + 1)
plt.plot(epoch_count, history.history['accuracy'], 'r-')
plt.legend(['Training Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()


# In[16]:


# 예측
model.predict(x_test)


# In[17]:


# 평가
model.evaluate(x_test,y_test)


# In[ ]:




