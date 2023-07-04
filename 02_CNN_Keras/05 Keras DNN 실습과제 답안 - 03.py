#!/usr/bin/env python
# coding: utf-8

# ## Keras DNN 실습과제 답안 - 03
# ### Multi-Classification , 'iris.csv'

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
tf.random.set_seed(5)
np.random.seed(5)


# In[2]:


species_list =['"setosa"','"versicolor"','"virginica"']

xy = np.loadtxt('iris.csv',delimiter=',',dtype=np.str,skiprows=1)
xy.shape


# In[3]:


x_train = np.float32(xy[:35,1:-1])
x_train = np.append(x_train , np.float32(xy[50:85,1:-1]),0)
x_train = np.append(x_train , np.float32(xy[100:135,1:-1]),0) # [105,4]

y_train = xy[:35,[-1] ]
y_train = np.append(y_train, xy[50:85,[-1]],0)
y_train = np.append(y_train, xy[100:135,[-1]],0) # [105,1]

for i in range(105):
    y_train[i,-1] = np.int32(species_list.index(y_train[i,-1]))
print(x_train.shape,y_train.shape)


# In[4]:


x_test = np.float32(xy[35:50,1:-1])
x_test = np.append(x_test , np.float32(xy[85:100,1:-1]),0)
x_test = np.append(x_test , np.float32(xy[135:,1:-1]),0) # [45,4]

y_test = xy[35:50,[-1] ]
y_test = np.append(y_test, xy[85:100,[-1]],0)
y_test = np.append(y_test, xy[135:,[-1]],0) # [45,1]

for i in range(45):
    y_test[i,-1] = np.int32(species_list.index(y_test[i,-1]))

print(x_test.shape, y_test.shape)


# In[5]:


# one-hot 인코딩
nb_classes = 3

# y_train = np.array(y_train,dtype=np.int32)   # Keras에서는 정수로 변환 불필요
y_train = tf.keras.utils.to_categorical(y_train, num_classes=nb_classes)

# y_test = np.array(y_test,dtype=np.int32)     # Keras에서는 정수로 변환 불필요
y_test = tf.keras.utils.to_categorical(y_test, num_classes=nb_classes)


# In[6]:


# 모델 구현
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=20, input_shape=[4,],activation='relu'),
    tf.keras.layers.Dense(units=3,activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[7]:


# 학습
history = model.fit(x_train, y_train, epochs=700)


# In[8]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[9]:


# 평가

accuracy = model.evaluate(x_test, y_test)

print('Accuracy',accuracy[1])


# ## 모델 개선 : 3 layers

# In[10]:


tf.keras.backend.clear_session()
tf.random.set_seed(5)
np.random.seed(5)


# In[11]:


# 모델 구현
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=40, input_shape=[4,],activation='relu'),
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=3,activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[12]:


# 학습
history = model.fit(x_train, y_train, epochs=700)


# In[13]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[14]:


# 평가
accuracy = model.evaluate(x_test, y_test)

print('Accuracy',accuracy[1])


# ### 원핫 인코딩 하지 않고 모델 구현 : 'sparse_categorical_crossentropy'

# In[72]:


tf.keras.backend.clear_session()
tf.random.set_seed(5)
np.random.seed(5)


# In[73]:


# species_list =['"setosa"','"versicolor"','"virginica"']
# xy = np.loadtxt('iris.csv',delimiter=',',dtype=np.str,skiprows=1)

x_train = np.float32(xy[:35,1:-1])
x_train = np.append(x_train , np.float32(xy[50:85,1:-1]),0)
x_train = np.append(x_train , np.float32(xy[100:135,1:-1]),0) # [105,4]

y_train = xy[:35,[-1] ]
y_train = np.append(y_train, xy[50:85,[-1]],0)
y_train = np.append(y_train, xy[100:135,[-1]],0) # [105,1]

for i in range(105):
    y_train[i,-1] = np.int32(species_list.index(y_train[i,-1]))

y_train = y_train.astype(np.int32)

print(x_train.shape,y_train.shape)


# In[74]:


x_test = np.float32(xy[35:50,1:-1])
x_test = np.append(x_test , np.float32(xy[85:100,1:-1]),0)
x_test = np.append(x_test , np.float32(xy[135:,1:-1]),0) # [45,4]

y_test = xy[35:50,[-1] ]
y_test = np.append(y_test, xy[85:100,[-1]],0)
y_test = np.append(y_test, xy[135:,[-1]],0) # [45,1]

for i in range(45):
    y_test[i,-1] = np.int32(species_list.index(y_test[i,-1]))

y_test = y_test.astype(np.int32)

print(x_test.shape, y_test.shape)


# In[75]:


# 모델 구현
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=40, input_shape=[4,],activation='relu'),
    tf.keras.layers.Dense(units=20, activation='relu'),
    tf.keras.layers.Dense(units=3,activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[76]:


# 학습
history = model.fit(x_train, y_train, epochs=700)


# In[77]:


# 시각화
epoch_count = range(1, len(history.history['loss']) + 1)
plt.plot(epoch_count, history.history['loss'], 'r-')
plt.legend(['Training Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# In[78]:


# 평가
accuracy = model.evaluate(x_test, y_test)

print('Accuracy',accuracy[1])


# In[ ]:





# In[ ]:




