#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[2]:


# 이미지 데이터 경로 설정
import os

base_dir = 'tmp/'

train_dir = os.path.join(base_dir, 'rps')
validation_dir = os.path.join(base_dir, 'rps-test-set')

# Directory with our training cat/dog pictures
train_rock_dir = os.path.join(train_dir, 'rock')
train_paper_dir = os.path.join(train_dir, 'paper')
train_scissors_dir = os.path.join(train_dir, 'scissors')

# Directory with our validation cat/dog pictures
validation_rock_dir = os.path.join(validation_dir, 'rock')
validation_paper_dir = os.path.join(validation_dir, 'paper')
validation_scissors_dir = os.path.join(validation_dir, 'scissors')

train_rock_fnames = os.listdir( train_rock_dir )
train_paper_fnames = os.listdir( train_paper_dir )
train_scissors_fnames = os.listdir( train_scissors_dir )

print(train_rock_fnames[:10])
print(train_paper_fnames[:10])
print(train_scissors_fnames[:10])

print('total training rock images :', len(os.listdir(train_rock_dir ) ))
print('total training paper images :', len(os.listdir(train_paper_dir ) ))
print('total training scissors images :', len(os.listdir(train_scissors_dir ) ))
print('total validation cat images :', len(os.listdir( validation_rock_dir ) ))
print('total validation paper images :', len(os.listdir( validation_paper_dir ) ))
print('total validation scissors images :', len(os.listdir( validation_scissors_dir ) ))


# In[3]:


# 데이터 시각화 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 15  # 시작 인덱스

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*2, nrows*2)
pic_index += 8
next_rock_pix = [os.path.join(train_rock_dir, fname) 
                for fname in train_rock_fnames[pic_index-8:pic_index]]
next_paper_pix = [os.path.join(train_paper_dir, fname) 
                for fname in train_paper_fnames[pic_index-8:pic_index]]
next_scissors_pix = [os.path.join(train_scissors_dir, fname) 
                for fname in train_scissors_fnames[pic_index-8:pic_index]]


# print(next_horse_pix)
# print(next_human_pix)

for i, img_path in enumerate(next_rock_pix+next_scissors_pix):  # 16회 반복
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


# In[4]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')    
])
model.compile(optimizer=RMSprop(learning_rate=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.summary()
# sparse 안써도 됨?


# In[5]:


# 이미지 증강
# rescale: 1./255

train_datagen = ImageDataGenerator(rescale=1./255)
#print(type(train_datagen))

validation_datagen = ImageDataGenerator(rescale=1./255)

#train data의 generator
train_generator = train_datagen.flow_from_directory(
                  train_dir, 
                  target_size=(150,150), # resize될 크기
                  batch_size=20,
                  class_mode='categorical' 
)

#validation data의 generator
validation_generator = validation_datagen.flow_from_directory(
                  validation_dir, 
                  target_size=(150,150), # resize될 크기
                  batch_size=20,
                  class_mode='categorical' 
)

print(train_generator)


# In[6]:


# 학습
history = model.fit(
                train_generator, # x
                steps_per_epoch=126, # 2520/20 = 8.xxx, train image number/train batch_size
                epochs=15,
                validation_data=validation_generator,
                validation_steps=18 # 372/20
)


# In[9]:


# 학습 결과 시각화
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history['accuracy' ]
val_acc  = history.history['val_accuracy' ]
loss     = history.history['loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[17]:


import numpy as np
from tensorflow.keras.preprocessing import image
def image_predict(file_name):
    path = 'tmp/rps-test-set/' + file_name
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)   # 2차원 ndarray로 변환
    x = np.expand_dims(x, axis=0) # 3차원으로 변환
    images = np.vstack([x])       # 4차원으로 변환
    classes = model.predict(images, batch_size=10)
    print(classes)
    if (tf.argmax(classes, axis=1)==0):
        print('paper')
    if (tf.argmax(classes, axis=1)==1):
        print('rock')
    if (tf.argmax(classes, axis=1)==2):
        print('scissors')
        
image_predict('paper/testpaper01-00.png')        
image_predict('rock/testrock01-00.png')        
image_predict('scissors/testscissors01-00.png')   



# In[ ]:




