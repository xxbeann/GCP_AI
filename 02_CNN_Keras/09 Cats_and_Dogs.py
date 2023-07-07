#!/usr/bin/env python
# coding: utf-8

# ### 09 cats and dog

# In[1]:


# cats_and_dogs classification model with CNN
# train : 2000 images [cat(1000) + dog(1000)]  , size는 다름
# validation : 1000 images [cat(500) + dog(500)] , size는 다름


# In[3]:


import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[4]:


# cats_and_dogs 데이터셋 다운로드 , Windows용
_TRAIN_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
urllib.request.urlretrieve(_TRAIN_URL, 'tmp/cats_and_dogs_filtered.zip')


# In[6]:


#압축해제
local_zip = 'tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('tmp/')
zip_ref.close()


# In[7]:


# 이미지 데이터 경로 설정
import os

base_dir = 'tmp/cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

print(train_cat_fnames[:10])
print(train_dog_fnames[:10])

print('total training cat images :', len(os.listdir(train_cats_dir ) ))
print('total training dog images :', len(os.listdir(train_dogs_dir ) ))

print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))


# In[11]:


# 데이터 시각화
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

pic_index = 0 # Index for iterating over images ,시작 인덱스

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols*3, nrows*3)

pic_index+=8

next_cat_pix = [os.path.join(train_cats_dir, fname) 
                for fname in train_cat_fnames[ pic_index-8:pic_index] 
               ]
next_dog_pix = [os.path.join(train_dogs_dir, fname) 
                for fname in train_dog_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off') # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


# ### CNN model 구현

# In[17]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')    
])
model.compile(optimizer=RMSprop(learning_rate=0.001),
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.summary()


# ### 텐서플로 이미지 증강 구현

# In[18]:


# rescale: 1./255

train_datagen = ImageDataGenerator(rescale=1./255)
#print(type(train_datagen))

validation_datagen = ImageDataGenerator(rescale=1./255)

#train data의 generator
train_generator = train_datagen.flow_from_directory(
                  train_dir, 
                  target_size=(150,150), # resize될 크기
                  batch_size=20,
                  class_mode='binary' # 2진 분류
)

#validation data의 generator
validation_generator = validation_datagen.flow_from_directory(
                  validation_dir, 
                  target_size=(150,150), # resize될 크기
                  batch_size=20,
                  class_mode='binary' # 2진 분류
)

print(train_generator)


# In[19]:


# 학습
history = model.fit(
                train_generator, # x
                steps_per_epoch=100, # 2000/20 = 8.xxx, train image number/train batch_size
                epochs=15,
                validation_data=validation_generator,
                validation_steps=50 # 1000/20
)


# In[20]:


# 학습 결과 시각화
#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
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


# ### 이미지 분류 예측

# In[23]:


import numpy as np
from tensorflow.keras.preprocessing import image
def image_predict(file_name):
    path = 'cats_and_dogs예측용이미지/' + file_name
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)   # 2차원 ndarray로 변환
    x = np.expand_dims(x, axis=0) # 3차원으로 변환
    images = np.vstack([x])       # 4차원으로 변환
    classes = model.predict(images, batch_size=10)
    # print(classes[0],end=' ')
    if classes[0] == 1.0:
        print(file_name ,': dog')
    else:
        print(file_name ,': cat')
image_predict('cat_01.jpg')        
image_predict('cat_02.jpg')        
image_predict('cat_03.jpg')        
image_predict('cat_04.jpg')        
image_predict('cat_05.jpg') 

image_predict('dog_01.jpg')        
image_predict('dog_02.jpg')        
image_predict('dog_03.jpg')        
image_predict('dog_04.jpg')        
image_predict('dog_05.jpg')

image_predict('cat_dog_01.jpg')   
image_predict('cat_dog_02.jpg')   
image_predict('cat_dog_03.jpg')   
image_predict('cat_dog_04.jpg')   
image_predict('cat_dog_05.jpg')


# ### 모델 개선

# In[24]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')    
])
model.compile(optimizer=RMSprop(learning_rate=0.0001), # 1e-4
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.summary()


# In[25]:


# 이미지 증강

train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)
#print(type(train_datagen))

validation_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=40,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

#train data의 generator
train_generator = train_datagen.flow_from_directory(
                  train_dir, 
                  target_size=(150,150), # resize될 크기
                  batch_size=20,
                  class_mode='binary', # 2진 분류
                  
)

#validation data의 generator
validation_generator = validation_datagen.flow_from_directory(
                  validation_dir, 
                  target_size=(150,150), # resize될 크기
                  batch_size=20,
                  class_mode='binary', # 2진 분류
                  
)

print(train_generator)


# In[26]:


# 학습
history = model.fit(
                train_generator, # x
                steps_per_epoch=100, # 2000/20 = 8.xxx, train image number/train batch_size
                epochs=30,
                validation_data=validation_generator,
                validation_steps=50 # 1000/20
)

# 성능이 약간 개선됨
# 100 epochs로 학습
# 기타 : 이미지의 shape을 (300,300,3)로 할 경우 정확도 감소됨


# In[27]:


# 시각화
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:




