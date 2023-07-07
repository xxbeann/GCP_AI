#!/usr/bin/env python
# coding: utf-8

# ### 08 Horse or Human

# In[1]:


# horse-or-human classification model with CNN
# train : 1027 images [horse(500) + human(527)] , 300*300 pixels
# validation : 256 images [horse(128) + human(128)] , 300*300 pixels


# In[2]:


import tensorflow as tf
import urllib
import zipfile


# In[3]:


# horse-or-human 데이터셋 다운로드
_TRAIN_URL = "https://storage.googleapis.com/download.tensorflow.org/data/horse-or-human.zip"
_TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/validation-horse-or-human.zip"
urllib.request.urlretrieve(_TRAIN_URL, 'horse-or-human.zip')
urllib.request.urlretrieve(_TEST_URL, 'validation-horse-or-human.zip')

# 새로운 데이터 다운로드 주소
# https://storage.googleapis.com/learning-datasets/horse-or-human.zip


# In[4]:


# 압축해제
local_zip = 'horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('tmp/horse-or-human/')
zip_ref.close()

local_zip = 'validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('tmp/validation-horse-or-human/')
zip_ref.close()

train_dir = 'tmp/horse-or-human/'
validation_dir = 'tmp/validation-horse-or-human/'


# In[8]:


# 이미지 데이터 경로 설정
import os

# Directory with our training horse pictures
train_horse_dir = os.path.join('tmp/horse-or-human/horses')

# Directory with our training human pictures
train_human_dir = os.path.join('tmp/horse-or-human/humans')

# Directory with our training horse pictures
validation_horse_dir = os.path.join('tmp/validation-horse-or-human/horses')

# Directory with our training human pictures
validation_human_dir = os.path.join('tmp/validation-horse-or-human/humans')

train_horse_names = os.listdir(train_horse_dir)
# print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
# print(train_human_names[:10])

validation_horse_names = os.listdir(validation_horse_dir)
# print(validation_horse_names[:10])

validation_human_names = os.listdir(validation_human_dir)
# print(validation_human_names[:10])


# In[9]:


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
next_horse_pix = [os.path.join(train_horse_dir, fname) 
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) 
                for fname in train_human_names[pic_index-8:pic_index]]

# print(next_horse_pix)
# print(next_human_pix)

for i, img_path in enumerate(next_horse_pix+next_human_pix):  # 16회 반복
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


# ### 데이터 전처리 : 이미지 증강(Image Augmentation)

# In[16]:


# ! pip install Augmentor
import Augmentor

# ! mkdir augmentation_test

# 증강 시킬 이미지 폴더 경로
img = Augmentor.Pipeline('augmentation_test/')

# 좌우 반전
img.flip_left_right(probability=1.0)

# 상하 반전
img.flip_top_bottom(probability=1.0)

# 왜곡
img.random_distortion(probability=1.0,grid_width=10,grid_height=10,magnitude=8)

# 증강 이미지수 --> 증강 실행
img.sample(100)


# ### CNN model 구현

# In[29]:


# Conv2d - maxpool2d * 5회 16-32-64-64-64 , filter(3,3), strides:1
#                                           poolsize : (2,2), strides:2
# Dense * 2회            Hidden layer output : 512
# 출력층 binary categry, output:1

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512,activation='relu'),    
    tf.keras.layers.Dense(units=1,activation='sigmoid')
])
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# ### 텐서플로 이미지 증강 구현

# In[33]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
#print(type(train_datagen))

validation_datagen = ImageDataGenerator(rescale=1./255)

#train data의 generator
train_generator = train_datagen.flow_from_directory(
                  train_dir, 
                  target_size=(300,300), # resize될 크기
                  batch_size=128,
                  class_mode='binary' # 2진 분류
)

#validation data의 generator
validation_generator = validation_datagen.flow_from_directory(
                  validation_dir, 
                  target_size=(300,300), # resize될 크기
                  batch_size=32,
                  class_mode='binary' # 2진 분류
)

print(train_generator)


# In[34]:


# 학습
history = model.fit(
                train_generator, # x
                steps_per_epoch=8, # 1027/128 = 8.xxx, train image number/train batch_size
                epochs=15,
                validation_data=validation_generator,
                validation_steps=8 # 256/31
)


# In[36]:


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

# In[45]:


# 참고 소스
import numpy as np
from tensorflow.keras.preprocessing import image
def image_predict(file_name):
    path = 'horse-or-human예측용이미지/' + file_name
    img = image.load_img(path, target_size=(300, 300))
    x = image.img_to_array(img)   # 2차원 ndarray로 변환
    x = np.expand_dims(x, axis=0) # 3차원으로 변환
    images = np.vstack([x])       # 4차원으로 변환
    classes = model.predict(images, batch_size=10)
    if classes[0] == 1.0:
        print(file_name ,': human')
    else:
        print(file_name ,': horse')
        
image_predict('말01.jpg')        
image_predict('말02.jpg') 
image_predict('기린.jpg')
image_predict('말과사람.jpg')
image_predict('말과사람02.jpg')
image_predict('말과사람06.jpg')
image_predict('사람01.jpg')
image_predict('사람02.jpg')
image_predict('사람12.jpg')
image_predict('말타기게임하는사람.jpg')


# ### 모델 개선

# In[46]:


# ! mkdir tmp\saved_train_image
# ! mkdir tmp\saved_val_image


# In[50]:


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
                  target_size=(300,300), # resize될 크기
                  batch_size=20,
                  class_mode='binary', # 2진 분류
                  save_to_dir='tmp/saved_train_image'
)

#validation data의 generator
validation_generator = validation_datagen.flow_from_directory(
                  validation_dir, 
                  target_size=(300,300), # resize될 크기
                  batch_size=20,
                  class_mode='binary', # 2진 분류
                  save_to_dir='tmp/saved_val_image'
)

print(train_generator)

# DirectoryIterator object  , 실제 사용 시점(학습시)에 데이터 생성
# fit 호출시 1 epoch 마다 train 이미지 1027개 생성  : 15 epochs 일 경우 총 15*1027 증강 이미지 생성
# fit 호출시 1 epoch 마다 validation 이미지 256개 생성  : 15 epochs 일 경우 총 15*256 증강 이미지 생성


# In[51]:


# 모델 구현
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')    
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[52]:


# 학습
history = model.fit(
                train_generator, # x
                steps_per_epoch=50, # 1027/20 = 8.xxx, train image number/train batch_size
                epochs=10,
                verbose=1,
                validation_data=validation_generator,
                validation_steps=12 # 256/20
)


# In[ ]:





# In[ ]:




