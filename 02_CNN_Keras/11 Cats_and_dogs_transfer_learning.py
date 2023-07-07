#!/usr/bin/env python
# coding: utf-8

# ### 11 Cats_and_dogs_transfer_learning

# In[1]:


# cats_and_dogs classification model with CNN
# train : 2000 images [cat(1000) + dog(1000)]  , size는 다름
# validation : 1000 images [cat(500) + dog(500)] , size는 다름


# In[2]:


import tensorflow as tf
import urllib
import zipfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras import layers
from tensorflow.keras import Model


# In[3]:


# Google InceptionV3 pretrained model 가중치 다운로드
_INCEPTION_URL = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
urllib.request.urlretrieve(_INCEPTION_URL, 'tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[4]:


# Colab용 : Linux/GPU사용 
# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5


# In[5]:


# Inception model(pre-trained model) 불러오기

from tensorflow.keras.applications.inception_v3 import InceptionV3

locals_weights_file = 'tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape=(150,150,3),
                               include_top=False,
                               weights=None)

pre_trained_model.load_weights(locals_weights_file)

print(len(pre_trained_model.layers)) #311 

# 사전 훈련 계층의 가중치의 훈련(학습) 가능 여부를 설정 : False(학습 안함) , 고정
for layer in pre_trained_model.layers:
    layer.trainable = False
    
pre_trained_model.summary()


# In[6]:


# 마지막 층 출력 확인하기(마지막 출력 층을 선택)
#  mixed7 (Concatenate)  :    (None, 7, 7, 768)
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape:', last_layer.output_shape)
last_output = last_layer.output


# In[7]:


# layer의 층 번호 구하기
# for i,layer in enumerate(pre_trained_model.layers): # 311회 반복
#     if(layer.name == 'mixed7'):
#         print(i+1)

# layer의 층 번호 구하기
pre_trained_model.layers.index(pre_trained_model.get_layer('mixed7')) + 1


# In[8]:


# 모델 구성  : pre_trained model layer(229)  + 4  --> 233 layers

# (1) Flatten layer
x = tf.keras.layers.Flatten()(last_output)

# (2) FC(=Dense) layer
x = layers.Dense(1024,activation='relu')(x)

# (3) Dropout layer
x = layers.Dropout(0.2)(x)

# (4) Output layer : 2진분류, sigmoid
x = layers.Dense(1,activation='sigmoid')(x)

model = Model(pre_trained_model.input,x)
model.compile(optimizer=RMSprop(learning_rate=0.0001), # 1e-4
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.summary()


# In[9]:


# cats_and_dogs 데이터셋 다운로드 , Windows용
_TRAIN_URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
urllib.request.urlretrieve(_TRAIN_URL, 'tmp/cats_and_dogs_filtered.zip')


# In[10]:


#압축해제
local_zip = 'tmp/cats_and_dogs_filtered.zip'
zip_ref = zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('tmp/')
zip_ref.close()


# In[11]:


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


# In[12]:


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


# In[13]:


# 학습
history = model.fit(
                train_generator, # x
                steps_per_epoch=100, # 2000/20 = 8.xxx, train image number/train batch_size
                epochs=20,
                validation_data=validation_generator,
                validation_steps=50 # 1000/20
)


# In[16]:


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

plt.plot(epochs, loss, 'g', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#  결과 : InceptionV3 pretrained model 사용 시 validation의 정확도가 크게 향상됨
#  val_accuracy: 0.7360 ---> val_accuracy: 0.9660


# In[ ]:




