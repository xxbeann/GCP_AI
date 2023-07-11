#!/usr/bin/env python
# coding: utf-8

# ### 03 Sarcasm_분류모델

# In[1]:


# sarcasm json data binary classification
# total 26,709 headlines


# In[20]:


import json
import tensorflow as tf
import urllib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[3]:


# sarcasm.json 데이터셋 파일 다운로드 
url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
urllib.request.urlretrieve(url, 'sarcasm.json')


# In[4]:


# 데이터 파일 불러오기
with open("sarcasm.json", 'r') as f:
    datastore = json.load(f)
    
sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])


# In[5]:


# 데이터 프레임으로 보기(참고)
import pandas as pd
df = pd.DataFrame(datastore)
df = df.iloc[:,1:]

print(type(datastore[0]))  # <class 'dict'>
print(df.shape)            # (26709, 2)
print(df['is_sarcastic'].value_counts())   # 0    14985 : not sarcastic 
                                           # 1    11724 : sarcastic
df.head(10)


# ### 텍스트 전처리

# In[6]:


# 전처리를 위한 변수 설정
vocab_size = 10000    # 토큰화에 사용될 최대 어휘수
embedding_dim = 16    # Embedding 계층의 output size
max_length = 100      # 한 문장의 길이, 데이터 셋의 길이, maxlen, sequence length
trunc_type = 'post'   # remove values from sequences larger than 'maxlen'
padding_type = 'post' # padding 방식
oov_tok = '<OOV>'     # out of vocabulary 단어 집합에 없는 단어
training_size = 20000 # 학습 데이터 갯수


# In[13]:


# train(20000)/test(6709) data split

training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]


# In[18]:


# 토큰나이저를 시행하여 단어를 숫자값, 인덱스로 변환하여 저장
# 가장 빈도가 높은 10000개의 단어들만 사용하여 토큰화

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_tok)
# print(tokenizer)

# 단어 인덱스를 생성
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
# print(word_index)

# 문자열을 정수 인덱스의 리스트로 변환 : 정수 인코딩
training_sequences = tokenizer.texts_to_sequences(training_sentences)
print(type(training_sequences)) # class list

# 패딩, 벡터 표현을 얻음 : 신경망에 입력할 X값, ndarray로 출력
training_padded = pad_sequences(training_sequences,maxlen=max_length,padding=padding_type,
                               truncating=trunc_type)

# test용 단어 인덱스를 따로 만들지 않고 train용 인덱스를 사용
# test 데이터 : 정수 인덱스의 리스트로 변환, 정수 인코딩
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

# test 데이터 패딩, 벡터 표현을 얻음, ndarray로 출력
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                              padding=padding_type,truncating=trunc_type)

print(training_padded.shape)
print(testing_padded.shape)


# In[22]:


# list를 ndarray로 모두 변환

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)
print(training_labels.shape)
print(testing_labels.shape)


# In[24]:


# 학습모델
model = tf.keras.Sequential([
    
    # input : (none,100), output : (None,100,16)
    # Embedding layer Params : 10000*16 = 160000
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# In[25]:


# 학습
num_epochs = 30
history = model.fit(training_padded,training_labels,epochs=num_epochs,validation_data=(testing_padded,testing_labels))


# In[26]:


# 시각화
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# In[27]:


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_sentence(training_padded[2]))
print(training_sentences[2])
print(labels[2])


# In[28]:


reverse_word_index


# In[30]:


e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)


# In[32]:


# 예측

sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night"]

sequences = tokenizer.texts_to_sequences(sentence) # 정수 인코딩, 리스트
padded = pad_sequences(sequences,maxlen=max_length,padding=padding_type,truncating=trunc_type)
preds = model.predict(padded)
print(preds)
print(np.round(preds))


# ### 모델개선
# 

# In[38]:


# 단방향 LSTM
model = tf.keras.Sequential([
    # input : (None,100), output : (None,100,16)
    # Embedding layer Params : 10000*16 = 160000
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.LSTM(64,return_sequences=True),
    tf.keras.layers.LSTM(32),
    # tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.summary()


# In[39]:


# 양방향 LSTM : 출력이 hidden_size의 2배가 된다.
model = tf.keras.Sequential([
    # input : (None,100), output : (None,100,16)
    # Embedding layer Params : 10000*16 = 160000
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
model.summary()


# In[40]:


# 학습
num_epochs = 30
history = model.fit(training_padded,training_labels,epochs=num_epochs,validation_data=(testing_padded,testing_labels))


# In[41]:


# 성능 약간 개선된


# ### Keras Tokenizer 설명

# In[47]:


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words=100,oov_token='<OOV>')
tokenizer.fit_on_texts(sentences) # 정수 인덱스를 생성
word_index = tokenizer.word_index
print('\nWord Index =', word_index)

# 정수 인코딩
sequences = tokenizer.texts_to_sequences(sentences)
print('Sequences =', sequences)

# 패딩
padded = pad_sequences(sequences,maxlen=5)
print('/nPadded Sequences:')
print(padded)


# In[49]:


# 아래문장을 정수 인코딩하고 문장의 길이를 10으로 패딩하세요
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

tokenizer = Tokenizer(num_words=100,oov_token='<OOV>')
tokenizer.fit_on_texts(test_data) # 정수 인덱스를 생성
word_index = tokenizer.word_index
print('\nWord Index =', word_index)

# 정수 인코딩
sequences = tokenizer.texts_to_sequences(test_data)
print('Sequences =', sequences)

# 패딩
padded = pad_sequences(sequences,maxlen=10)
print('/nPadded Sequences:')
print(padded)


# In[ ]:




