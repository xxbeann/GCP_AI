#!/usr/bin/env python
# coding: utf-8

# ### 텍스트 생성

# In[1]:


import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np 

import urllib


# In[2]:


# Colab용 
# !wget --no-check-certificate \
#     https://storage.googleapis.com/learning-datasets/irish-lyrics-eof.txt \
#     -O /tmp/irish-lyrics-eof.txt


# In[3]:


# irish-lyrics-eof.txt 데이터셋 파일 다운로드 
url = 'https://storage.googleapis.com/learning-datasets/irish-lyrics-eof.txt'
urllib.request.urlretrieve(url, 'irish-lyrics-eof.txt')


# In[4]:


# 데이터 불러오기
data = open('irish-lyrics-eof.txt').read()
# print(data)

corpus = data.lower().split('\n')
# corpus

# 토큰화
tokenizer = Tokenizer()

tokenizer.fit_on_texts(corpus)  # 정수 인덱스
total_words = len(tokenizer.word_index) + 1  
print(total_words)           # 2690
print(tokenizer.word_index)


# In[5]:


input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # print(token_list)
    for i in range(1,len(token_list)):  # 1,2,3,4,5,6
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
    
# 패딩
max_sequence_len = max([len(x) for x in input_sequences])  # 16
input_sequences = pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre')
# input_sequences

# X,Y(label)
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

# 원핫인코딩
ys = tf.keras.utils.to_categorical(labels,num_classes=total_words)

print(xs)
print(xs.shape)   # (12038, 15)
print(labels)
print(ys.shape)   # (12038, 2690)
print(ys)


# In[6]:


print(tokenizer.word_index['in'])
print(tokenizer.word_index['the'])
print(tokenizer.word_index['town'])
print(tokenizer.word_index['of'])
print(tokenizer.word_index['athy'])
print(tokenizer.word_index['one'])
print(tokenizer.word_index['jeremy'])
print(tokenizer.word_index['lanigan'])


# In[7]:


print(xs[5])
print(ys[5])


# In[8]:


# 모델 구현
model = Sequential() 
model.add(Embedding(total_words,100,input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words,activation='softmax'))

model.compile(optimizer = Adam(learning_rate=0.01),
             loss='categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[9]:


# 학습 : 약 3~4 분소요(GPU)
history = model.fit(xs,ys,epochs=100)


# In[11]:


# 시각화
import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()
    
plot_graphs(history, 'accuracy')


# In[12]:


# 텍스트(문장) 생성
seed_text = "I've got a bad feeling about this"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0] # 인코딩
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    # predicted = model.predict_classes(token_list) # tf 2.7에서 오류
    predicted = model.predict(token_list,verbose=0)
    predicted = tf.argmax(predicted,-1)
    output_word = ''
    
    for word,index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += ' ' + output_word
#     print(token_list)
#     print(predicted)
#     print(output_word)
#     print(seed_text)
#     input()

print(seed_text)


# In[ ]:




