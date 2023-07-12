#!/usr/bin/env python
# coding: utf-8

# ### 감성분석(Sentiment Analysis)

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


# Naver Sentiment Movie Corpus v1.0 다운로드
path_to_train_file = tf.keras.utils.get_file('train.txt', 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt')
path_to_test_file = tf.keras.utils.get_file('test.txt', 'https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt')


# In[3]:


path_to_train_file


# In[4]:


path_to_test_file


# In[5]:


# 데이터를 메모리에 불러옵니다. encoding 형식으로 utf-8 을 지정해야합니다.
train_text = open(path_to_train_file, 'rb').read().decode(encoding='utf-8')
test_text = open(path_to_test_file, 'rb').read().decode(encoding='utf-8')
print('train text length:',len(train_text),' characters')
print('test  text length:',len(test_text),' characters')
print('\n')
print(train_text[:5])
print(train_text[:300])


# In[6]:


# import pandas as pd
# df_train = pd.read_csv(path_to_train_file,sep='\t')
# df_train


# In[7]:


# 각 문장을 '\n'으로 분리 -> 헤더제외 -> '\t'으로 분리 -> 마지막 문자 정수로 변환하여 2차원으로 변환
train_Y = np.array([[int(row.split('\t')[2])] for row in train_text.split('\n')[1:] if row.count('\t') > 0])
test_Y = np.array([[int(row.split('\t')[2])] for row in test_text.split('\n')[1:] if row.count('\t') > 0])
print(train_Y.shape,test_Y.shape)  # (150000, 1) (50000, 1)
print(train_Y[:5])


# In[8]:


# X 값을 추출
train_text_X = [row.split('\t')[1] for row in train_text.split('\n')[1:] if row.count('\t') > 0]
print(len(train_text_X))  # 150000
train_text_X[:5]


# ### 텍스트 전처리

# In[9]:


# train 데이터의 입력(X)에 대한 정제(Cleaning)
import re
# From https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
def clean_str(string):    
    string = re.sub(r"[^가-힝A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'{2,}", "\'", string)
    string = re.sub(r"\'", "", string)

    return string.lower()

train_text_X = [clean_str(sentence) for sentence in train_text_X]
train_text_X[:5]    


# In[10]:


# 문장을 띄어쓰기 단위로 단어 분리
sentences = [sentence.split(' ') for sentence in train_text_X]
sentences[:5]


# In[11]:


# 150000개 문장의 단어 길이 확인
import matplotlib.pyplot as plt
sentence_len = [len(sentence) for sentence in sentences]
sentence_len.sort()
plt.plot(sentence_len)
plt.xlabel('index')
plt.ylabel('count')
plt.show()
print(sum([int(l<=25)for l in sentence_len])) 
# 15 만개 문장 중 단어 갯수가 25개 이하인 문장의 수가 142587개이다, 25를 sequence_length로 사용


# In[12]:


# 단어 정제 및 문장 길이 줄임
# 문장의 최대 길이를 25로 잡고 그 이상은 잘라내고 단어의 앞에서 5글자씩만 잘라서 사용한다
sentences_new =[]
for sentence in sentences:
    sentences_new.append([word[:5] for word in sentence[:25]])
sentences = sentences_new
sentences[:5]


# In[13]:


#  Tokenizer와 pad_sequences를 사용한 문장 전처리
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 빈도수가 높은 20000개 단어만 사용하여 숫자로 반환하고 나머지는 공백으로 반환
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(sentences)  # 정수 인덱스
word_index = tokenizer.word_index
# print(len(word_index))

# 정수 인코딩
sequences = tokenizer.texts_to_sequences(sentences)
# sequences[:10]

# 패딩
train_X = pad_sequences(sequences,padding='post')
print(train_X.shape)  # (150000, 25)
train_X[:10]


# In[14]:


# Tokenizer의 동작 확인
print(tokenizer.index_word[19999])
print(tokenizer.index_word[20000])
temp = tokenizer.texts_to_sequences(['#$#$#', '경우는', '잊혀질', '연기가']) # 빈도가 낮은 '잊혀질'은 공백으로 반환
print(temp)
temp = pad_sequences(temp, padding='post')
print(temp)  # 최대길이가 1이므로 1보다 작은 문장은 0이 패딩된다


# ### 학습 모델 구현

# In[15]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(20000,300,input_length=25),
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(2,activation='softmax')
])

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.summary()


# In[16]:


# 학습
history = model.fit(train_X,train_Y,epochs=5,batch_size=128,validation_split=0.2)


# In[17]:


history.history['loss']
history.history['val_loss']
history.history['accuracy']
history.history['val_accuracy']


# In[18]:


# 감성 분석 모델 학습 결과 확인
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='loss')
plt.plot(history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'g-', label='accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()


# In[19]:


## 정확도 측정

# X 값을 추출
test_text_X = [row.split('\t')[1] for row in test_text.split('\n')[1:] if row.count('\t') > 0]
print(len(test_text_X))  # 50000

test_text_X = [clean_str(sentence) for sentence in test_text_X]
# test_text_X[:5] 

# 문장을 띄어쓰기 단위로 단어 분리
sentences = [sentence.split(' ') for sentence in test_text_X]
# sentences[:5]

# 문장의 최대 길이를 25로 잡고 그 이상은 잘라내고 단어의 앞에서 5글자씩만 잘라서 사용한다
sentences_new =[]
for sentence in sentences:
    sentences_new.append([word[:5] for word in sentence[:25]])
sentences = sentences_new
print(sentences_new[:5])

# 정수 인코딩
sequences = tokenizer.texts_to_sequences(sentences)
# sequences[:10]

# 패딩
test_X = pad_sequences(sequences,padding='post')
print(test_X.shape)  # (50000, 25)
test_X[:10]


# In[20]:


accr = model.evaluate(test_X,test_Y,verbose=0)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))  # Accuracy: 0.801


# ### 예측

# In[21]:


# 임의의 문장 감성 분석 예측 결과 확인
test_sentence = '재미있을 줄 알았는데 완전 실망했다. 너무 졸리고 돈이 아까웠다.'
test_sentence = test_sentence.split(' ') # ['재미있을', '줄', '알았는데', '완전', '실망했다.', '너무', '졸리고', '돈이', '아까웠다.']
test_sentences = []
now_sentence = []

for word in test_sentence:
    now_sentence.append(word)
    test_sentences.append(now_sentence[:])
    # print(now_sentence)
    # print(test_sentences)

# test_sentences    # 9개

sequences = tokenizer.texts_to_sequences(test_sentences)
test_X_1 = pad_sequences(sequences,padding='post',maxlen=25)
# test_X_1    

prediction = model.predict(test_X_1)

for idx,sentence in enumerate(test_sentences):
    print(sentence)
    print(prediction[idx])
# 0 : 부정, 1 : 긍정


# In[22]:


# 감성 분석 함수 구현
def sentiment_predict(test_sentence):
    test_sentence = test_sentence.split(' ')
    test_sentences = []
    now_sentence = []

    for word in test_sentence:
        now_sentence.append(word)
        test_sentences.append(now_sentence[:])

    sequences = tokenizer.texts_to_sequences(test_sentences) 
    test_X_1 = pad_sequences(sequences,padding='post',maxlen=25) 

    prediction = model.predict(test_X_1)  

    for idx, sentence in enumerate(test_sentences):
        score = prediction[idx]
        
    if(score[0] > score[1]):
        print("{:.2f}% 확률로 부정 리뷰입니다.".format(score[0] * 100))
    else:
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format((1 - score[0]) * 100))          


# In[23]:


sentiment_predict('이 영화 개꿀잼 ~')


# In[24]:


sentiment_predict('넘 재미없어 내내 졸았어요')


# In[25]:


sentiment_predict('돈이 아까워요 ')


# In[26]:


sentiment_predict('이 영화 하품만 나와요~')


# In[27]:


sentiment_predict('두번 봐도 재미있어요')


# In[28]:


sentiment_predict('이 영화 핵노잼 ㅠㅠ')


# In[29]:


sentiment_predict('이 영화 왜 만든거야')


# In[30]:


sentiment_predict('이 영화 꼭 보세요')


# In[31]:


sentiment_predict('안녕하세요')


# In[32]:


sentiment_predict('그저 그래요')


# In[33]:


sentiment_predict('나는 영화를 자주 봅니다')


# In[ ]:




