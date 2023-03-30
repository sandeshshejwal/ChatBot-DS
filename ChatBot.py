#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np


# In[3]:


with open('train_qa.txt' , "rb") as fp:
    train_data = pickle.load(fp)
    


# In[7]:


len(train_data)


# In[5]:


with open('test_qa.txt' , "rb") as fp:
    test_data = pickle.load(fp)


# In[8]:


len(test_data)


# In[10]:


train_data[0][2]


# In[11]:


vocab = set()


# In[184]:


all_data = test_data + train_data


# In[185]:


for story , question , answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))


# In[186]:


vocab.add('yes')
vocab.add('no')


# In[206]:


len(vocab)


# In[207]:


vocab_len = len(vocab) + 1


# In[208]:


max_story_len = max([len(data[0]) for data in all_data])
max_story_len


# In[209]:


max_ques_len = max([len(data[1]) for data in all_data])
max_ques_len


# In[210]:


from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


# In[211]:


tokenizer = Tokenizer(filters = [])


# In[212]:


tokenizer.fit_on_texts(vocab)


# In[213]:


tokenizer.word_index


# In[214]:


# train dataset
train_story_text = []
train_question_text = []
train_answers = []

for story , question , answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)


# In[215]:


train_story_seq = tokenizer.texts_to_sequences(train_story_text)


# In[216]:


train_story_seq


# In[217]:


def vectorize_stories(data , word_index = tokenizer.word_index, max_story_len = max_story_len , max_ques_len = max_ques_len ):
    X = []
    Xq =[]
    Y =[]
    
    for story , query , answer in data :
        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        y = np.zeros(len(word_index) + 1)
        y[word_index[answer]] = 1
        
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    
    return(pad_sequences(X,maxlen=max_story_len),
           pad_sequences(Xq,maxlen=max_ques_len),
           np.array(Y) )


# In[218]:


inputs_train , queries_train, answers_train = vectorize_stories(train_data)


# In[219]:


inputs_test , queries_test , answers_test = vectorize_stories(test_data)


# In[220]:


inputs_train


# In[221]:


queries_test


# In[222]:


answers_test


# In[223]:


tokenizer.word_index['yes']


# In[224]:


tokenizer.word_index['no']


# In[225]:


from keras.models import Sequential, Model
from keras.layers import Embedding
from keras.layers import Input , Activation , Dense , Permute , Dropout , add, dot , concatenate , LSTM


# In[226]:


input_sequence = Input((max_story_len,))
question = Input((max_ques_len,))


# In[227]:


#input encoder m
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim = vocab_len , output_dim = 64))
input_encoder_m.add(Dropout(0.3))


# In[228]:


#input encoder C
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim = vocab_len , output_dim = max_ques_len))
input_encoder_c.add(Dropout(0.3))


# In[229]:


question_encoder = Sequential()
question_encoder.add(Embedding(input_dim = vocab_len , output_dim = 64 , input_length = max_ques_len))
question_encoder.add(Dropout(0.3))


# In[230]:


#Encode the sequences
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)


# In[231]:


match = dot([input_encoded_m , question_encoded ] , axes = (2,2))
match = Activation('softmax')(match)


# In[232]:


response = add([match,input_encoded_c])
response = Permute((2,1))(response)


# In[233]:


answer = concatenate([response,question_encoded])


# In[234]:


answer = LSTM(32)(answer)


# In[235]:


answer = Dropout(0.5)(answer)
answer = Dense(vocab_len)(answer)


# In[236]:


answer = Activation('softmax')(answer)


# In[240]:


model = Model([input_sequence , question] , answer)
model.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])


# In[241]:


model.summary()


# In[242]:


history = model.fit([inputs_train , queries_train] , answers_train,
                   batch_size = 32, epochs = 20,
                   validation_data = ([inputs_test , queries_test],answers_test)
                   )


# In[247]:


import matplotlib.pyplot as plt
print(history.history.keys())
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")





# In[250]:


#save
model.save("chatbot_modelDS")


# In[251]:


#evaluate the test set
model.load_weights("chatbot_modelDS")


# In[253]:


pred_results = model.predict(([inputs_test , queries_test]))


# In[254]:


test_data[0][0]


# In[260]:


story = ' '.join(word for word in test_data[13][0])


# In[261]:


story


# In[262]:


query = ' '.join(word for word in test_data[13][1])


# In[263]:


query


# In[264]:


test_data[13][2]


# In[266]:


val_max = np.argmax(pred_results[13])

for key , val in tokenizer.word_index.items():
    if val == val_max:
        k = key
        
print("Predicted Answer is" , k)
print("Probability of Certanity" , pred_results[13][val_max])


# In[ ]:




