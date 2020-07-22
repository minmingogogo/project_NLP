# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:07:32 2020

@author: Scarlett
"""


# step1 load data & config=============================================================================
# step2 load model=============================================================================
# step3 train & save model=============================================================================
# step4 evaluate & analyse =============================================================================
import os
import re
import json
import copy
import tqdm
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import tokenizer_from_json
'''
cpu性能参数设置
'''
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # gpu 资源
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 1
os.environ['KMP_BLOCKTIME'] = "1"
os.environ['KMP_SETTINGS'] = "1"
os.environ['KMP_AFFINITY'] = "granularity=fine,verbose,compat,1,0"
os.environ['OMP_NUM_THREADS'] = "8"
tf.compat.v1.Session(config=config)

# step1 load data & config=============================================================================
path = 'nlptasks/task2_ner/anhui_Address_8cate'
if os.path.exists(path):
    print('data exists')
else:
    print('data not exists')
train_path = os.path.join(path,'train_data')
test_path = os.path.join(path,'test_data')
file_path = train_path
patt = 'PCZSVNK' 
patt_path = os.path.join(file_path,patt)       
char_tokenizer_json_dir = os.path.join(patt_path,'char_tokenizer.json')
bio_tokenizer_json_dir = os.path.join(patt_path,'bio_tokenizer.json')
bieo_tokenizer_json_dir = os.path.join(patt_path,'bieo_tokenizer.json')
label_tokenizer_json_dir = os.path.join(patt_path,'label_tokenizer.json')
char_tensor_dir = os.path.join(patt_path,'char_tensor.txt')
bio_tensor_dir = os.path.join(patt_path,'bio_tensor.txt')
bieo_tensor_dir = os.path.join(patt_path,'bieo_tensor.txt')
label_tensor_dir = os.path.join(patt_path,'label_tensor.txt')
   
def data_load():    
    with open(char_tokenizer_json_dir) as f:
        data = json.load(f)
        char_tokenizer = tokenizer_from_json(data)    
    # input_tokenizer.index_word[6]
        
    with open(bio_tokenizer_json_dir) as f:
        data = json.load(f)
        bio_tokenizer = tokenizer_from_json(data)
        
    with open(bieo_tokenizer_json_dir) as f:
        data = json.load(f)
        bieo_tokenizer = tokenizer_from_json(data)    
    # input_tokenizer.index_word[6]
        
    with open(label_tokenizer_json_dir) as f:
        data = json.load(f)
        label_tokenizer = tokenizer_from_json(data)  

    char_tensor = np.loadtxt(char_tensor_dir)
    char_tensor = char_tensor.astype('int32')#  转回整型 
    
    bio_tensor = np.loadtxt(bio_tensor_dir)
    bio_tensor = bio_tensor.astype('int32')#  转回整型  

    bieo_tensor = np.loadtxt(bieo_tensor_dir)
    bieo_tensor = bieo_tensor.astype('int32')#  转回整型 
    
    label_tensor = np.loadtxt(label_tensor_dir)
    label_tensor = label_tensor.astype('int32')#  转回整型  
    tensor_set = (char_tensor,bio_tensor,bieo_tensor,label_tensor) 
    tokenizer_set = (char_tokenizer,bio_tokenizer,bieo_tokenizer,label_tokenizer)
    return tokenizer_set,tensor_set
if __name__ == '__main__':

    tokenizer_set,tensor_set = data_load()
    
    char_tokenizer,bio_tokenizer,bieo_tokenizer,label_tokenizer = tokenizer_set
    char_tensor,bio_tensor,bieo_tensor,label_tensor = tensor_set
                       
    char_tokenizer.index_word[2]
    bio_tokenizer.index_word[2]
    vocab_size_inp = len(char_tokenizer.word_counts)
    vocab_size_tar_bio = len(bio_tokenizer.word_counts)
    vocab_size_tar_bieo = len(bieo_tokenizer.word_counts)
    print('vocab_size:',vocab_size_inp+1,vocab_size_tar_bio+1,vocab_size_tar_bieo+1)
    print('需要验证tokenizer最后一位')
    print(label_tensor.shape)
    print(char_tokenizer.index_word[vocab_size_inp])
    print('查看label : 0位置只占位没对应label值，但是下面自定义模块会用到：\n',label_tokenizer.index_word)
    
# step2  model=============================================================================
from models.common_module.customized_optimizer import CustomizedSchedule
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint

config = {
    'epoch':10,
    'batch_size':64,
    'num_classes':94,        
    'max_len':80,
    'embed_size':256,
    'vocab_size':2835,
    'hidden_dim':512,
    'num_filters':512,
    'kernel_size':5,}

# model = textcnn(config)
    
'''2.1 Model textcnn classify  ''' 

tf.keras.backend.clear_session()
input_layer=L.Input(shape=(None,),name='feature_input')
emb_layer =L.Embedding(input_dim=config['vocab_size'],
                       output_dim=config['embed_size'],
                       name = 'embedding')(input_layer)        
# if is_pretrain:
#     ''' 预训练词向量'''
#     emb_layer=L.Embedding(input_dim=config.vocab_size,
#                          output_dim=config.embed,
#                          weights = [embedding_matrix], 
#                          trainable = True,name = 'pretrain_emb')(input_layer)   
con_layer = L.Conv1D(filters=config['num_filters'],
                     kernel_size=config['kernel_size'])(emb_layer)                        
con_layer = L.BatchNormalization()(con_layer)
con_layer = L.Activation('relu')(con_layer)
con_layer = L.GlobalMaxPool1D()(con_layer)

output_layer = L.Dense(config['hidden_dim'],activation = 'relu',name="feature_output")(con_layer)        
output_layer=L.Dense(config['num_classes'],activation='softmax',name = 'out_classify')(output_layer)


model=K.models.Model(inputs=[input_layer],
                     outputs=[output_layer])

'''2.2 Optimizer  ''' 
learning_rate = CustomizedSchedule(config['embed_size'])
optimizer = K.optimizers.Adam(learning_rate,
                                  beta_1 = 0.9,
                                  beta_2 = 0.98,
                                  epsilon = 1e-9)              

    
    
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
               metrics=['accuracy'])    

model.summary()


earlystop_callback = EarlyStopping(patience=5,min_delta=1e-3,monitor = 'loss')        
callbacks = [earlystop_callback]
# step3 train & save model=============================================================================

x = char_tensor[:10000]
y = label_tensor[:10000]
history=model.fit(x,y,epochs=10,
                    validation_split = 0.2,
                    callbacks = callbacks)
#   evaluate
s_id = 10000
num = 1000
e_id = s_id+num
x_val = char_tensor[s_id:e_id]
y_val =  label_tensor[s_id:e_id]
for i in range(5):
    s_id = s_id + +num + num *i*2
    e_id = s_id+num
    print('s : {} e :{}'.format(s_id,e_id))
    x_val = char_tensor[s_id:e_id]
    y_val =  label_tensor[s_id:e_id]   
        
    model.evaluate(x_val,y_val)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

