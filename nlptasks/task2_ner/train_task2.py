# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:19:02 2020

@author: Scarlett
"""

# step1 load data & config=============================================================================
# step2 load model=============================================================================
# step3 train & save model=============================================================================
# step4 evaluate & analyse =============================================================================
import os
import re
import json
import tqdm
import time
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from models.common_module.attention import Attention
from models.common_module.encoder import Encoder
from models.common_module.decoder import Decoder
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from sklearn.model_selection import train_test_split
'''
cpu性能参数设置
'''
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 8
config.inter_op_parallelism_threads = 1
os.environ['KMP_BLOCKTIME'] = "1"
os.environ['KMP_SETTINGS'] = "1"
os.environ['KMP_AFFINITY'] = "granularity=fine,verbose,compat,1,0"
os.environ['OMP_NUM_THREADS'] = "8"
tf.compat.v1.Session(config=config)

# step1 load data & config=============================================================================
path = 'nlptasks//task2_ner//anhui_Address_8cate'
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
                 
data_load()               
char_tokenizer.index_word[2]
bio_tokenizer.index_word[2]
vocab_size_inp = len(char_tokenizer.word_counts)
vocab_size_tar = len(bio_tokenizer.word_counts)
print('vocab_size:',vocab_size_inp+1,vocab_size_tar+1)
print('需要验证tokenizer最后一位')
print(char_tokenizer.index_word[vocab_size_inp])

# step2 load model=============================================================================
from models.common_module.Model_BiLSTM_CRF import BiLSTM_CRF
from models.common_module.customized_optimizer import CustomizedSchedule
from models.common_module.CRF_Layer import CRF
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint

#   model & loss_func & optimizer

config = {
    'epoch':10,
    'batch_size':128,
    'embed_size':256,# 需要等于 attention_units 残差连接 
    # embedding_layer
    'vocab_size':2835,
    'max_len':45,
    'num_classes':18,
    #   lstm
    'hidden_units':512
        }

bilstm_crf = BiLSTM_CRF(config)

model = K.Sequential([bilstm_crf])
model.build(input_shape = (config['batch_size'],config['max_len']))
model.summary()
learning_rate = CustomizedSchedule(config['embed_size'],warmup_steps = 100)
optimizer = K.optimizers.Adam(learning_rate,
                                  beta_1 = 0.9,
                                  beta_2 = 0.98,
                                 epsilon = 1e-9) 


model.compile(loss= bilstm_crf.crf.get_loss,
              optimizer = optimizer,
               metrics = ['accuracy']
              )

checkpoint_path = 'nlptasks//task2_ner//checkpoint'
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_filepath = checkpoint_path + '//bilstm_crf'

earlystop_callback = EarlyStopping(patience=5,
                                   min_delta=1e-3,
                                   monitor = 'loss')
model_checkpoint_callback = ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            # monitor='val_acc',
                            # mode='max',
                            save_best_only=True)
callbacks = [earlystop_callback,
             model_checkpoint_callback]



# crf = CRF(3,name='crf_layer')

# loss_func = crf.get_loss
# y_true, y_pred
# y_true = [[0, 0, 1], [0, 1, 0]]
# y_pred = [[0.1, 0.9, 0.8],[0.05, 0.95, 0]]
# y_true = tf.convert_to_tensor(y_true,dtype=tf.int64)
# y_pred = tf.convert_to_tensor(y_pred,dtype=tf.int64)
# loss_func(y_true, y_pred)


# step3 train & save model=============================================================================
#   model & loss_func & optimizer
history = model.fit(char_tensor[:5000],
          bio_tensor[:5000],
          validation_split = 0.2,
          batch_size = config['batch_size'],
          epochs = 50,
          callbacks = callbacks
          )


# step4 evaluate & analyse =============================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt

y_pred = model.predict(char_tensor[16000:27000])
y_true = bio_tensor[16000:27000]
acc = tf.keras.metrics.Accuracy()
acc.reset_states()
_ = acc.update_state(y_true, y_pred)
acc.result().numpy()

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)











