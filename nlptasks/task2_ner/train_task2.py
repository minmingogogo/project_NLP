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
    print(char_tokenizer.index_word[vocab_size_inp])

# step2 load model=============================================================================
from models.common_module.Model_BiLSTM_CRF import BiLSTM_CRF
from models.common_module.customized_optimizer import CustomizedSchedule
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping,ModelCheckpoint

#   model & loss_func & optimizer

config = {
    'epoch':10,
    'batch_size':32,
    'embed_size':256,# 需要等于 attention_units 残差连接 
    # embedding_layer
    'vocab_size':2835,
    'max_len':45,
    'num_classes':19,
    #   lstm
    'hidden_units':256
        } 

bertconfig = {
    'epoch':10,
    'batch_size':32,
    'embed_size':256,# 需要等于 attention_units 残差连接 
    # embedding_layer
    'vocab_size':2835,
    'max_len':45,
    # 'num_classes':18,
    #   bert
    'tgt_size':19,
    'num_layers':4,
    'head':8,
    'ffw_rate':4,
    'attention_dropout':0.2,
    'layer_norm_epsilon':1e-5,    
    #   lstm
    # 'hidden_units':512,
    'rnn':False,
    'rnn_unit':256,
    'rnn_dropout':0.2,    
        } 



def create_model(config):
    tf.keras.backend.clear_session()
    from models.common_module.Model_BiLSTM_CRF import BiLSTM_CRF
   
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
                   metrics = [tf.keras.metrics.Accuracy()]
                  )
    return model
def create_bertmodel(config):
    from models.common_module.Model_Bert_CRF import Ner
    ner = Ner(config)
    model = K.Sequential([ner])
    model.build(input_shape = (config['batch_size'],config['max_len']))
    model.summary()
    learning_rate = CustomizedSchedule(config['embed_size'],warmup_steps = 100)
    optimizer = K.optimizers.Adam(learning_rate,
                                      beta_1 = 0.9,
                                      beta_2 = 0.98,
                                     epsilon = 1e-9) 
    model.compile(loss= ner.crf.get_loss,
                  optimizer = optimizer,
                   metrics = [tf.keras.metrics.Accuracy()]
                  )
    return model

def create_callbacks(checkpoint_filepath):
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
    return callbacks

checkpoint_path = 'nlptasks/task2_ner/checkpoint'
# checkpoint_model_path = os.path.join(checkpoint_path,model_name)
# if not os.path.exists(checkpoint_model_path):
#     os.makedirs(checkpoint_model_path)
# checkpoint_filepath = checkpoint_model_path + '/bieo.ckpt'


# step3 train & save model=============================================================================
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()


def train(model,sub_config,x,y,
          epochs = 10, 
          history_dir = None,
          checkpoint_filepath = None,
          load_weight = False):
    callbacks = create_callbacks(checkpoint_filepath)
    if load_weight:
        history = model.fit(x,
                            y,
                            validation_split = 0.2,
                            # batch_size = config['batch_size'],
                            epochs = 0,
                            # callbacks = callbacks
                            )        
        model.load_weights(checkpoint_filepath)            
    
    history = model.fit(x,
                        y,
                        validation_split = 0.2,
                        batch_size = sub_config['batch_size'],
                        epochs = epochs,
                        callbacks = callbacks
                        )
    history_df = pd.DataFrame(history.history)
    
    if history_dir:
        history_df.to_excel(history_dir)
    # plot_learning_curves(history)
    return history_df

if __name__ == '__main__':

    
    model_name = 'bert_crf'
    checkpoint_path = 'nlptasks/task2_ner/checkpoint'
    checkpoint_model_path = os.path.join(checkpoint_path,model_name)
    if not os.path.exists(checkpoint_model_path):
        os.makedirs(checkpoint_model_path)
    analyse_dir = os.path.join(path,'analyse/Model_bert')
    
    
    # 训练单个模型=============================================================================
    # 
    checkpoint_filepath = checkpoint_model_path + '/bio.ckpt'
    history_dir = os.path.join(analyse_dir +'/bio_history.xlsx')  
    
    x = char_tensor[:5000]
    y =  bio_tensor[:5000]
    
    tf.keras.backend.clear_session()
    model = create_bertmodel(bertconfig)
    history_df = train(model,bertconfig,x,y,
                       epochs = 20 ,
                       history_dir = history_dir,
                       checkpoint_filepath = checkpoint_filepath)
    
    
    # 训练多个model =============================================================================
    model_name = 'bert_crf'
    checkpoint_path = 'nlptasks/task2_ner/checkpoint'
    analyse_dir = os.path.join(path,'analyse','Model_'+ model_name)
    if not os.path.exists(analyse_dir):
        os.makedirs(analyse_dir)  
    for type_name in ['bio','bieo']:
        print(type_name)
        sub_config = copy.deepcopy(bertconfig)
        checkpoint_model_path = os.path.join(checkpoint_path,model_name,type_name)
        if not os.path.exists(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)    
        
        checkpoint_filepath = checkpoint_model_path + '/'+ type_name+'.ckpt'
        history_dir = os.path.join(analyse_dir +'/'+ type_name + '_history.xlsx')     
        if type_name =='bio':
            x = char_tensor[:5000]
            y =  bio_tensor[:5000] 
        elif type_name =='bieo':
            sub_config['tgt_size'] = 27  
            x = char_tensor[:5000]
            y =  bieo_tensor[:5000]        
        print('tgt_size : ',sub_config['tgt_size'])
        tf.keras.backend.clear_session()
        model = create_bertmodel(sub_config)   
        history_df = train(model,sub_config,x,y,
                           epochs = 30 ,
                           history_dir = history_dir,
                           checkpoint_filepath = checkpoint_filepath)    
        
    
    
    
    
    model_name = 'bilstm_crf'
    
    checkpoint_path = 'nlptasks/task2_ner/checkpoint'
    # checkpoint_model_path = os.path.join(checkpoint_path,model_name)
    # if not os.path.exists(checkpoint_model_path):
    #     os.makedirs(checkpoint_model_path)
    analyse_dir = os.path.join(path,'analyse','Model_'+ model_name)
    
    for type_name in ['bio','bieo']:
        # type_name ='bieo'
        print(type_name)
        sub_config = copy.deepcopy(config)
        checkpoint_model_path = os.path.join(checkpoint_path,model_name,type_name)
        if not os.path.exists(checkpoint_model_path):
            os.makedirs(checkpoint_model_path)    
    
        checkpoint_filepath = checkpoint_model_path + '/'+ type_name+'.ckpt'
        history_dir = os.path.join(analyse_dir +'/'+ type_name + '_history.xlsx')      
        if type_name =='bio':
            x = char_tensor[:5000]
            y =  bio_tensor[:5000] 
        elif type_name =='bieo':
            sub_config['num_classes'] = 27  
            x = char_tensor[:5000]
            y =  bieo_tensor[:5000]        
        print('num_classes : ',sub_config['num_classes'])
        tf.keras.backend.clear_session()
        model = create_model(sub_config)  

        history_df = train(model,sub_config,x,y,
                           epochs = 30 ,
                           history_dir = history_dir,
                           checkpoint_filepath = checkpoint_filepath)    

# step4 evaluate & analyse =============================================================================

# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(NpEncoder, self).default(obj)
# # 原文链接：https://blog.csdn.net/Zhou_yongzhe/article/details/87692052


# def get_random_train_eval_data(x,y,starnum = 3,train_size = 5000,eval_size = 1000):
#     star = starnum *1000
#     end = star+train_size
#     eval_star = end
#     eval_end = end+eval_size    
#     x_train = x[star:end]
#     y_train = y[star:end]
#     x_eval = x[eval_star:eval_end]
#     y_eval = y[eval_star:eval_end]
#     return x_train,y_train,x_eval,y_eval



