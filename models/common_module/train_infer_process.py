# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:58:17 2020

@author: Scarlett
"""

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



def get_batch(inp_array,tar_array,batch_size):
    '''
    按 batch 分割
    '''
    batchs = math.ceil(len(inp_array)/batch_size)-1
    num = int(batchs*batch_size)
    inp_array = inp_array[:num]
    tar_array = tar_array[:num]
    dataset = tf.data.Dataset.from_tensor_slices((inp_array, tar_array))
    dataset = dataset.batch(batch_size)
    print('batchs : {}'.format(batchs))
    #   校验是否每个dataset batch 都与config 一致
    tag = 0
    ind = 0
    for i in dataset:
        ind+=1
        inp,tar = i
        if inp.shape[0]<config['batch_size']:
            tag+=1
            print('batch_size not equal in ind: ',ind)    
    print('not equal batch_size number : ',tag)
    return dataset,batchs

train_set,batchs = get_batch(input_train,target_train,config['batch_size'])

model = model  
ckpt, ckpt_manager = ckpt_set
optimizer = optimizer
loss_func = 
accuracy_func = 
metrics = {'Loss','Accuracy'}
earlystop_num = 10



def train(train_set,train_ops):
    '''
    train_ops: 
        model,ckpt_set,optimizer,metrics
    
    '''
    train_loss,train_batch_loss = train_ops
    @tf.function(experimental_relax_shapes=True)
    def train_step(input_ids,input_labels):
        with tf.GradientTape() as tape:
            output = model(input_ids)
            # logits,loss = model(batch_data,batch_label,training = training)  
            loss = 
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)#   loss 得到固定值后，多次调用train_loss(loss) ，会一直变；下面打印train_loss.result()平均损失而执行的实例操作
        train_batch_loss(batch_loss)#   loss 得到固定值后，多次调用train_loss(loss) ，会一直变；下面打印train_loss.result()平均损失而执行的实例操作
        #   获取所有变量
        # variables = encoder.trainable_variables + decoder.trainable_variables
        variables = model.trainable_variables 
        gradients = tape.gradient(loss,variables)
        optimizer.apply_gradients(zip(gradients,variables))



    train_loss.reset_states()
    train_batch_loss.reset_states()
    patience = 0
    last_loss = None  
    delta  =0.001    
    for index,batch in enumerate(train_set):
        # if index>1:
        #     break
        inp_ids,tar_ids = batch
        # batch_label = tf.cast(batch_label,dtype = batch_data.dtype)
        train_step(input_ids,input_labels)
        #-------early stop----------------
        if not last_loss:
            last_loss = train_loss.result()
        else:
            if abs(last_loss - train_loss.result())>delta:
                patience = 0
                last_loss = train_loss.result()
            else:
                patience +=1
            if patience >=5:
                # print('earlystop Epoch {} Loss : {:.4f}  Accuracy :{:.4f}'.format(epoch,train_loss.result(), train_batch_loss.result()))
                break
        #-------early stop----------------        
        # print('Epoch {} Loss : {:.4f}  batch_loss :{:.4f}'.format(epoch,train_loss.result(), train_batch_loss.result()))
        if index % 100 == 0 and index > 0:
            save_path = ckpt_manager.save()
            # print("Saved checkpoint {}".format(save_path))
    save_path = ckpt_manager.save()
    print("Saved checkpoint {}".format(save_path))            
    return train_loss.result(), train_batch_loss.result()


#   定义单步损失函数的计算方法
def loss_function(real,pred):
    mask = tf.math.logical_not(tf.math.equal(real,0))
    loss_ = loss_object(real,pred)
    #   mask 类型变换
    mask = tf.cast(mask,dtype = loss_.dtype)
    #   乘法以后 padding 部分都不算在loss 中
    loss_ *= mask
    #   乘完mask 后再聚合tf.reduce_mean,这也是loss_object 中reduction ="None"的原因
    return tf.reduce_mean(loss_)
loss_object = K.losses.SparseCategoricalCrossentropy(
    from_logits = True,
    reduction = 'none' )

#   定义多步损失函数做梯度下降
@tf.function
def train_step(inp,targ,encoding_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        encoding_outputs,encoding_hidden = encoder(inp,encoding_hidden)
        
        decoding_hidden = encoding_hidden
        for t in range(0,targ.shape[1]-1):
            decoding_input = tf.expand_dims(targ[:,t],1)
            print('train_step decoding_input.shape : {}'.format(decoding_input.shape))
            #   attention_weight 没用到 用_ 替代
            predictions,decoding_hidden,_ = decoder(
                decoding_input,decoding_hidden,encoding_outputs)
            loss += loss_function(targ[:,t+1], predictions)
    
    #   做成batch loss;为了比较不同batch 下的loss 效果，要平均一下
    batch_loss = loss/int(targ.shape[0])
    #   获取所有变量
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss,variables)
    optimizer.apply_gradients(zip(gradients,variables))
    return batch_loss
    



def infer(config,input_eval,target_eval,input_tokenizer,output_tokenizer,analyse_dir):
    ''' 这部分只适合单条测试  '''
    from models.common_module.Model_seq2seq_attention import Seq2Seq_Attention,loss_function
    from analyse_func.bleu import bleu_v2_output
    # restore model from ckpt
    model = Seq2Seq_Attention(config)
    #从训练的检查点恢复权重
    ckpt = tf.train.Checkpoint(model=model)
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir + model_name)
    #添加expect_partial()关闭优化器相关节点warnning打印
    status = ckpt.restore(latest_ckpt).expect_partial()
    