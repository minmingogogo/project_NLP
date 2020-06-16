# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:47:58 2020

@author: Scarlett
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L

class Create_masks():
    '''
    Parameters
    ----------
    inp : (B,M)
        原语言上的句子.
    tar : (B,M)
        目标语言上的句子.
    Returns
    -------
    Encoder :
        encoder_padding_mask 
    Decoder : 
        decoder_mask:
        encoder_decoder_padding_mask: 
    '''     
    
    def create_masks(self,inp,tar):
        # create_padding_mask : (B,M)=> (B,1,1,M) * [0]
        encoder_padding_mask = self.create_padding_mask(inp)
        encoder_decoder_padding_mask = self.create_padding_mask(inp)
        decoder_padding_mask = self.create_padding_mask(tar)        
        # create_look_ahead_mask : (M,M) ：下三角0 上三角1
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        # decoder_mask:(B,1,M,M) 合并操作;(M,M) 复制(B,1)份 
        decoder_mask = tf.maximum(decoder_padding_mask,look_ahead_mask)        
        return encoder_padding_mask,decoder_mask,encoder_decoder_padding_mask

    def create_padding_mask(self,batch_data):
        ''' 全零数组 '''
        #   batch_data.shape:[batch_size,seq_len]
        #   padding_mask.shape [batch_size,seq_len] 全零
        padding_mask = tf.cast(tf.math.equal(batch_data,0),tf.float32)
        #   为了便于跟attention_weight计算添加两个维度在中间
        #   padding_mask.shape:[batch_size,1,1,seq_len]
        padding_mask = padding_mask[:,tf.newaxis,tf.newaxis,:] 
        return  padding_mask 
    def create_look_ahead_mask(self,size):
        ''' 标记矩阵，下三角为 0上 三角为 1
        input:
            size : max_len 
        output:
            mask （size,size); 为标记矩阵，下三角为 0上 三角为 1
        ------
        tf.linalg.band_part  
            作用：主要功能是以对角线为中心，取它的副对角线部分，其他部分用0填充。    
        参数：
            input:输入的张量.
            num_lower:下三角矩阵保留的副对角线数量，从主对角线开始计算，相当于下三角的带宽。取值为负数时，则全部保留。
            num_upper:上三角矩阵保留的副对角线数量，从主对角线开始计算，相当于上三角的带宽。取值为负数时，则全部保留。
        '''
        mask = 1 - tf.linalg.band_part(tf.ones((size,size)),-1,0)
        return mask 
if __name__ == '__main__':
    config = {
        'epoch':10,
        'embed_size':256,
        'vocab_size':3196,
        'max_len':16,
        'mlp_units':64,
        'num_classes':96,
        'hidden_dim':512,
        'activation':'relu',
        'batch_size':16,
        'units':64,
        'attention_units':64
        }     
    
    inp = np.random.randint(0,1000,2000).reshape((-1,config['max_len']))
    tar = np.random.randint(0,5,2000).reshape((-1,20))
    cmask = Create_masks()
    encoder_padding_mask,decoder_mask,encoder_decoder_padding_mask = cmask.create_masks(inp,tar)
    