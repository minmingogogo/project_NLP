# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 11:46:30 2020

@author: Scarlett


位置编码有多种生成方式：
    正弦余弦
    学习
    相对位置表达
https://zhuanlan.zhihu.com/p/57732839


i:[0,2]	|	                i=0		                |                     i=1	                    |
		sin             	cos	                    sin	                     cos
		embedding		d_model = 4	
 seq1                	0	                  1	                      2	                      3
token  pos
    我	0	sin(0 /  (10000^(0/4))	cos(0 /  (10000^(1/4))	sin(0 /  (10000^(2/4))	cos(0 /  (10000^(3/4))
	很	1	sin(1 /  (10000^(0/4))	cos(1 /  (10000^(1/4))	sin(1 /  (10000^(2/4))	cos(1 /  (10000^(3/4))
	开	2	sin(2 /  (10000^(0/4))	cos(2 /  (10000^(1/4))	sin(2 /  (10000^(2/4))	cos(2 /  (10000^(3/4))
	心	3	sin(3 /  (10000^(0/4))	cos(3 /  (10000^(1/4))	sin(3 /  (10000^(2/4))	cos(3 /  (10000^(3/4))
	pad	4	sin(4 /  (10000^(0/4))	cos(4 /  (10000^(1/4))	sin(4 /  (10000^(2/4))	cos(4 /  (10000^(3/4))
max_len=5
"""
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L

# generates position embedding(生成位置编码)   
# 偶数 PE(pos,2i) = sin(pos/10000^(2i/d_model))
# 奇数 PE(pos,2i+1) = cos(pos/10000^(2i/d_model))  
#%% sin_cos_pos_encoding
def sin_cos_pos_embedding(embed_size,max_len):
    ''' 正弦余弦位置编码
    output.shape(1,M,E)
    注意：输出后用于拼接计算前需要对齐数据格式
    pos_emb = tf.cast(self.position_embedding,dtype = x.dtype)
    '''
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embed_size) for i in range(embed_size)]
        if pos != 0 else np.zeros(embed_size) for pos in range(max_len)])    # ____________________________
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    '''  归一化 '''
    denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    position_enc = position_enc / (denominator + 1e-8)
    position_enc = position_enc[tf.newaxis, :]
    pos_sincos_embedding = tf.convert_to_tensor(position_enc)

    return pos_sincos_embedding

def sin_cos_pos_encoding(config):
    ''' 正弦余弦位置编码
    output.shape(1,M,E)
    '''
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / config['embed_size']) for i in range(config['embed_size'])]
        if pos != 0 else np.zeros(config['embed_size']) for pos in range(config['max_len'])])    # ____________________________
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    '''  归一化 '''
    denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    position_enc = position_enc / (denominator + 1e-8)
    position_enc = position_enc[tf.newaxis, :]
    pos_sincos_embedding = tf.convert_to_tensor(position_enc)
    return pos_sincos_embedding


#%% numeric_pos_embedding



def numeric_pos_embedding(config):
    ''' 输入长度不固定时候用 Numeric_pos_encoding'''
    pos_emb = L.Embedding(config['max_len'],config['embed_size'])#与输入emb 一致
    position_ids = tf.range(0, config['max_len'], dtype=tf.int32)[tf.newaxis, :]
    pos_embedding = pos_emb(position_ids)
    return pos_embedding
    
def shape_list(x):
    ''' Numeric_pos_encoding 工具 '''
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class Numeric_pos_embedding(L.Layer):
    def __init__(self,config):
        super(Numeric_pos_embedding,self).__init__()
        self.config= config
        self.pos_emb = L.Embedding(config['max_len'],# seq embedding 是config['vocab_size']
                                   config['embed_size'])#与输入
    def call(self,seq_embedding):
        '''
        seq_embedding : 经过Embedding 层的输出 (batch_size,max_len,emb_size)
        '''
        batch,seq_len,emb = shape_list(seq_embedding)
        position_ids = tf.range(0, seq_len, dtype=tf.int32)[tf.newaxis, :]
        position_embedding = self.pos_emb(position_ids) 
        return position_embedding



if __name__ == '__main__':
    config = {
        'epoch':10,
        'embed_size':256,
        'vocab_size':3196,
        'max_len':80,
        'mlp_units':64,
        'num_classes':96,
        'hidden_dim':512,
        'activation':'relu',
        'batch_size':16
        }
            
    x = np.random.randint(0,255,config['batch_size']*config['max_len'])
    x = x.reshape((-1,config['max_len']))
    emb = L.Embedding(config['vocab_size'],config['embed_size'],
                      input_shape = ['max_len'])
    
    x_emb = emb(x)
    pos1 = sin_cos_pos_encoding(config)
    pos2 = numeric_pos_embedding(config)

    pos3_emb = Numeric_pos_embedding(config)
    pos3 = pos3_emb(x_emb)
    
    pos1.shape
    # Out[10]: TensorShape([1, 80, 256])
    pos2.shape
    pos3.shape
    tmp = pos1[0]
    for i in range(80):
        # print(i)
        print('{}---| mean :{:.4f},| max:{:.4f},| min{:.4f}|   '.format(i,
                                                      np.mean(tmp[i].numpy()),
                                                      np.max(tmp[i].numpy()),
                                                      np.min(tmp[i].numpy())))    

    tmp = pos2[0]
    print('| pos | mean | max | min |   ')
    print('|  ----  | ----  | ----  | ----  |')
    for i in range(80):
        # print(i)
        print('| {} | {:.4f} | {:.4f}| {:.4f} |   '.format(i,
                                                      np.mean(tmp[i].numpy()),
                                                      np.max(tmp[i].numpy()),
                                                      np.min(tmp[i].numpy())))    

