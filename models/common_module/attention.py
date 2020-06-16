# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:40:23 2020

@author: Scarlett

1.相关资料
    global attention 和 local attention
        https://blog.csdn.net/weixin_40871455/article/details/85007560
    论文
        chrome-extension://cdonnmffkdaoajfknoeeecmchibpmkmg/assets/pdf/web/viewer.html?file=https%3A%2F%2Fnlp.stanford.edu%2Fpubs%2Femnlp15_attn.pdf
    tf 源码
        https://github.com/tensorflow/addons/blob/v0.10.0/tensorflow_addons/seq2seq/attention_wrapper.py#L693-L853
    self-attention 步骤图
        https://www.jianshu.com/p/b1030350aadb
    几种attention 效果对比
        https://github.com/uzaymacar/attention-mechanisms#local-attention

2.比较
    Global Attention 和 Local Attention 各有优劣，实际中 Global 的用的更多一点，因为：
    Local Attention 当 encoder 不长时，计算量并没有减少

3.模块构成
    Attention
    selfAttention   
    MultiHeadAttention
    Attention local 模式未完成，需要时间步输入，要训练配合，计算量比global 大很多。



"""
import sklearn
import os,sys
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential


print(tf.__version__)


#%% LSTM + Attention

class Attention(K.Model):
    '''  global  attention 
        维度说明 ： batch_size : B ,max_len: M ,hidden_units : U
        attention_units : A ,attention_output_dim : O 
        input.shape
            all_hidden_state.shape (B,M,U)
            hidden.shape : (B,U) => (B,1,U)

        ---- hidden state 可来源以下Rnncell :  
        all_hidden_state,hidden = gru(inp) 
                ==> all_hidden_state.shape (B,M,U)
        all_hidden_state,hidden,cell_state = lstm(inp)
                ==> all_hidden_state.shape (B,M,U)
        all_hidden_state,hidden,cell_state = bilist(inp)
                ==> all_hidden_state.shape (B,M,2U)       
        --all_hidden_state[:,-1,: ] = hidden                                 
        --可以只输入 all_hidden_state       
           
    '''
    def __init__(self,config,output_dim = 1):
        super(Attention, self).__init__()
        self.config = config
        self.W1 = L.Dense(config['attention_units'])
        self.W2 = L.Dense(config['attention_units'])
        self.V = L.Dense(output_dim)  
        #  BahdanauAttention 使用output_dim = 1
        #  意义是 attemtion [ B M 1] 每个token 对应一个权重值,
        #  为了方便对后面程序理解没有直接写V = L.Dense(1)  
    def dot_score(self,hidden,all_hidden_state):
        '''
        不做线性变换直接点乘
        shape:
            all_hidden_state.shape (B,M,U)
            hidden.shape :  (B,1,U)   
            score.shape (B,M,U)            
        '''
        score = hidden * all_hidden_state
        return score
    def general_score(self,hidden,all_hidden_state):
        '''
        线性变换后点乘
        shape:
            all_hidden_state.shape (B,M,U) =W2>(B,M,A)
            hidden.shape :  (B,1,U)    =W1=>(B,1,A)
            score.shape (B,M,A)       
        '''          
        score = self.W1(hidden) * self.W2(all_hidden_state)
        return score        
    def concat_score(self,hidden,all_hidden_state):
        '''BahdanauAttention 
        线性变换后相加，再线性变换输出
        shape:
            all_hidden_state.shape (B,M,U) =W2>(B,M,A)
            hidden.shape :  (B,1,U)    =W1=>(B,1,A)
            score.shape (B,M,A)     =V=>(B,M,O)     
        '''
        concat = self.W1(hidden) + self.W2(all_hidden_state)
        score = self.V(tf.nn.tanh(concat))
        return score   
    def align(self,hidden,all_hidden_state,mode):
        '''
        三种分数区别：
                是否对输入、输出进行线性变换
                采用点乘还是相加
        input.shape 
            all_hidden_state.shape (B,M,U)
            hidden.shape :  (B,1,U)   
            score.shape (B,M,U)/(B,M,A)/(B,M,O)
        output.shape       
            attention_weights.shape 
            doc/general/Bahdanau : (B,M,U)/(B,M,A)/(B,M,O)
        '''
        if mode =='concat':
            score = self.concat_score(hidden,all_hidden_state)
        elif mode =='dot':
            score = self.dot_score(hidden,all_hidden_state)
        elif mode == 'general':
            score = self.general_score(hidden,all_hidden_state)
        # print('score.shape:{}'.format(score.shape))
        attention_weights = tf.nn.softmax(score,axis = 1) 
        # print('attention_weights.shape:{}'.format(attention_weights.shape))
        return attention_weights
     
    def call(self,all_hidden_state,hidden=None,mode = 'concat'):
        ''' 
        维度说明 ： batch_size : B ,max_len: M ,hidden_units : U
        input.shape
            all_hidden_state.shape (B,M,U)
            hidden.shape : (B,U) => (B,1,U)
        output.shape
            A/O <= U ，  A 和 O 的维度都小于 U
            attention_weights.shape 
            doc/general/Bahdanau : (B,M,U)/(B,M,A)/(B,M,O)        
            context_vector.shape :(B,1,U)
       
        '''
        try : 
            dim_hidden = hidden.shape
        except:
            ''' 没有hidden '''
            hidden = all_hidden_state[:,-1,:][:,tf.newaxis,:]
        else:
            if len(hidden.shape) ==2 and len(all_hidden_state.shape) ==3:
                ''' 维度不一致 hidden增加一个维度 调整到一致'''
                hidden = tf.expand_dims(hidden,1)
        attention_weights = self.align(hidden,all_hidden_state,mode)
        context_vector = attention_weights * all_hidden_state 

        '''
        3D shape
            attention_weights.shape  (B,M,1)
            context_vector.shape   (B,M,U)  =sum(max_len)>(B,1,U) 
        '''        
        #  context_vector M维度上求和，保留 3D 输出
        context_vector = tf.math.reduce_sum(context_vector,axis = 1,keepdims = True)
        # print(' context_vec.shape:{}\n attention_w.shape:{}'.format(context_vector.shape,attention_weights.shape))        
        return context_vector,attention_weights

def debug(return_sequences = True,
          return_state= True):
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
    embed = L.Embedding(config['vocab_size'],
                            config['embed_size'],
                            input_length =config['max_len'])        
    encoding_units = config['units']
    gru = L.GRU(encoding_units,
                #   需要每一步的输出
                return_sequences = return_sequences,
                return_state= return_sequences,
                recurrent_initializer = 'glorot_uniform')  
    
    lstm = L.LSTM(encoding_units,
                #   需要每一步的输出
                return_sequences = return_sequences,
                return_state= return_sequences,
                recurrent_initializer = 'glorot_uniform')  
    bilstm = L.Bidirectional(L.GRU(encoding_units,
                #   需要每一步的输出
                return_sequences = return_sequences,
                return_state= return_sequences,
                recurrent_initializer = 'glorot_uniform'))  
    
    inp = np.random.randint(0,1000,2000).reshape((-1,config['max_len']))
    # inp.shape: (125, 16)
    x_emb = embed(inp)
    # x_emb.shape: TensorShape([125, 16, 256])
    return config,x_emb,gru,lstm,bilstm    

if __name__ == '__main__':
    
    #   获取测试变量
    config,x_emb,gru,lstm,bilstm = debug(return_sequences = True,return_state= True)
    all_hidden_state,hidden_state,cell_state = lstm(x_emb)
    print(all_hidden_state.shape,hidden.shape,score.shape)
    #   测试 Attention
    atten = Attention(config)
    context_vector,attention_weights = atten(all_hidden_state)
    print(context_vector.shape,attention_weights.shape)
    context_vector,attention_weights = atten(all_hidden_state,mode = 'dot')
    print(context_vector.shape,attention_weights.shape)





#%% Embedding + Self_Attention

from keras import backend as Kb

class Self_Attention(L.Layer):
    ''' 不带mask 的self attention
    维度说明 ： batch_size : B ,max_len: M ,emb_size : E , attention_units :U
    input.shape
        output_dim : U
        x.shape (B,M,E)

    output_dim相当于 units / filters,此处相当于 attention_units
    如果 output_dim = emb_size / U=E 则输出与输入相同形状，
    如果 output_dim<>emb_size / U<>E 则自定义输出形状，
    
    '''
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        ''' 初始化与输入相关的可训练权重 
        
        kernel:[q:[E,U],
                k:[E,U],
                v:[E,U],]
        '''
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        
        super(Self_Attention, self).build(input_shape)  # 一定要在最后这样写保证调用
    def call(self, x,dk_scale = 1):
        ''' 
        线性变换
            x.shape (B,M,E)
            q\k\v.shape = kernel[i].shape (1,E,U)
            dot(x,kernel[i]).shape (B,M,U)
        计算
            Q*K_T :(B,M,U)*(B,U,M) = (B,M,M)
            sim(QK) = softmax(Q*K_T / E**0.5 )  shape : (B,M,M)
            output = batch_dot(sim(QK)*V) = (B,M,M)*(B,M,U) = (B,M,U)
            M<=U
        '''
        WQ = Kb.dot(x, self.kernel[0])
        WK = Kb.dot(x, self.kernel[1])
        WV = Kb.dot(x, self.kernel[2])

        # print("WQ.shape",WQ.shape)
        # print("Kb.permute_dimensions(WK, [0, 2, 1]).shape",Kb.permute_dimensions(WK, [0, 2, 1]).shape)
        QK = Kb.batch_dot(WQ,Kb.permute_dimensions(WK, [0, 2, 1]))
        dk = WK.shape[2] * dk_scale  # 可以考虑对dk缩放，有看到这样的用法可以尝试
        QK = QK / (dk**0.5)
        # print("QK1.shape",QK.shape)
        QK = Kb.softmax(QK)

        # print("QK.shape",QK.shape)

        V = Kb.batch_dot(QK,WV)

        return V

    def compute_output_shape(self, input_shape):

        return (input_shape[0],input_shape[1],self.output_dim)
    
if __name__ == '__main__':
    
    atten = Self_Attention(config['attention_units'])
    context_vector = atten(all_hidden_state)
    print(context_vector.shape)


#%% MultiHeadAttention

def scaled_dot_product_attention(q,k,v,mask):
    '''
        缩放点积注意力
    Parameters
    ----------
    q\k\v : .shape (B,H,M,D) 
    mask : .shape == [...,seq_len_q,seq_len_k]
    compute:  attention_weights = softmax(((q*k_T)/(dk**0.5) + mask))
              output = attention_weights*V
    Returns
    -------
    output:  (B,H,M,D) 与v 一致
    attention_weights:(为了可视化需要返回)weights of attention

    '''
    matmul_qk = tf.matmul(q,k,transpose_b = True) #(B,H,M,D) *(B,H,D,M)
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk) #(B,H,M,M) 
    
    if mask is not None:
        # mask 中凡是要弃除的都是1，保留的都是0,使在softmax后的值趋近于0
        scaled_attention_logits += (mask * -1e9)        
    #   attention_weights : (B,H,M,M) 
    #   output : (B,H,M,D)
    attention_weights = tf.nn.softmax(
        scaled_attention_logits,axis =-1) #(B,H,M,M) 
    output = tf.matmul(attention_weights,v) #(B,H,M,M) * (B,H,M,D)
    return output,attention_weights

def print_scaled_dot_product_attention(q,k,v):
    temp_out,temp_att = scaled_dot_product_attention(q, k, v, None)
    print("Attention weights are:\n{}".format(temp_att))
    print("Output is :\n{}".format(temp_out))
    
    
class MultiHeadAttention(L.Layer):
    ''' 多头注意力
        理论部分和实现有不一样
        理论部分：           实际中
        x -> Wq0 ->q0       q -> Wq0 ->q0
        x -> Wk0 ->k0       k -> Wk0 ->k0
        x -> Wv0 ->v0       k -> Wk0 ->k0  
        实践时,初始值不一定是一样的，因此输入x改成q\k\v
    '''
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention,self).__init__()
        ''' 
        维度说明 ： batch_size : B ,max_len: M ,attention_units : U , num_heads : H                  
                  depth : D  =>  U/H = D 
        input:
            d_model : attention_units(隐藏层维度)
            d_model 必须是num_heads的倍数保证可均匀分割
       -------     
           d_model = 6,num_heads = 3
            |  h1  |   h2  |   h3  | 
            |v1 v2 | v3 v4 | v5 v6 |d_model
                    depth = 2
        '''
        self.num_heads = num_heads
        self.d_model = d_model
        ''' d_model 要均匀分成 num_heads,要整除 '''
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads #   depth 每份多少个向量
    
        #   定义三个全连接层做q,k,v 的转换Wq,Wk,Wv
        self.WQ = L.Dense(self.d_model)        
        self.WK = L.Dense(self.d_model)           
        self.WV = L.Dense(self.d_model)   

        #   拼接再经过全连接层(用于调整输出维度)
        self.dense = L.Dense(self.d_model)
        
    def split_heads(self,x):
        # x.shape (B,M,U)=>(B,M,H,U/H)=>(B,M,H,D)
        # 重排:(B,M,H,D) =[0,2,1,3] =>(B,H,M,D) 
        batch_size = x.shape[0]
        x = tf.reshape(x,(batch_size,-1,self.num_heads,self.depth))
        return tf.transpose(x,perm = [0,2,1,3])

    def call(self,q,k,v,mask):
        #   线性变换 
        q = self.WQ(q)      #q.shape:(B,M,U)
        k = self.WK(k)      #k.shape:(B,M,U)
        v = self.WV(v)      #v.shape:(B,M,U)
        
        #   按多头split 一个维度 , q\k\v.shape:(B,H,M,D)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        
        #   scaled_attention_outputs : (B,H,M,D)
        #   attention_weights : (B,H,M,M) 
        scaled_attention_outputs ,attention_weights = scaled_dot_product_attention(q, k, v, mask)

        #   多头信息存在第二维和第四维上(B,H,M,D) 
        #   重排调整回原来顺序( B,H,M,D) =[0,2,1,3]=> (B,M,H,D) 
        scaled_attention_outputs = tf.transpose(
            scaled_attention_outputs,perm = [0,2,1,3])
        #   合并多头信息 : (B,M,H*D) = (B,M,U)
        batch_size = scaled_attention_outputs.shape[0]
        concat_attention = tf.reshape(scaled_attention_outputs,
                                      (batch_size,-1,self.d_model))
        #   调整输出维度output_dim : O => (B,M,O)，维度一致则不变(B,M,U)
        output = self.dense(concat_attention)
        
        return output,attention_weights    
    

if __name__ == '__main__':

    from common_module.create_masks import Create_masks
    config = {
        'epoch':10,
        'embed_size':256,
        'vocab_size':3196,
        'max_len':16,
        'tar_embed_size':256,
        'tar_vocab_size':1000,   
        'tar_max_len':16,
        # 'mlp_units':64,
        # 'num_classes':96,
        # 'hidden_dim':512,
        # 'activation':'relu',
        'batch_size':16,
        'units':64,
        'attention_units':64,
        'num_heads':8,
        }     
    
    inp = np.random.randint(0,1000,2000).reshape((-1,config['max_len']))
    tar = np.random.randint(0,200,2000).reshape((-1,config['tar_max_len']))
    emb_inp = L.Embedding(config['vocab_size'],config['embed_size'])
    emb_tar = L.Embedding(config['tar_vocab_size'],config['tar_embed_size'])
    inp_emb = emb_inp(inp)
    tar_emb = emb_tar(tar)
    
    
    cmask = Create_masks()
    encoder_padding_mask,decoder_mask,encoder_decoder_padding_mask = cmask.create_masks(inp,tar)
    

    mha = MultiHeadAttention(config['attention_units'],config['num_heads'])
    #   无mask
    output,attention_weights  = mha(inp_emb,inp_emb,inp_emb,None)
    print(output.shape,attention_weights.shape)
    #(125, 16, 64) (125, 8, 16, 16)
    #   有 padding mask 
    output,attention_weights  = mha(inp_emb,inp_emb,inp_emb,encoder_padding_mask)
    print(output.shape,attention_weights.shape)
    #(125, 16, 64) (125, 8, 16, 16)
    #   有 lookahead mask 
    output,attention_weights  = mha(inp_emb,tar_emb,tar_emb,decoder_mask)
    print(output.shape,attention_weights.shape)
    #(125, 16, 64) (125, 8, 16, 16)
        
        
    
    
    
#%%带惩罚项
class SelfAttention(L.Layer):
    """
    Layer for implementing self-attention mechanism. Weight variables were preferred over Dense()
    layers in implementation because they allow easier identification of shapes. Softmax activation
    ensures that all weights sum up to 1.

    @param (int) size: a.k.a attention length, number of hidden units to decode the attention before
           the softmax activation and becoming annotation weights
    @param (int) num_hops: number of hops of attention, or number of distinct components to be
           extracted from each sentence.
    @param (bool) use_penalization: set True to use penalization, otherwise set False
    @param (int) penalty_coefficient: the weight of the extra loss
    @param (str) model_api: specify to use TF's Sequential OR Functional API, note that attention
           weights are not outputted with the former as it only accepts single-output layers
    """
    def __init__(self, size, num_hops=8, use_penalization=True,
                 penalty_coefficient=0.1, model_api='functional', **kwargs):
        if model_api not in ['sequential', 'functional']:
            raise ValueError("Argument for param @model_api is not recognized")
        self.size = size
        self.num_hops = num_hops
        self.use_penalization = use_penalization
        self.penalty_coefficient = penalty_coefficient
        self.model_api = model_api
        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        base_config['size'] = self.size
        base_config['num_hops'] = self.num_hops
        base_config['use_penalization'] = self.use_penalization
        base_config['penalty_coefficient'] = self.penalty_coefficient
        base_config['model_api'] = self.model_api
        return base_config

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.size, input_shape[2]),                                # (size, H)
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.num_hops, self.size),                                 # (num_hops, size)
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):  # (B, S, H)
        # Expand weights to include batch size through implicit broadcasting
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)                                     # (B, H, S)
        attention_score = tf.matmul(W1, hidden_states_transposed)                                   # (B, size, S)
        attention_score = Activation('tanh')(attention_score)                                       # (B, size, S)
        attention_weights = tf.matmul(W2, attention_score)                                          # (B, num_hops, S)
        attention_weights = Activation('softmax')(attention_weights)                                # (B, num_hops, S)
        embedding_matrix = tf.matmul(attention_weights, inputs)                                     # (B, num_hops, H)
        embedding_matrix_flattened = Flatten()(embedding_matrix)                                    # (B, num_hops*H)

        if self.use_penalization:
            attention_weights_transposed = Permute(dims=(2, 1))(attention_weights)                  # (B, S, num_hops)
            product = tf.matmul(attention_weights, attention_weights_transposed)                    # (B, num_hops, num_hops)
            identity = tf.eye(self.num_hops, batch_shape=(inputs.shape[0],))                        # (B, num_hops, num_hops)
            frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product - identity)))  # distance
            self.add_loss(self.penalty_coefficient * frobenius_norm)  # loss

        if self.model_api == 'functional':
            return embedding_matrix_flattened, attention_weights
        elif self.model_api == 'sequential':
            return embedding_matrix_flattened



#%% 未完成    local
    
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation, Permute
from tensorflow.keras.layers import Multiply, Lambda, Reshape, Dot, Concatenate, RepeatVector, \
    TimeDistributed, Permute, Bidirectional
class Attention_to_be_continue(K.Model):
    '''        
    @param (str) context:  ['many-to-many', 'many-to-one']
            任务场景有 "many to one " 和 "many to many "两种
            "many to many "一般用于 encoder - decoder 架构    
             "many to one " 一般用于分类
    @param (str) alignment_type:['global', 'local-m', 'local-p', 'local-p*']
            主要分 golbal 和local模式 
            golbal:全局注意力
            local-m : monotonic alignment 单调对齐
            local-p : 预测位置局部高斯分布 
            local-p* : 自适应学习方法（暂时只适用多对一场景）        
  
    @param (int) window_width:  'local' 模式下注意力单元的宽度
    @param (str) score_function: align 的计算方式
           'dot', 'general', and 'location' both by Luong et al. (2015), 
           'concat' by Bahdanau et al. (2015), 
           'scaled_dot' by Vaswani et al. (2017)
    @param (str) model_api: ['sequential', 'functional']
           用在sequential模式构建的模型时，只能接收一个输出 context_vector
           函数api 可以接受多个输出，因此返回 context_vector, attention_weights
        

        建议:如果模型不能收敛，或者测试精度低于预期，试着调整循环层的隐藏单元大小，
        训练过程中的批处理大小，或者使用“局部”注意力时的参数@window_width
    '''    
    
    def __init__(self,config,
                 context='many-to-many',
                 alignment_type='global', 
                 window_width=None,
                 score_function='general',
                 model_api='functional', **kwargs):
        super(Attention,self).__init()

        if context not in ['many-to-many', 'many-to-one']:
            raise ValueError("Argument for param @context is not recognized")
        if alignment_type not in ['global', 'local-m', 'local-p', 'local-p*']:
            raise ValueError("Argument for param @alignment_type is not recognized")
        if alignment_type == 'global' and window_width is not None:
            raise ValueError("Can't use windowed approach with global attention")
        if context == 'many-to-many' and alignment_type == 'local-p*':
            raise ValueError("Can't use local-p* approach in many-to-many scenarios")
        if score_function not in ['dot', 'general', 'location', 'concat', 'scaled_dot']:
            raise ValueError("Argument for param @score_function is not recognized")
        if model_api not in ['sequential', 'functional']:
            raise ValueError("Argument for param @model_api is not recognized")
        super(Attention, self).__init__(**kwargs)
        self.context = context
        self.alignment_type = alignment_type
        self.window_width = window_width  # D
        self.score_function = score_function
        self.model_api = model_api
    def get_config(self):
        ''' 参数层 可以不用 '''
        base_config = super(Attention, self).get_config()
        base_config['alignment_type'] = self.alignment_type
        base_config['window_width'] = self.window_width
        base_config['score_function'] = self.score_function
        base_config['model_api'] = self.model_api
        return base_config        








