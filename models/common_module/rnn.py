# -*- coding: utf-8 -*-


import tensorflow.keras as K
import tensorflow.keras.layers as L
import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional
import tensorflow as tf

class rnn_layer(L.Layer):
    """docstring for rnn_layer"""
    def __init__(self,config):
        super(rnn_layer, self).__init__()
        self.bi_lstm = Bidirectional(LSTM(config['units'], 
                                          return_sequences=True,return_state=False))
        self.lstm = LSTM(config['units'],
                         return_sequences=True,return_state=False)
        self.dropout = Dropout(config['dropout'])
    def bi_rnn(self,feature, training):
        self.training = training
        rnn_layer = self.bi_lstm(feature)
        return self.dropout(rnn_layer, training =self.training)
    def rnn(self,feature,label_input):
        rnn_layer = self.lstm(feature)
        return self.dropout(rnn_layer, training=self.training)
    
    
if __name__ == '__main__': 
    config = {
        'epoch':10,
        'embed_size':256,
        'vocab_size':3196,
        'max_len':16,
        'batch_size':16,
        'units':80,
        'mlp_units':64,
        'num_classes':96,
        'hidden_dim':512,
        'activation':'relu',
        'dropout':0.5
        }        
    
    ''' rnn gru lstm bilstm 输入输出说明
    units 是否要与 embedding_dim 一致？
    
    '''

    embed = L.Embedding(config['vocab_size'],
                            config['embed_size'],
                            input_length =config['max_len'])        
    encoding_units = config['units']
    gru = L.GRU(encoding_units,
                #   需要每一步的输出
                return_sequences = True,
                return_state= True,
                recurrent_initializer = 'glorot_uniform')  
    
    lstm = L.LSTM(encoding_units,
                #   需要每一步的输出
                return_sequences = True,
                return_state= True,
                recurrent_initializer = 'glorot_uniform')  
    biLstm = L.Bidirectional(L.GRU(encoding_units,
                #   需要每一步的输出
                return_sequences = True,
                return_state= True,
                recurrent_initializer = 'glorot_uniform'))  
    
    inp = np.random.randint(0,1000,2000).reshape((-1,config['max_len']))
    # inp.shape: (125, 16)
    x_emb = embed(inp)
    # x_emb.shape: TensorShape([125, 16, 256])
# =============================================================================  
    gru_output,hidden_state = gru(x_emb)    
    # gru_output.shape : TensorShape([125, 16, 80])
    # hidden_state.shape : TensorShape([125, 80])
    gru_output[0][-1] == hidden_state[0]
    ''' 
    gru_output : 是所有时间步的hidden state,因此是 中间多一个时间步维度，时间步为max_len
    hidden_state 为最后一个时间步的状态
    所以在 return_sequences = True, return_state= True,时
    gru_output[0][-1] == hidden_state[0]
    
    '''
# =============================================================================

    lstm_output,hidden_state,cell_state = lstm(x_emb)
    # lstm_output.shape :TensorShape([125, 16, 80])
    # hidden_state.shape : TensorShape([125, 80]) 
    # cell_state.shape : TensorShape([125, 80]) 
    lstm_output[0][-1]==hidden_state[0]
    ''' 
    lstm_output : 是所有时间步的hidden state,因此是 中间多一个时间步维度，时间步为max_len
    hidden_state 为最后一个时间步的状态
    所以在 return_sequences = True, return_state= True,时
    lstm_output[0][-1]==hidden_state[0]
    
    '''    
# =============================================================================
    bilstm_output,hidden_state,cell_state = biLstm(x_emb)
    # bilstm_output.shape :TensorShape([125, 16, 160])
    # hidden_state.shape : TensorShape([125, 80])
    # cell_state.shape : TensorShape([125, 80])
    bilstm_output[0][-1][:80]==hidden_state[0]
    ''' 
    bilstm_output : 是双向所有时间步的hidden state,
                    因此是 中间多一个时间步维度，时间步为max_len，最后units * 2
    hidden_state 为最后一个时间步的状态
    所以在 return_sequences = True, return_state= True,时
    bilstm_output[0][-1][:80]==hidden_state[0]
    
    '''      
    
    
    lstm = L.LSTM(encoding_units,
                #   需要每一步的输出
                return_sequences = True,
                return_state= True,
                recurrent_initializer = 'glorot_uniform')      
    lstm_output = lstm(x_emb)
    t1 = lstm_output[0]
    t2 = lstm_output[1]
    t3 = lstm_output[2]
    
    
    
    