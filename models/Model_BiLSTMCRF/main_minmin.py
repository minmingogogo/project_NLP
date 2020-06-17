# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:17:17 2020

@author: Scarlett
"""
import pandas as pd
import numpy as np
import collections
import math
import os
import random
import tqdm
import tensorflow as tf

class Get_corpus():
    def __init__(self):
        self.corpus_path = 'D:\\myGit\\Algorithm\\A_my_nlp_project\\corpus\\Corpus_Ner\\chinese_seqlist_labelslist'
        self.x_train_dir = os.path.join(self.corpus_path,'source.txt')
        self.y_train_dir = os.path.join(self.corpus_path,'target.txt')

    def create_text(self,x_train,y_train,seq_add_start = False,seq_add_end = False):
        seq_list = []
        label_list = []
        # text_words = []
        # text_labels = []
        error = []
        print('检验一致性')    
        if len(x_train) == len(y_train):
            for i in tqdm.tqdm(range(len(x_train))):
                a = x_train.loc[i][0]
                b = y_train.loc[i][0]
                a = a.split(' ')
                b = b.split(' ')
                miss_num  = np.abs(len(a)-len(b))
                if miss_num == 1:
                    if len(a)>len(b):
                        # print('文本多一位')
                        if a[-1]!='':                
                            print('a最后一位',a[-1])
                            error.append(i)
                            continue
                    else:
                        error.append(i)
                        continue
                elif miss_num > 1:
                    error.append(i)
                    continue 
                  # 是否添加开头结尾标记
                if seq_add_start:
                    a = ['<START>'] + a
                    b = ['START'] + b
                if seq_add_end:
                    a =  a + ['<END>']
                    b =  b + ['END']
                seq_list.append(a)
                label_list.append(b)
                # text_words.extend(a)
                # text_labels.extend(b)
        print('error num : {}'.format(len(error)))
        return seq_list,label_list,error
                          
             
   
    
    def getVocab_train_data(self,seq_list,label_list,
                            vocab_size = 30000):
        '''
        Parameters
        ----------
        text_words : ['i, am ,happy,...']
            文本数据.
    
        Returns
        -------
        word2id : {'word':1}
        id2word : {1:'word'}
        data : [234, 222, 15,...]
        count : 筛选出的词典的词 [('word':2566),...]
        '''  
        def create_bi_dic(unk_tag,pad_tag,text_words):
            count = [(pad_tag,1),
                     (unk_tag,1)]
            #   检索最常见单词
            total_count = collections.Counter(text_words).most_common()
            count.extend(collections.Counter(text_words).most_common(vocab_size-1))
            
            #   删除少于 min_occurrence 的样本
            # min_count = 10
            # tmp = pd.DataFrame(count,columns = ['word','freq'])
            # vocab = tmp[tmp['freq']>=min_count]
            # ind = len(vocab)  #   记得 unk 
            # count = count[:ind]
            # tmp = pd.DataFrame(total_count,columns = ['word','freq'])
            # unk = tmp[~tmp['word'].isin(list(vocab['word']))]
            # unk_count = np.sum(unk['freq'])
            # count[0] = (unk_tag, unk_count)             
            # vocab_size = len(count)
            #   word2id , id2word
            word2id = dict()
            for i,(word,_) in enumerate(count):
                word2id[word] = i
            id2word = dict(zip(word2id.values(),word2id.keys()))
            print('vocab_size : ',vocab_size)
            return word2id,id2word          
        #   part1  生成映射表
        
        text_words = []
        for i in range(len(seq_list)):
            text_words.extend(seq_list[i])
        unk_tag = '<UNK>'   
        pad_tag = '<PAD>'          
        word2id,id2word = create_bi_dic(unk_tag,pad_tag,text_words)
              
        text_words = []
        for i in range(len(label_list)):
            text_words.extend(label_list[i])
        unk_tag = 'UNK' 
        pad_tag = 'PAD'            
        tag2id,id2tag = create_bi_dic(unk_tag,pad_tag,text_words)
                
        #   text to id
        x_train = []
        for seq in seq_list:
            add = list(map(lambda x : word2id.get(x,1),seq))
            x_train.append(add)
        y_train = []
        for seq in label_list:
            add = list(map(lambda x : tag2id.get(x,1),seq))
            y_train.append(add)
            
        
        return word2id,id2word,tag2id,id2tag,x_train,y_train


        
    def ner_data(self,max_seq_len = 128): 
        print('max_seq_len = 128')           
        x_train = pd.read_csv(open(self.x_train_dir,'rb'),header = None)
        y_train = pd.read_csv(open(self.y_train_dir,'r'),encoding='utf-8',header = None)        
              
       
        seq_list,label_list,error = self.create_text(x_train,y_train,
                                                seq_add_start=True,
                                                seq_add_end=True)   
                
        word2id,id2word,tag2id,id2tag,x,y = self.getVocab_train_data(seq_list,label_list)
               
        
        x_train =tf.keras.preprocessing.sequence.pad_sequences(x,
                                                        value=word2id["<PAD>"],
                                                        padding='post',
                                                        maxlen=max_seq_len)
        
        y_train =tf.keras.preprocessing.sequence.pad_sequences(y,
                                                        value=tag2id["PAD"],
                                                        padding='post',
                                                        maxlen=max_seq_len)

        return word2id,id2word,tag2id,id2tag,x_train,y_train,x,y

# =============================================================================
# train
# =============================================================================
corpus = Get_corpus()
word2id,id2word,tag2id,id2tag,x_train,y_train,x,y = corpus.ner_data()

vocab_size = len(word2id)
max_seq_len = int(x_train.shape[1])
tagSum = len(tag2id)

from BiLSTMCRF import MyBiLSTMCRF
# myModel=MyBiLSTMCRF(vocabSize,maxLen, tagIndexDict,tagSum,sequenceLengths)

'''
x_train ： type : numpy.ndarray; shape:(batch_size,max_seq_len)
y_train ： type : numpy.ndarray; shape:(batch_size,max_seq_len)

'''

nerModel=MyBiLSTMCRF(vocab_size,max_seq_len, tag2id,tagSum)
nerModel.myBiLSTMCRF.summary()

history=nerModel.fit(x_train,y_train,epochs=20)












