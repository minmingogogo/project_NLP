# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 19:36:59 2020

@author: Scarlett
"""

import numpy as np
import pandas as pd
import sklearn
import os
import json

def load_embedding_data(config):
    dump_dir = 'D:\\myGit\\Algorithm\\A_my_nlp_project\\project_NLP\\pretrain_weights'
    filename = 'gensim_50000vocab_300d'
    filepath = os.path.join(dump_dir,filename)
    embed_matrix_dir = os.path.join(filepath,'embedding_matrix.txt')
    word2id_dir = os.path.join(filepath,'word2id.json')
    id2word_dir = os.path.join(filepath,'id2word.json')
    
    word2id = json.load(open(word2id_dir))    
    id2word = json.load(open(id2word_dir))    
    embedding_matrix = np.load(embed_matrix_dir,allow_pickle=True)
    # config['max_len'] = embedding_matrix.shape[1]
    config['embed_size'] = embedding_matrix.shape[1]
    if config['vocab_size']< embedding_matrix.shape[0]:
        embedding_matrix = embedding_matrix[:config['vocab_size']]
    
    return embedding_matrix,word2id,id2word



