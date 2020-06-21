import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LayerNormalization,Conv1D,Dropout,Embedding

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) *
        (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf

def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def merge_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
    return tf.reshape(x, new_x_shape)

def split_heads(x,n_head):
    x_shape = shape_list(x)
    new_x_shape = x_shape[:-1] + [n_head, x_shape[-1] // n_head]
    x = tf.reshape(x, new_x_shape)
    return tf.transpose(x, (0, 2, 1, 3))#(batch, head, seq_length, head_features)

def positional_encoding(config):
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / config['embed_size']) for i in range(config['embed_size'])]
        if pos != 0 else np.zeros(config['embed_size']) for pos in range(config['max_len'])])
    # type 1 时刻 min max mean,(80, 256)
    # ____________________________
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    # type 2 时刻 min max mean
    # ____________________________
    #   归一化
    position_enc = position_enc / (denominator + 1e-8)
    position_enc = position_enc[tf.newaxis, :]
    pos_sincos_embedding = tf.convert_to_tensor(position_enc)
    # pos_sincos_embedding.shape
    # type 3 时刻 min max mean
    # ____________________________
    return pos_sincos_embedding



class attention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(attention, self).__init__()
        self.conv1 = Conv1D(filters = 3 * config['embed_size'],kernel_size=1)
        self.conv2 = Conv1D(filters = config['embed_size'],kernel_size=1)
        self.dropout = Dropout(config['attention_dropout'])
        self.head = config['head']
        self.n_state = config['embed_size']
    def call(self,x, scale_att=False,training=1):
        self.training = training
        x = self.conv1(x)
        q, k, v = tf.split(x, 3, axis=2)
        # 
        #   这里报错是应为头数量不能平均分
        # print('config['embed_size'] % config.head :{} 该数量不为0 则报错')
        assert self.n_state % self.head == 0
        q = split_heads(q, self.head)
        k = split_heads(k, self.head)
        v = split_heads(v, self.head)

        w = tf.matmul(q, k, transpose_b=True)
        if scale_att:
            dk = tf.cast(shape_list(k)[-1], tf.float32)
            w = w / tf.math.sqrt(dk)
        w = tf.nn.softmax(w, axis=-1)
        w = self.dropout(w, training=self.training)
        a = tf.matmul(w, v)

        a = merge_heads(a)
        a = self.conv2(a)
        a = self.dropout(a, training=self.training)

        return a

class FFN(tf.keras.layers.Layer):
    """docstring for FFN"""
    def __init__(self, config):
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=config['ffw_rate']*config['embed_size'],kernel_size=1)
        self.conv2 = Conv1D(filters=config['embed_size'],kernel_size=1)
        self.dropout = Dropout(config['attention_dropout'])
    def call(self,x,training):
        self.training = training
        ffn0 = self.conv1(x)
        act_ffn0 = gelu(ffn0)
        ffn1 = self.conv2(act_ffn0)
        ffn1 = self.dropout(ffn1, training=self.training)
        return ffn1

class attention_block(tf.keras.layers.Layer):
    def __init__(self, config):
        super(attention_block, self).__init__()
        self.attention = attention(config)
        self.ln = LayerNormalization(epsilon=config['layer_norm_epsilon'])
        self.ffn = FFN(config)
    def call(self,x, scale_att=False,training=True):
        a = self.attention(x,scale_att=scale_att,training = training)
        x = x + a
        x = self.ln(x)

        m = self.ffn(x,training)
        x = x + m#  残差连接
        x = self.ln(x)

        return x


# class transformer(tf.keras.layers.Layer):
#     """docstring for transformer"""
#     def __init__(self, config):
#         super(transformer, self).__init__()
#         self.config = config
#         self.wde = Embedding(config.vocab_size, config['embed_size'])
#         self.pte = Embedding(config['max_len'], config['embed_size'])
#         self.attention_blocks = [attention_block(config) for i in range(self.config.n_layer)]
#         self.ln = LayerNormalization(epsilon=self.config.layer_norm_epsilon)
#     def call(self,input_ids,training):
#         seq_embedding = self.wde(input_ids)
#         batch,seq_len,emb = shape_list(seq_embedding)
#         position_ids = tf.range(0, seq_len, dtype=tf.int32)[tf.newaxis, :]
#         position_embedding = self.pte(position_ids)
#         hidden_state = self.ln(seq_embedding + position_embedding)
#         for i in range(self.config.n_layer):
#             hidden_state = self.attention_blocks[i](hidden_state,scale_att=True,training=training)
#         return hidden_state
    
class transformer(tf.keras.layers.Layer):
    """
    docstring for transformer
    采用sin cos 位置编码
    """
    def __init__(self, config):
        super(transformer, self).__init__()
        self.config = config
        self.wde = Embedding(config['vocab_size'],
                            config['embed_size'])
        
        # 数字位置编码
        
        self.pte = Embedding(config['max_len'], config['embed_size'])
      
        self.attention_blocks = [attention_block(config) for i in range(self.config['num_layers'])]
        self.ln = LayerNormalization(epsilon=self.config['layer_norm_epsilon'])
    def call(self,input_ids,training):
        seq_embedding = self.wde(input_ids)
        batch,seq_len,emb = shape_list(seq_embedding)
        # position_ids = tf.range(0, seq_len, dtype=tf.int32)[tf.newaxis, :]
        # position_embedding = self.pte(position_ids)        
        position_embedding = positional_encoding(self.config)
        position_embedding = tf.cast(position_embedding,dtype=seq_embedding.dtype)
        hidden_state = self.ln(seq_embedding + position_embedding)
        for i in range(self.config['num_layers']):
            hidden_state = self.attention_blocks[i](hidden_state,scale_att=True,training=training)
        return hidden_state
        
    
if __name__ == '__main__':
    config = {
        'epoch':1000,
        'vocab_size':3747,
        'tgt_size':44,
        'max_len':256,
        'num_layers':4,
        'head':8,
        'embed_size':256,
        'ffw_rate':4,
        'attention_dropout':0.2,
        'layer_norm_epsilon':1e-5,
        'batch_size':16,
        'lr':1e-4,
        'dynamics_lr':True,
        # rnn=True,
        'rnn_unit':384,
        'rnn_dropout':0.2,
        # label_train_type='BIES',
        # train_type='ts',
        # reset_vocab=False,        
        }
    
    tf_layer = transformer(config)
    
#     for i in cla_batch.take(1):
#         input_ids = i[0]
#         y_ids = i[1]
    
# train_set = cla_batch
# for epoch in range(5):
#     @tf.function(experimental_relax_shapes=True)
#     def train_step(input_ids,input_labels):
#         with tf.GradientTape() as tape:
#             # loss = ner(batch_data,batch_label)
#             loss = cla_model(input_ids,input_labels,training = training)
#             # loss = cla_model(batch_data,batch_label,training = training)
            
#         gradients = tape.gradient(loss, cla_model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, cla_model.trainable_variables))
#         train_loss(loss)#   loss 得到固定值后，多次调用train_loss(loss) ，会一直变；下面打印train_loss.result()平均损失而执行的实例操作

#     train_loss.reset_states()
#     # tq = tqdm(enumerate(train_set))
#     patience = 0
#     last_loss = None  
#     delta  =0.001    
#     for index,batch in enumerate(train_set):
#         # if index>1:
#         #     break
#         batch_data,batch_label = batch
#         train_step(batch_data,batch_label)
#         #-------early stop----------------
#         if not last_loss:
#             last_loss = train_loss.result()
#         else:
#             if abs(last_loss - train_loss.result())>delta:
#                 patience = 0
#                 last_loss = train_loss.result()
#             else:
#                 patience +=1
#             if patience >=5:
#                 print('earlystop Epoch {} Loss {:.4f}'.format(epoch,train_loss.result()))
#                 break
#         #-------early stop----------------        
        
        
#         print('Epoch {} Loss {:.4f}'.format(epoch,train_loss.result()))
#         if index % 50 == 0 and index > 0:
#             save_path = ckpt_manager.save()
#             print("Saved checkpoint {}".format(save_path))    
#     training=True
#     config = nerConfig()
#     ner = ner_model(config,training=True)
#     tf_layer= transformer(config,training)
#     loss = ner(batch_data,batch_label)
#     batch_data,batch_label = batch
#     input_ids  = batch_data
#     out = tf_layer(input_ids)
    
#     for i in cla_batch.take(1):
#         input_ids = i[0]
#         tar= i[1]

# #   输入层    
# wde = Embedding(config.vocab_size, config['embed_size'])    
# seq_embedding = wde(input_ids)    
# # seq_embedding.shape
# # Out[70]: TensorShape([16, 80, 256])    
# tmp = seq_embedding[0]
# for i in range(10):
#     # print(i)
#     print('{}--- mean :{}, max:{}, min{} '.format(i,
#                                                   np.mean(tmp[i].numpy()),
#                                                   np.max(tmp[i].numpy()),
#                                                   np.min(tmp[i].numpy())))   
    
# #   数字位置编码    
# position_ids = tf.range(0, config['max_len'], dtype=tf.int32)[tf.newaxis, :]
# pteNum_layer = Embedding(config['max_len'], config['embed_size'])    
# position_embedding = pteNum_layer(position_ids)    
# tf.float32
# TensorShape([1, 80, 256])    
# tmp = position_embedding[0]
# for i in range(10):
#     # print(i)
#     print('{}--- mean :{}, max:{}, min{} '.format(i,
#                                                   np.mean(tmp[i].numpy()),
#                                                   np.max(tmp[i].numpy()),
#                                                   np.min(tmp[i].numpy())))
# 0--- mean :0.0022594851907342672, max:0.04993665590882301, min-0.049978554248809814 
# 1--- mean :0.0014203183818608522, max:0.04976823553442955, min-0.04941979795694351 
# 2--- mean :-0.0014445874840021133, max:0.04975130781531334, min-0.04986117035150528 
# 3--- mean :-0.002562296111136675, max:0.047771941870450974, min-0.049711015075445175 
# 4--- mean :-0.0010646393056958914, max:0.049910854548215866, min-0.04988689348101616 
# 5--- mean :0.0005007024155929685, max:0.049938250333070755, min-0.04936189576983452 
# 6--- mean :-0.0006684516556560993, max:0.04959428682923317, min-0.04993182420730591 
# 7--- mean :0.002912007737904787, max:0.04993424192070961, min-0.04992300271987915 
# 8--- mean :0.0019131595036014915, max:0.049503158777952194, min-0.04982781410217285 
# 9--- mean :-0.0009648598497733474, max:0.049853090196847916, min-0.04992120340466499 

# #   sin cos 编码   
    
# position_enc = np.array([
#     [pos / np.power(10000, 2 * i / config['embed_size']) for i in range(config['embed_size'])]
#     if pos != 0 else np.zeros(config['embed_size']) for pos in range(config['max_len'])])
# # type 1 时刻 min max mean,(80, 256)
# # ____________________________
# position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
# position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
# denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
# # type 2 时刻 min max mean
# # ____________________________
# #   归一化
# position_enc = position_enc / (denominator + 1e-8)
# position_enc = position_enc[tf.newaxis, :]
# pos_sincos_embedding = tf.convert_to_tensor(position_enc)
# pos_sincos_embedding.shape
# Out[76]: TensorShape([1, 80, 256])
# tmp = pos_sincos_embedding[0]
# for i in range(10):
#     # print(i)
#     print('{}--- mean :{}, max:{}, min{} '.format(i,
#                                                   np.mean(tmp[i].numpy()),
#                                                   np.max(tmp[i].numpy()),
#                                                   np.min(tmp[i].numpy())))

# 0--- mean :0.0, max:0.0, min0.0 
# 1--- mean :0.04597140616790708, max:0.08826072576118466, min1.0192189606463282e-09 
# 2--- mean :0.046100648414741055, max:0.08825532971803061, min-0.025266216074379354 
# 3--- mean :0.0449097830734701, max:0.08839034240794431, min-0.08303521731436805 
# 4--- mean :0.04315491736003979, max:0.08826443066588409, min-0.08796946840405073 
# 5--- mean :0.041973695426334556, max:0.08824718715406535, min-0.08761139946998135 
# 6--- mean :0.04174156751153274, max:0.08839680281553744, min-0.08839665993629041 
# 7--- mean :0.041902534437688294, max:0.08826721506915171, min-0.0882261201731595 
# 8--- mean :0.042000793306546644, max:0.08823429591965566, min-0.0882340743901494 
# 9--- mean :0.041720034164190206, max:0.0884093891447807, min-0.08810353221908088 

# pteSinCos_layer = Embedding(config['max_len'], config['embed_size'],weights = [position_enc])


