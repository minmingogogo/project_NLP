import tensorflow as tf
# from .transformer import transformer
# from .rnn import rnn_layer
# from .crf import crf_log_likelihood,crf_decode
from model_transformer.transformer import transformer
from model_transformer.rnn import rnn_layer
from model_transformer.crf import crf_log_likelihood,crf_decode
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import categorical_crossentropy

# y_true = [[0, 1, 0], [0, 0, 1]]
# y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
# loss = categorical_crossentropy(y_true, y_pred)





class Forward(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Forward, self).__init__()
        init = tf.keras.initializers.GlorotUniform()
        tgt_size = config.tgt_size
        self.transition_params = tf.Variable(lambda : init([tgt_size,tgt_size]))
    def call(self,output,label_input=None,training=1):
        
        if training:
            # print(1)
            ins = [output,label_input,self.transition_params]
            loss, self.transition_params = crf_log_likelihood(ins)
            return loss
        else:
            decode_tags, _ = crf_decode([output,self.transition_params])
            return decode_tags
        
        
# def compute_loss(self, predictions, labels, num_class=2, ignore_index=None):
#     if ignore_index is None:
#         loss_func = CrossEntropyLoss()
#     else:
#         loss_func = CrossEntropyLoss(ignore_index=ignore_index)
#     return loss_func(predictions.view(-1, num_class), labels.view(-1))


#         self.next_loss_func = CrossEntropyLoss()
#         self.mlm_loss_func = CrossEntropyLoss(ignore_index=0)

#     def compute_loss(self, predictions, labels, num_class=2, ignore_index=-100):
#         loss_func = CrossEntropyLoss(ignore_index=ignore_index)
#         return loss_func(predictions.view(-1, num_class), labels.view(-1))

        
class ner_model(tf.keras.Model):
    """docstring for ner_model"""
    def __init__(self, config):
        super(ner_model, self).__init__()
        self.config = config
        # self.training = training
        self.tf_layer = transformer(config)
        self.rnn_layer = rnn_layer(config)
        self.dense_layer = tf.keras.layers.Dense(config.tgt_size)
        self.ffw = Forward(config)
    def call(self,seq_input,label_input=None,training=1):
        output = self.tf_layer(seq_input,training)
        if self.config.rnn:
            output = self.rnn_layer.bi_rnn(output,training)
        output = self.dense_layer(output)
        if training:
            return self.ffw(output, label_input,training)
        else:
            return self.ffw(output,training)
        
class Classify_model(tf.keras.Model):
    """docstring for classify_model"""
    def __init__(self, config):
        super(Classify_model, self).__init__()
        self.config = config
        # self.training = training
        self.tf_layer = transformer(config)
        self.maxpooling_layer = L.GlobalAveragePooling1D()
        self.dense_layer = L.Dense(config.num_classes,activation = 'relu')
        # self.ffw = Forward(config)
        self.softmax_layer = L.Dense(config.num_classes,activation = 'softmax')
    def call(self,seq_input,label_input=None,training=1):
        output = self.tf_layer(seq_input,training)
        output = self.maxpooling_layer(output)
        output = self.dense_layer(output)
        output = self.softmax_layer(output)
        if training:
            loss = categorical_crossentropy(label_input, output)
            return output,loss
        else:
            return output
     
        
        
        
# if __name__ == '__main__':
# #     config = nerConfig()
    
#     cla_model2 = Classify_model(config)
#     dev_set = cla_batch
#     # dev_set = test_cla_batch
#     for index,batch_val in enumerate(dev_set):
#         if index>1:
#             break

#     infers = []
#     datas = []
#     labels = []
#     # dev_set = test_cla_batch
#     # cla_model, train_loss, ckpt, ckpt_manager, optimizer = train_classify_ops

#     for index,batch_val in enumerate(dev_set):
#         # if index>1:
#         #     break    
#         batch_val_data,batch_val_label = batch_val
#         batch_val_pre = cla_model2(batch_val_data)
#         for batch_ind in range(len(batch_val_pre)):
#             infers.append(batch_val_pre[batch_ind].numpy())
#             datas.append(batch_val_data[batch_ind].numpy())
#             labels.append(batch_val_label[batch_ind].numpy())

      
#         batch_val_data,batch_val_label = batch_val
#         batch_val_pre = cla_model(batch_val_data)  
#         seq_input = batch_val_data
#         training = 0
#         output = cla_model.tf_layer(seq_input,training)
#         # TensorShape([16, 80, 256])
#         output = cla_model.maxpooling_layer(output)
#          # TensorShape([16, 256])
#         output = cla_model.dense_layer(output)
#         # TensorShape([16, 96])
#         output = cla_model.softmax_layer(output)    
#          # TensorShape([16, 96])
    
#     # model.build(input_shape=(config.batch_size,config.max_length))
    
#     transformer_weights = ner.tf_layer.get_weights()
#     outdir = checkpoint_dir + 'transformer_weights.pkl'
#     f= open(outdir,'wb')
#     pickle.dump((transformer_weights),f)        
#     ''' 加载 transformer_weights'''
#     import pickle
#     checkpoint_dir = "checkpoint/"    
#     transformer_save_dir = checkpoint_dir + 'transformer_weights.pkl'
#     f= open(transformer_save_dir,'rb')
#     transformer_weights = pickle.load(f)    
    
    
    
#     b = L.Dense(1,
#                 kernel_initializer=tf.constant_initializer(2.))
#     b(tf.convert_to_tensor([[10., 20., 30.]]))
#     b.set_weights(transformer_weights)
#     y_cla_label
#     x    
#     b.set_weights(a.get_weights())
#     b.get_weights()
#     # =============================================================================
#     #  加载原来模型   
#     # =============================================================================
 
    
#     transformer_fine_tune = Sequential()
#     transformer_fine_tune.add(transformer(config))
#     #   加全连接层
#     transformer_fine_tune.add(L.Dense(config.num_classes,activation = 'relu'))    
#     transformer_fine_tune.add(L.Dense(config.num_classes,activation = 'softmax'))
#     transformer_fine_tune.compile(optimizer = 'adam',
#                                   loss = 'categorical_crossentropy',
#                                   metrics=['accuracy'])
#     transformer_fine_tune.build((None,80))
#     transformer_fine_tune.summary()
    
#     transformer_fine_tune.fit(x,y_cla,
#                               batch_size = config.batch_size,
#                               epochs = 10,
#                               verbose = 2,
#                               validation_split = 0.2
#                               )

# #   只训练最后一层
#     transformer_fine_tune.layers[0].set_weights(transformer_weights)        
#     transformer_fine_tune.layers[0].trainable = False    
    
# #     training=True
# #     ner = ner_model(config)
# #     loss = ner(batch_data,batch_label,training)
# #     batch_data,batch_label = batch
#     # tf 层 = ner.tf_layer.get_weights()
        
        