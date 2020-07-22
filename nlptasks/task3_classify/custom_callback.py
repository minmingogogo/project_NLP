# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 01:21:22 2020

@author: Scarlett
"""

'''2.3 customize callback  ''' 
class Customize_Accuracy(K.metrics.Metric):
    def __init__(self,name = "cate_Accuracy", **kwargs):
        super(Customize_Accuracy,self).__init__(name=name, **kwargs)
        self.acc =  self.add_weight(name="cate_A", initializer="zeros")
    def update_state(self,y_true,y_pred,sample_weight = None):
        
# =============================================================================
#   
        # y_pred = model.predict(x_val[:3])
        # y_true = y_val[:3]
# =============================================================================
        # 计算：
        # 1 取y_pred 最大值
        # 2 max1-max2>threshold
        threshold = 0.4
        pred = np.argmax(y_pred, 1)        #用于 1
        sorted_soft_list = np.sort(y_pred) #用于 2
        jiexi_result = [max_ind if sorted_soft_list[i, -1] - sorted_soft_list[i, -2] >= threshold else 0 for i,max_ind in enumerate(pred) ]
        total = len([x for x in jiexi_result if x !=0])
        true = np.argmax(y_true, 1) 
        tp = len(true[true == jiexi_result])
        acc = tp/total
        values = tf.cast(acc, "float32")
        
        self.acc.assign_add(values)
        

        max_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        max_pred = tf.squeeze(max_pred)
        sorted_soft_ind= tf.argsort(y_pred, direction='DESCENDING')
        max_true = tf.argmax(y_true, axis=1)
        sorted_soft_value= tf.sort(y_pred, direction='DESCENDING')
        is_greater = [1 if (x[0]-x[1])>threshold else 0 for x in sorted_soft_value]
        total = tf.reduce_sum(is_greater)
        is_equal = max_true == max_pred
        
        [1 if x[1]==1 and x[0] ]
        values = tf.cast(values, "float32")

    def result(self):
        return self.acc

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.acc.assign(0.0)
     
        
class Customize_callback(K.callbacks.Callback):
    def __init__(self,name = "cate_Accuracy", **kwargs):
        super(Customize_callback,self).__init__(name=name, **kwargs)
        self.acc =  self.add_weight(name="cate_A", initializer="zeros")
    def update_state(self,y_true,y_pred,sample_weight = None):
        
# =============================================================================
#   
        # y_pred = model.predict(x_val[:3])
        # y_true = y_val[:3]
# =============================================================================
        # 计算：
        # 1 取y_pred 最大值
        # 2 max1-max2>threshold
        threshold = 0.4
        pred = np.argmax(y_pred, 1)        #用于 1
        sorted_soft_list = np.sort(y_pred) #用于 2
        jiexi_result = [max_ind if sorted_soft_list[i, -1] - sorted_soft_list[i, -2] >= threshold else 0 for i,max_ind in enumerate(pred) ]
        total = len([x for x in jiexi_result if x !=0])
        true = np.argmax(y_true, 1) 
        tp = len(true[true == jiexi_result])
        acc = tp/total
        values = tf.cast(acc, "float32")
                
# =============================================================================
class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )
        
        
        
        
        
        
# =============================================================================