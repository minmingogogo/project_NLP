
总结一些tf2 keras框架下实现的自然语言任务,及keras自定义操作

 1. nlp 任务， 已上传：  
   [task_2 实体识别](https://github.com/minmingogogo/project_NLP/tree/master/nlptasks/task2_ner)  
   [task_4 seq2seq 翻译](https://github.com/minmingogogo/project_NLP/tree/master/nlptasks/task4_seq2seq)  
   [task_5 实体抽取](https://github.com/minmingogogo/project_NLP/tree/master/nlptasks/task5_extraction)  
   
 2. 灵活模型架构下自定义keras模块    
   > 说明
    大型模型显存占用扼住了整个项目命运咽喉，从设备性能,接口性能,移动端部署考虑，
    降低显存占用方案与模型设计/选择同等重要。
    显存占用包括：输入输出，模型参数，梯度计算。模型架构精简，优化器选择都是降低显存的手段。
    
   **自定义optimizer**    
      
      对比了ALbert 与 Bert 相同任务性能  
      对比 AdaFactor 与 Adam 两种优化器的效果  
      
   **自定义loss/metrics**       
   一般情况下loss func 由y_pred,y_true 计算得到,model.add_loss(loss_tensor) 与 model.compile(loss=custom_lossfunc)两种都可以轻易改写，
   此处总结了更灵活的多输入loss实现方式
   
      实现更复杂的多输入loss : triplet loss,center loss 
      
   **自定义callback**    
   继承Callback 写自定义模块可以实现基于batch,epoch 等灵活操作
   
      打印自定义metrics    
      基于自定义best_acc保存模型   
      
     

## 
