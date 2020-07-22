
# 分类任务模型架构

> 多分类 ：地址分单任务
##	1.数据集说明：
	


##	2.试验方案：

# 优化器/优化策略比较

> 以一个二分类情感分类任务对比几种优化器优化策略效果
> *.ipynb 文件均为colab 上实现，可以直接查看复现
> 同名*.py 文件为导出副本

##	1.数据集说明：
	根据用户正负面评价标记的文本
	text:蒙牛大果粒！果断想起高三奋斗的那段日子～
	label:[1]


##	2.试验方案：
		
	A:
	model : 
		bert_base 4层 + Lambda 取[Cls] + Dense(num_classes)
	optimizer:
		lr = 1e-4
		Adam
		AdamLM:分段线性学习率,学习率线性变化warmup内增长，后衰减到10%
		AdamEMA:权重滑动平均
		Adam_GAcc:梯度累计
		Adam_LayerAdapt : 层自适应
		Adam_WD : 权重衰减
		lr = 1e-3
		SGD
		SGD_LookAhead : LOOKAHEAD

	B:
	model : 
		albert_small + Lambda 取[Cls] + Dense(num_classes)
	optimizer:
		后缀为采用的learning_rate
		al_Adam_1e-4
		al_Adam_1e-3
		al_SGD_1e-4
		al_SGD_1e-3

###	3.测试结果  
**fine-tune 方案选择优化器就可以了，叠加其他策略未必会更好。**  
在这个二分类任务中，模型表现各种优化器的表现，训练过程albert_small sgd 1e-3最快收敛，其次是Adam 1e-4.   
测试集表现  AdamWD最快达到最优，SGD次之。 考虑显存占用的话，SGD无疑是比Adam更好的选择。   
见下图

###	4.测试结论
如果预训练模型一般在前1-2个epoch 就可以达到最优，此时优化器用简单的省显存的

####	train accuracy  
![train](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task3_classify/img/trainacc.png)

####	best test accuracy  	
![test](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task3_classify/img/valacc.png)

####	loss
![loss](https://github.com/minmingogogo/project_NLP/blob/master/nlptasks/task3_classify/img/loss.png)

###	5.改进与优化
	
通过分析预测错误数据发现，有部分是标记错误或者正负情绪皆有的。  
而模型真正错误的几乎是在后半句转折上。相应的改正策略如下：    

1.增加输入特征，长句截断为若干段，每段embedding ，目的是加入长句情绪变化过程让模型学习这种变化模式  
2. 知识蒸馏  

	1）model ensamble 对训练集采用stacking方案（一般分5-8份），
	假设获得5个模型，每个模型都对训练集预测，如果5个预测值都一致则取出数据修正

	2）设置分割阈值，长度超阈值的分割两段分别入模型判定，如果两段预测都与标记不一致则取出修正
	


自训练模型时候叠加优化策略可能有提升帮助，在下一个任务中比较  
----  
阅读材料
玩转Keras之小众需求：自定义优化器(https://zhuanlan.zhihu.com/p/44448328)  


