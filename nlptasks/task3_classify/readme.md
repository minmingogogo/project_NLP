
# 分类任务模型架构

> 多分类 ：地址分单任务


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

###	3.测试结果：
	





