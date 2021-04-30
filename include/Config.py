import tensorflow as tf


class Config:
	language = 'zh_en'		# zh_en | ja_en | fr_en
	e1 = 'data/' + language + '/ent_ids_1'
	e2 = 'data/' + language + '/ent_ids_2'
	ill = 'data/' + language + '/ref_ent_ids'
	kg1 = 'data/' + language + '/triples_1'
	kg2 = 'data/' + language + '/triples_2'

	seed = 3		# 30% of seeds
	epochs = 2000
	combine_loss_beta = 0.8		# se_loss的比重
	beta_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	early_stopping = 40		# early stopping
	# neg_K = 45

	# SE setting
	dim = 300
	# act_func = tf.nn.relu
	act_func = tf.nn.selu
	alpha = 0.1
	beta = 0.3
	gamma = 3.0		# margin based loss of SE
	# se_learning_rate = 0.001
	se_learning_rate = 0.001
	se_neg_K = 145  # number of negative samples for each positive one of SE

	# AE setting
	dropout = 0.		# Dropout rate (1 - keep probability)
	ae_gamma = 3.0 		# Hyper-parameter for margin based loss of AE
	ae_dim = 300
	ae_neg_K = 45		# number of negative samples for each positive one of AE
	ae_learning_rate = 0.1		# Initial learning rate of AE

	# bootstrap setting
	th = [0.5, 0.65]		# Threshold of [SE, AE]
	boot_K = [40, 40]		# 取boot_K对作为输入到二分图的结果
