import tensorflow as tf


class Config:
	language = 'ja_en'		# zh_en | ja_en | fr_en
	e1 = 'data/' + language + '/ent_ids_1'
	e2 = 'data/' + language + '/ent_ids_2'
	ill = 'data/' + language + '/ref_ent_ids'
	kg1 = 'data/' + language + '/triples_1'
	kg2 = 'data/' + language + '/triples_2'

	seed = 3		# 30% of seeds
	epochs = 600
	combine_loss_beta = 0.9		# se_loss的比重
	early_stopping = 40		# early stopping
	# neg_K = 45

	# SE setting
	dim = 300
	act_func = tf.nn.relu
	alpha = 0.1
	beta = 0.3
	gamma = 1.0		# margin based loss of SE
	se_learning_rate = 0.001
	se_neg_K = 145  # number of negative samples for each positive one of SE

	# AE setting
	dropout = 0.		# Dropout rate (1 - keep probability)
	ae_gamma = 3.0 		# Hyper-parameter for margin based loss of AE
	ae_dim = 300
	ae_neg_K = 45		# number of negative samples for each positive one of AE
	ae_learning_rate = 15		# Initial learning rate of AE

	# bootstrap setting
	th = [0.7, 0.7]		# Threshold of [SE, AE]
	boot_K = [40, 20]		# 取boot_K对作为输入到二分图的结果
