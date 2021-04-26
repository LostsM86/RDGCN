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
	neg_K = 125		# number of negative samples for each positive one
	combine_loss_beta = 0.9		# se_loss的比重
	early_stopping = 40		# early stopping

	# SE setting
	dim = 300
	act_func = tf.nn.relu
	alpha = 0.1
	beta = 0.3
	gamma = 1.0		# margin based loss of SE
	learning_rate = 0.001

	# AE setting
	dropout = 0.		# Dropout rate (1 - keep probability)
	ae_gamma = 3.0 		# Hyper-parameter for margin based loss of AE
