import tensorflow as tf


class Config:
	language = 'ja_en' # zh_en | ja_en | fr_en
	e1 = 'data/' + language + '/ent_ids_1'
	e2 = 'data/' + language + '/ent_ids_2'
	ill = 'data/' + language + '/ref_ent_ids'
	kg1 = 'data/' + language + '/triples_1'
	kg2 = 'data/' + language + '/triples_2'
	# se
	epochs = 600
	dim = 300
	act_func = tf.nn.relu
	alpha = 0.1
	beta = 0.99
	gamma = 1.0  # margin based loss
	k = 100  # number of negative samples for each positive one
	seed = 3  # 30% of seeds
	th = 0.2		# bootstraping filter
	boot_K = 30		# 前boot_k作为candidate
	heuristic = True
	early_stopping = 10

	# ae
	data_frequent_p = 0.95
	batch_size = 2000
	num_sampled_negs = 1000
	num_train = 100
	min_frequency = 5  # small data 5
	min_props = 2
	lr = 0.1
	v = 2
	embedding_size = 300

	# combine hits
	loss_beta = 0.9
