import tensorflow as tf

from include.Config import Config
from include.metrics import get_hits, get_combine_hits, get_all_combine_hits
from include.Model_AE import GCN_Align
from include.Model_SE import build
from include.utils import *
from include.bootstrap import bootstrapping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import functools
print = functools.partial(print, flush=True)


def ae_session_init(train, ae_input, num_supports):
    print('####ae_session_init####')
    ph_ae = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32),      # tf.placeholder(tf.float32)
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder_with_default(0, shape=()),
        'ILL_left': tf.placeholder(tf.int32, [None]),
        'ILL_right': tf.placeholder(tf.int32, [None]),
    }
    model_ae = GCN_Align(ph_ae, input_dim=ae_input[2][1], output_dim=Config.ae_dim,
                         ILL=train, sparse_inputs=True, featureless=False, logging=True)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    return sess, ph_ae, model_ae


def se_session_init(e, train, KG1, KG2):
    print('####se_session_init####')
    sess, train_step, output_layer, loss = build(Config.se_learning_rate, Config.dim, Config.act_func, Config.alpha, Config.beta,
                                                 Config.gamma, Config.se_neg_K, Config.language[0:2], e, KG1 + KG2)
    return sess, train_step, output_layer, loss 


seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)


def training(sess_se, output_layer_se, loss_se, op_se, se_neg_K,
             sess_ae, model_ae, ae_input, support, ph_ae, ae_neg_K, base_attr, base_KG,
             e, epochs, train, test, all_ent_list,
             th, boot_K):
    print('####traning####')
    # init some data
    last_k_loss_se, last_k_loss_ae = [], []
    last_k_loss_ae = []
    # init some data for boot
    labeled_alignment_se, labeled_alignment_ae= set(), set()
    ents1_se, ents2_se, ents1_ae, ents2_ae,= [], [], [], []

    # base half neg data
    L = len(train)
    train = np.array(train)
    train_left = train[:, 0] 
    train_right = train[:, 1]
    neg_left = (np.ones((L, ae_neg_K), dtype=int) * (train_left.reshape((L, 1)))).reshape((L * ae_neg_K,))
    neg2_right = (np.ones((L, ae_neg_K), dtype=int) * (train_right.reshape((L, 1)))).reshape((L * ae_neg_K,))

    # ae 由于未预编码，得先训练30个epoch
    # ae feeddict init
    feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
    feed_dict_ae.update({ph_ae['dropout']: Config.dropout})
    for epoch in range(0, 30):
        if epoch == 0:
            neg2_left = np.random.choice(e, L * ae_neg_K)
            neg_right = np.random.choice(e, L * ae_neg_K)
        feed_dict_ae.update({'neg_left:0': neg_left,
                             'neg_right:0': neg_right,
                             'neg2_left:0': neg2_left,
                             'neg2_right:0': neg2_right,
                             "ILL_left:0": train_left,
                             "ILL_right:0": train_right})
        outs_ae = sess_ae.run([model_ae.opt_op, model_ae.loss, model_ae.outputs], feed_dict=feed_dict_ae)
        print("Epoch:", '%04d' % epoch, "AE_train_loss=", "{:.5f}".format(outs_ae[1]))
        if epoch % 10 == 0:
            neg_right = get_neg(train_left, all_ent_list[1], train_right, outs_ae[2], ae_neg_K)
            neg2_left = get_neg(train_right, all_ent_list[0], train_left, outs_ae[2], ae_neg_K)
    get_hits(outs_ae[2], test)

    align_left_se = train_left
    align_right_se = train_right
    align_left_ae = train_left
    align_right_ae = train_right
    ae_train_flag = True
    se_train_flag = True
    for epoch in range(epochs):
        # get outvec & get neg & train SE
        if se_train_flag:
            vecs_se = sess_se.run(output_layer_se)

            if epoch % 10 == 0:
                L = len(align_left_se)
                neg_left = (np.ones((L, se_neg_K), dtype=int) * (align_left_se.reshape((L, 1)))).reshape((L * se_neg_K,))
                neg2_right = (np.ones((L, se_neg_K), dtype=int) * (align_right_se.reshape((L, 1)))).reshape((L * se_neg_K,))
                neg_right = get_neg(align_left_se, all_ent_list[1], align_right_se, vecs_se, se_neg_K)
                neg2_left = get_neg(align_right_se, all_ent_list[0], align_left_se, vecs_se, se_neg_K)

            feed_dict_se = {"neg_left:0": neg_left,
                            "neg_right:0": neg_right,
                            "neg2_left:0": neg2_left,
                            "neg2_right:0": neg2_right,
                            "ILL_left:0": align_left_se,
                            "ILL_right:0": align_right_se}
            _, l_s = sess_se.run([op_se, loss_se], feed_dict=feed_dict_se)
            last_k_loss_se.append(l_s)

        # get neg & train AE
        if ae_train_flag:
            vecs_ae = sess_ae.run(model_ae.outputs, feed_dict=feed_dict_ae)

            if epoch % 10 == 0:
                L = len(align_left_ae)
                neg_left = (np.ones((L, ae_neg_K), dtype=int) * (align_left_ae.reshape((L, 1)))).reshape((L * ae_neg_K,))
                neg2_right = (np.ones((L, ae_neg_K), dtype=int) * (align_right_ae.reshape((L, 1)))).reshape((L * ae_neg_K,))
                neg_right = get_neg(align_left_ae, all_ent_list[1], align_right_ae, vecs_ae, ae_neg_K)
                neg2_left = get_neg(align_right_ae, all_ent_list[0], align_left_ae, vecs_ae, ae_neg_K)

            feed_dict_ae.update({'neg_left:0': neg_left,
                                 'neg_right:0': neg_right,
                                 'neg2_left:0': neg2_left,
                                 'neg2_right:0': neg2_right,
                                 "ILL_left:0": align_left_ae,
                                 "ILL_right:0": align_right_ae})
            _, l_a = sess_ae.run([model_ae.opt_op, model_ae.loss], feed_dict=feed_dict_ae)
            last_k_loss_ae.append(l_a)

        # print loss 
        print("Epoch:", '%04d' % epoch, "AE_train_loss=", "{:.5f}".format(l_a), "SE_train_loss=", "{:.5f}".format(l_s))
        # print("Epoch:", '%04d' % epoch, "AE_train_loss=", "{:.5f}".format(l_a))

        # get hits & bootstrap
        if epoch % 10 == 0:
            if se_train_flag:
                print('SE')
                get_hits(vecs_se, test)
            if ae_train_flag:
                print('AE')
                get_hits(vecs_ae, test)
            print('SE + AE')
            get_combine_hits(vecs_se, vecs_ae, Config.combine_loss_beta, test)

            if se_train_flag & ae_train_flag:
                labeled_alignment_se, ents1_se, ents2_se \
                    = bootstrapping(vecs_se, test, labeled_alignment_se, th[0], boot_K[0])
                # labeled_alignment_ae, ents1_ae, ents2_ae \
                #     = bootstrapping(vecs_ae, test, labeled_alignment_ae, th[1], boot_K[1])
                # ents1_se, ents2_se, ents1_ae, ents2_ae = del_duplicate(ents1_se, ents2_se, ents1_ae, ents2_ae)
                print(labeled_alignment_se)
                # print(labeled_alignment_ae)
            if ents1_se != []:
                align_left_ae = np.append(train_left, np.array(ents1_se))
                align_right_ae = np.append(train_right, np.array(ents2_se))
                ae_input, support = get_new_all_ae_data(e, base_attr, base_KG, align_left_ae, align_right_ae)
                feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
                feed_dict_ae.update({ph_ae['dropout']: Config.dropout})

            # if ents1_ae != []:
            #     align_left_se = np.append(train_left, np.array(ents1_ae))
            #     align_right_se = np.append(train_right, np.array(ents2_ae))

            continue

        # early_stopping
        if ae_train_flag and epoch > Config.early_stopping \
                and last_k_loss_ae[-1] > np.mean(last_k_loss_ae[-(Config.early_stopping + 1):-1]):
            print('AE early stopping...')
            ae_train_flag = False
        if se_train_flag and epoch > Config.early_stopping \
                and last_k_loss_se[-1] > np.mean(last_k_loss_se[-(Config.early_stopping + 1):-1]):
            print('SE early stopping...')
            se_train_flag = False
        if ae_train_flag is False and se_train_flag is False:
            print("Early stopping...")
            break
    
    vecs_se = sess_se.run(output_layer_se)
    vecs_ae = sess_ae.run(model_ae.outputs, feed_dict=feed_dict_ae)
    get_hits(vecs_se, test)
    get_hits(vecs_ae, test)
    get_all_combine_hits(vecs_se, vecs_ae, Config.beta_list, test)

    sess_se.close()
    sess_ae.close()
