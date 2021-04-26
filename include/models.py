import tensorflow as tf

from include.Config import Config
from metrics import get_hits, get_combine_hits
from Model_AE import GCN_Align
from Model_SE import build
from utils import *


def get_neg(ILL, cand_ent_list, other_ILL, output_layer, k):
    # ILL_R, all_ent1_list,  ILL_L, out, k)
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    cand_ent_vec = np.array([output_layer[e2] for e2 in cand_ent_list])
    sim = scipy.spatial.distance.cdist(ILL_vec, cand_ent_vec, metric='cityblock')
    for i in range(t):
        rank = sim[i, :].argsort()
        for j in rank[0:k+1]:
            if cand_ent_list[j] != other_ILL[i]:
                neg.append(cand_ent_list[j])
        if len(neg) == (i + 1) * k + 1:
            neg = neg[:-1]
    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg


def ae_session_init(train, ae_input, num_supports):
    ph_ae = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32), #tf.placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder_with_default(0, shape=())
    }
    model_ae = GCN_Align(ph_ae, input_dim=ae_input[2][1], output_dim=Config.ae_dim,
                       ILL=train, sparse_inputs=True, featureless=False, logging=True)
    sess = tf.Session()
    sess.run(model_ae.init)
    return sess, ph_ae, model_ae


def se_session_init(e, train, KG1, KG2):
    sess, train_step, output_layer, loss = build(Config.learning_rate, Config.dim, Config.act_func, Config.alpha, Config.beta,
                                                 Config.gamma, Config.neg_K, Config.language[0:2], e, train, KG1 + KG2)
    return sess, train_step, output_layer, loss 


def training(sess_se, output_layer_se, loss_se, op_se,
             sess_ae, model_ae, ae_input, support, ph_ae,
             e, epochs, neg_K, train, test, all_ent1_list, all_ent2_list):
    # init some data
    last_k_loss_se, last_k_loss_ae = [], []
    last_k_loss_ae = []
    labeled_alignment = set()
    ents1, ents2 = [], []

    # ae feeddict init
    feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
    feed_dict_ae.update({ph_ae['dropout']: Config.dropout})

    # base half neg data
    train_len = len(train)
    train = np.array(train)
    train_left = train[:, 0] 
    train_right = train[:, 1]
    T = np.ones((train_len, neg_K)) * (train_left.reshape((train_len, 1)))
    neg_left = T.reshape((train_len * neg_K,))
    T = np.ones((train_len, neg_K)) * (train_right.reshape((train_len, 1)))
    neg2_right = T.reshape((train_len * neg_K,))

    # ae 由于未预编码，得先训练30个epoch
    for epoch in range(30):
        if epoch == 0:
            neg2_left = np.random.choice(e, train_len * neg_K)
            neg_right = np.random.choice(e, train_len * neg_K)
        feed_dict_ae.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
        outs_ae = sess_ae.run([model_ae.opt_op, model_ae.loss, model_ae.outputs], feed_dict=feed_dict_ae)
        print("Epoch:", '%04d' % epoch, "AE_train_loss=", "{:.5f}".format(outs_ae[1]))
        neg_right = get_neg(train_left, all_ent2_list, train_right, outs_ae[2], neg_K)
        neg2_left = get_neg(train_right, all_ent1_list, train_left, outs_ae[2], neg_K)

    align_left = train_left
    align_right = train_right
    for epoch in range(epochs):
        # get outvec & get hits & bootstrap
        vecs_se = sess_se.run(output_layer_se)
        vecs_ae = sess_ae.run(model_ae.outputs, feed_dict=feed_dict_ae)

        if epoch in range(20, epochs, 5):
            get_hits(vecs_se, test)
            get_hits(vecs_ae, test)
            get_combine_hits(vecs_se, vecs_ae, Config.combine_loss_beta, test)
        
            # labeled_alignment, ents1, ents2 \
            #     = bootstrap(vecs_se, vecs_ae, test, labeled_alignment)
            #
            # if ents1 != []:
            #     align_left = np.append(train_left, np.array(ents1))
            #     align_right = np.append(train_right, np.array(ents2))

        # train se
        neg_right = get_neg(align_left, all_ent2_list, align_right, vecs_se, neg_K)
        neg2_left = get_neg(align_right, all_ent1_list, align_left, vecs_se, neg_K)
        feed_dict_se = {"neg_left:0": neg_left,
                        "neg_right:0": neg_right,
                        "neg2_left:0": neg2_left,
                        "neg2_right:0": neg2_right}
        _, l_s = sess_se.run([op_se, loss_se], feed_dict=feed_dict_se)
        last_k_loss_se.append(l_s)

        # train ae
        neg_right = get_neg(align_left, all_ent2_list, align_right, vecs_ae, neg_K)
        neg2_left = get_neg(align_right, all_ent1_list, align_left, vecs_ae, neg_K)
        feed_dict_ae.update({'neg_left:0': neg_left, 'neg_right:0': neg_right, 'neg2_left:0': neg2_left, 'neg2_right:0': neg2_right})
        _, l_a = sess_ae.run([model_ae.opt_op, model_ae.loss], feed_dict=feed_dict_ae)
        last_k_loss_ae.append(l_a)

        # print loss 
        print("Epoch:", '%04d' % epoch, "AE_train_loss=", "{:.5f}".format(l_a), "SE_train_loss=", "{:.5f}".format(l_s))

        # early_stopping
        if epoch > Config.early_stopping \
            and last_k_loss_ae[-1] > np.mean(last_k_loss_ae[-(Config.early_stopping + 1):-1])\
            and last_k_loss_se[-1] > np.mean(last_k_loss_se[-(Config.early_stopping + 1):-1]):
            print("Early stopping...")
            break
    
    vecs_se = sess_se.run(output_layer_se)
    vecs_ae = sess_ae.run(model_ae.outputs, feed_dict=feed_dict_ae) 
    get_hits(vecs_se, test)
    get_hits(vecs_ae, test)
    get_combine_hits(vecs_se, vecs_ae, Config.combine_loss_beta, test)
