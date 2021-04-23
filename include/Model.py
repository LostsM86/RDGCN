import math
from .Init import *
from include.Test import *
from include.bootstrap import *
import scipy
import json
import os
from include.Models_AE import *
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import functools
print = functools.partial(print, flush=True)


def rfunc(KG, e):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])
    r_num = len(head)
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    r_mat_ind = []
    r_mat_val = []
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1
        r_mat_ind.append([tri[0], tri[2]])
        r_mat_val.append(tri[1])
    r_mat = tf.SparseTensor(
        indices=r_mat_ind, values=r_mat_val, dense_shape=[e, e])

    return head, tail, head_r, tail_r, r_mat


def idf_func(kg1, kg2, r_num):
    r2idf = {}
    l1 = len(kg1)
    t1 = {}
    for tri in kg1:
        if tri[1] not in t1:
            t1[tri[1]] = 1
        else:
            t1[tri[1]] += 1
    for r in t1:
        r2idf[r] = math.log(l1 / (1 + t1[r]), 10)
    l2 = len(kg1)
    t2 = {}
    for tri in kg2:
        if tri[1] not in t2:
            t2[tri[1]] = 1
        else:
            t2[tri[1]] += 1
    for r in t2:
        r2idf[r] = math.log(l2 / (1 + t2[r]), 10)

    min_idf = min(r2idf.values())
    max_idf = max(r2idf.values())

    # for r in r2idf:
    #     r2idf[r] = (r2idf[r] - min_idf) / (max_idf - min_idf)
    r2_idf_list = []
    for i in range(r_num):
        if i in r2idf.keys():
            r2_idf_list.append(r2idf[r])
        else:
            r2_idf_list.append(3)
    return r2_idf_list

def get_mat(e, KG):
    # 出入度矩阵: [{0, 11}, {2, 3}, ...]  包括自己到自己的度，1<->2的度
    du = [{e_id} for e_id in range(e)]
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]].add(tri[2])
            du[tri[2]].add(tri[0])
    du = [len(d) for d in du]
    M = {}
    # M是双向图+自己到自己
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    for i in range(e):
        M[(i, i)] = 1
    return M, du


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG, KG1, KG2):
    print('getting a sparse tensor...')
    M, du = get_mat(e, KG)
    # r2idf = idf_func(KG1, KG2)
    ind = []
    val = []
    M_arr = np.zeros((e, e))
    for fir, sec in M:
        ind.append((sec, fir))
        val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
        # ind.append((fir, sec))
        # val.append(M[(fir, sec)] / ((math.sqrt(du[fir]) + math.sqrt(du[sec])) * r2idf[(fir, sec)]))
        M_arr[fir][sec] = 1.0
    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])

    # M_arr: primal graph 邻接矩阵，1<->2, 1<->1
    # M: dual graph的稀疏矩阵
    return M, M_arr


# add a layer
def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a diag layer...')
    w0 = init([1, dimension])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def add_full_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a full layer...')
    w0 = init([dimension_in, dimension_out])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.matmul(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def add_sparse_att_layer(inlayer, dual_layer, r_mat, act_func):
    dual_transform = tf.reshape(tf.layers.conv1d(
        tf.expand_dims(dual_layer, 0), 1, 1), (-1, 1))
    logits = tf.reshape(tf.nn.embedding_lookup(
        dual_transform, r_mat.values), [-1])
    print('adding sparse attention layer...')
    lrelu = tf.SparseTensor(indices=r_mat.indices,
                            values=tf.nn.leaky_relu(logits),
                            dense_shape=(r_mat.dense_shape))
    coefs = tf.sparse_softmax(lrelu)
    vals = tf.sparse_tensor_dense_matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def add_dual_att_layer(inlayer, inlayer2, adj_mat, act_func, hid_dim):
    in_fts = tf.layers.conv1d(tf.expand_dims(inlayer2, 0), hid_dim, 1)
    f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    logits = f_1 + tf.transpose(f_2)
    print('adding dual attention layer...')
    adj_tensor = tf.constant(adj_mat, dtype=tf.float32)
    bias_mat = -1e9 * (1.0 - (adj_mat > 0))
    logits = tf.multiply(adj_tensor, logits)
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

    vals = tf.matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def add_self_att_layer(inlayer, adj_mat, act_func, hid_dim):
    in_fts = tf.layers.conv1d(tf.expand_dims(
        inlayer, 0), hid_dim, 1, use_bias=False)        # 1 * r_num * 2dim
    f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    logits = f_1 + tf.transpose(f_2)
    print('adding self attention layer...')
    adj_tensor = tf.constant(adj_mat, dtype=tf.float32)
    logits = tf.multiply(adj_tensor, logits)
    bias_mat = -1e9 * (1.0 - (adj_mat > 0))
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

    vals = tf.matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def highway(layer1, layer2, dimension):
    kernel_gate = glorot([dimension, dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate * layer2 + carry_gate * layer1


def compute_r(inlayer, head_r, tail_r, idf):
    # 修改 加上idf
    # idf_t = tf.constant(idf, shape=[len(idf), 1], dtype=tf.float32)
    # head_l = tf.constant(head_r, dtype=tf.float32)
    # head_l = tf.matmul(head_l, idf_t)
    # head_l = tf.transpose(head_l)
    # tail_l = tf.constant(tail_r, dtype=tf.float32)
    # tail_l = tf.matmul(tail_l, idf_t)
    # tail_l = tf.transpose(tail_l)
    # L = tf.matmul(head_l, inlayer) / \
    #     tf.expand_dims(tf.reduce_sum(idf_t, axis=-1), -1)
    # R = tf.matmul(tail_l, inlayer) / \
    #     tf.expand_dims(tf.reduce_sum(idf_t, axis=-1), -1)

    head_l = tf.transpose(tf.constant(head_r, dtype=tf.float32))
    tail_l = tf.transpose(tf.constant(tail_r, dtype=tf.float32))
    L = tf.matmul(head_l, inlayer) / \
        tf.expand_dims(tf.reduce_sum(head_l, axis=-1), -1)
    R = tf.matmul(tail_l, inlayer) / \
        tf.expand_dims(tf.reduce_sum(tail_l, axis=-1), -1)



    r_embeddings = tf.concat([L, R], axis=-1)
    return r_embeddings


def get_dual_input(inlayer, head, tail, head_r, tail_r, idf):
    dual_X = compute_r(inlayer, head_r, tail_r, idf)
    print('computing the dual input...')
    count_r = len(head)
    dual_A = np.zeros((count_r, count_r))
    for i in range(count_r):
        for j in range(count_r):
            a_h = len(head[i] & head[j]) / len(head[i] | head[j])
            a_t = len(tail[i] & tail[j]) / len(tail[i] | tail[j])
            dual_A[i][j] = a_h + a_t
    return dual_X, dual_A


def get_input_layer(e, dimension, lang):
    print('adding the primal input layer...')
    with open(file='data/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    # TODO(zyj) 原下标 != id
    input_embeddings = tf.convert_to_tensor(embedding_list)
    ent_embeddings = tf.Variable(input_embeddings)
    return tf.nn.l2_normalize(ent_embeddings, 1)


def get_loss(outlayer, gamma, k):
    print('getting loss...')
    left = tf.placeholder(tf.int32, [None], "ILL_left")
    right = tf.placeholder(tf.int32, [None], "ILL_right")
    # left = ILL[:, 0]
    # right = ILL[:, 1]
    t = tf.shape(left)[0]
    # t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    p_loss = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = tf.placeholder(tf.int32, [None], "neg_left")
    neg_right = tf.placeholder(tf.int32, [None], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    n_loss = tf.reshape(tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1), [t, k])
    A = p_loss + gamma
    L1 = tf.nn.relu(tf.add(-n_loss, tf.reshape(A, [t, 1])))

    neg_left = tf.placeholder(tf.int32, [None], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [None], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    n_loss = tf.reshape(tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1), [t, k])
    L2 = tf.nn.relu(tf.add(-n_loss, tf.reshape(A, [t, 1])))

    t = tf.cast(t, tf.float32)
    return tf.div(tf.reduce_sum(L1) + tf.reduce_sum(L2), (2.0 * k * t))


def build(learning_rate, dimension, act_func, alpha, beta, gamma, k, lang, e, KG1, KG2):
    graph = tf.Graph()
    with graph.as_default():
        KG = KG1 + KG2
        # primal_X_0: 文件里读到的ent_embedding
        primal_X_0 = get_input_layer(e, dimension, lang)
        # M: dual graph的稀疏矩阵, 自环+双向图
        M, M_arr = get_sparse_tensor(e, KG, KG1, KG2)
        # head、tail: 关系r的头、尾集合
        # head_r、tail_r: 实体-关系矩阵，存在关系=1
        # r_mat: indice:(e, t) value(r) 稀疏矩阵，保存完整triple
        head, tail, head_r, tail_r, r_mat = rfunc(KG, e)
        idf = idf_func(KG1, KG2, len(head))

        print('first interaction...')
        dual_X_1, dual_A_1 = get_dual_input(
            primal_X_0, head, tail, head_r, tail_r, idf)
        dual_H_1 = add_self_att_layer(dual_X_1, dual_A_1, tf.nn.relu, 600)
        primal_H_1 = add_sparse_att_layer(
            primal_X_0, dual_H_1, r_mat, tf.nn.relu)
        primal_X_1 = primal_X_0 + alpha * primal_H_1

        print('second interaction...')
        dual_X_2, dual_A_2 = get_dual_input(
            primal_X_1, head, tail, head_r, tail_r, idf)
        dual_H_2 = add_dual_att_layer(
            dual_H_1, dual_X_2, dual_A_2, tf.nn.relu, 600)
        primal_H_2 = add_sparse_att_layer(
            primal_X_1, dual_H_2, r_mat, tf.nn.relu)
        primal_X_2 = primal_X_0 + beta * primal_H_2

        print('gcn layers...')
        gcn_layer_1 = add_diag_layer(
            primal_X_2, dimension, M, act_func, dropout=0.0)
        gcn_layer_1 = highway(primal_X_2, gcn_layer_1, dimension)
        gcn_layer_2 = add_diag_layer(
            gcn_layer_1, dimension, M, act_func, dropout=0.0)
        output_layer = highway(gcn_layer_1, gcn_layer_2, dimension)
        loss = get_loss(output_layer, gamma, k)

        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        init = tf.global_variables_initializer()
        sess = tf.Session(graph=graph)
        sess.run(init)
    return output_layer, loss, train_step, sess


# get negative samples
def get_neg(ILL, cand_ent_list, other_ILL, output_layer, k):
    # ILL_R, all_ent1_list,  ILL_L, out, k)
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    cand_ent_vec = np.array([output_layer[e2] for e2 in cand_ent_list])
    sim = scipy.spatial.distance.cdist(ILL_vec, cand_ent_vec, metric='cityblock')
    # KG_vec = np.array(output_layer)
    # sim = scipy.spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
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


def training(output_layer_se, loss_se, train_step_se, sess_se,
             model_ae,
             epochs, k, ILL, test, all_ent1_list, all_ent2_list, ref_ent1_list, ref_ent2_list):
    print('running...')
    cost_val = []
    ILL = np.array(ILL)
    ILL_L_BASE = ILL[:, 0]
    ILL_R_BASE = ILL[:, 1]
    ILL_L = np.array(ILL_L_BASE)
    ILL_R = np.array(ILL_R_BASE)
    labeled_alignment = set()
    ents1 = []
    ents2 = []


    # ae_batch_inputs = []
    # ae_batch_labels = []
    # range_types = []
    # for j in range(len(model_ae.data)):
    #     ae_batch_inputs[j] = model_ae.data[j][0]
    #     ae_batch_labels[j, 0] = model_ae.data[j][1]
    #     range_types[j, 0] = get_range_weight(model_ae.range_vec, ae_batch_inputs[j], ae_batch_labels[j, 0])

    for i in range(epochs):
        print(i)
        # if i == 0:
        #     se_outvec = sess_se.run(output_layer_se)
        #     print(len(se_outvec))
        #     exit()
        #     # ae_outvec = sess_ae.run(model_ae.embeddings)
        #
        # # bootstrap
        # # if i in range(10, epochs, 5):
        # #     se_outvec = sess_se.run(output_layer_se, feed_dict=feeddict)
        # #     get_hits(se_outvec, test)
        # #     print('>>>' + 'bootstraping')
        # #     labeled_alignment, ents1, ents2 \
        # #         = bootstrapping(se_outvec, test, ref_ent1_list, ref_ent2_list, labeled_alignment)
        # #     print(len(ents1))
        # #     print('<<<' + 'bootstraping_done')
        # #     if ents1 != []:
        # #         ILL_L = np.append(ILL_L_BASE, np.array(ents1))
        # #         ILL_R = np.append(ILL_R_BASE, np.array(ents2))
        #
        # # se 1 epoch
        # t = ILL_R.shape[0]
        # L = np.ones((t, k), dtype=int) * ILL_L.reshape((t, 1))
        # neg_left = L.reshape((t * k,))
        # L = np.ones((t, k), dtype=int) * ILL_R.reshape((t, 1))
        # neg2_right = L.reshape((t * k,))
        # neg_right = get_neg(ILL_L, all_ent2_list, ILL_R, se_outvec, k)
        # neg2_left = get_neg(ILL_R, all_ent1_list, ILL_L, se_outvec, k)
        # feeddict = {"neg_left:0": neg_left,
        #             "neg_right:0": neg_right,
        #             "neg2_left:0": neg2_left,
        #             "neg2_right:0": neg2_right,
        #             "ILL_left:0": ILL_L,
        #             "ILL_right:0": ILL_R}
        # _, ls = sess_se.run([train_step_se, loss_se], feed_dict=feeddict)
        # cost_val.append(ls)

        # ae 1 epoch
        # print(math.ceil(len(model_ae.data) / batch_size))
        average_loss = 0
        for step in range(math.ceil(len(model_ae.data) / batch_size)):
            dic = {}
            dic['train_inputs'], dic['train_labels'], dic['range_vecs'] \
                = generate_batch_random(model_ae.data, batch_size, model_ae.range_vec)
            watch, _, loss_ae = model_ae.mini_act(dic)
            average_loss += loss_ae

            attr_embeddings = model_ae.act()
            model_ae.ent2vec(attr_embeddings)
            ae_outvec = model_ae.ent_embedding_list
            print(max(max(row) for row in ae_outvec))
            print(min(min(row) for row in ae_outvec))
            model_ae.get_sim_mat()
            print("AE")
            get_hits_ae(model_ae.sim_mat, test)

            if step % 1000 == 0:
                if step > 0:
                    average_loss /= 1000
                    print("average loss at step", step, ":", average_loss)
        # average_loss /= math.ceil(len(model_ae.data) / batch_size) * batch_size

        # print loss
        # print('*****', '%d/%d' % (i, epochs), 'epochs --- ', 'AE_loss: ', "SE_train_loss=", "{:.5f}".format(ls), "AE_train_loss=",
        #       "{:.5f}".format(average_loss))

        # print hits
        if i in range(0, epochs):
            # se_outvec = sess_se.run(output_layer_se)
            # print("SE")
            # get_hits(se_outvec, test)

            attr_embeddings = model_ae.act()
            model_ae.ent2vec(attr_embeddings)
            ae_outvec = model_ae.ent_embedding_list
            print(max(max(row) for row in ae_outvec))
            print(min(min(row) for row in ae_outvec))
            model_ae.get_sim_mat()
            print("AE")
            get_hits_ae(model_ae.sim_mat, test)

            # print("SE+AE")
            # get_combine_hits(se_outvec, model_ae.sim_mat, test)

        # early_stoping
        if i > Config.early_stopping \
                and cost_val[-1] > np.mean(cost_val[-(Config.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    se_outvec = sess_se.run(output_layer_se)
    model_ae.ent2vec()
    ae_outvec = model_ae.ent_embedding_list
    print("SE")
    get_hits(se_outvec, test)
    print("SE+AE")
    get_combine_hits(se_outvec, ae_outvec, Config.loss_beta, test)
    sess_se.close()
    model_ae.sess.close()
    return se_outvec, cost_val
