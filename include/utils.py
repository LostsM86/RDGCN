import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import math
import time
import os
from include.Config import Config

import functools
print = functools.partial(print, flush=True)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, placeholders):
    """Construct feed dictionary for GCN-Align."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def loadattr(fns, e, ent2id):
    """The most frequent attributes are selected to save space."""
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
                    # 数据没有值
                    # {[k1, m], [k2, n], ...}

    fre = [ (k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True) ]
    # cnt 从大到小排序 [[k, cnt[k]], , ]

    num_features = min(len(fre), 2000)
    attr2id = {}
    for i in range(num_features):
        attr2id[fre[i][0]] = i      # attr2id[ki] = i
    M = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0
                            # {([ei, ai], 1), }
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[0])
        col.append(key[1])
        data.append(M[key])
    # 实体id - 属性id 的稀疏矩阵
    return sp.coo_matrix((data, (row, col)), shape=(e, num_features)) # attr


def get_dic_list(e, KG):
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        M[(tri[0], tri[2])] = 1
        M[(tri[2], tri[0])] = 1
    dic_list = {}
    for i in range(e):
        dic_list[i] = []
    for pair in M:
        dic_list[pair[0]].append(pair[1])
    return dic_list


def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def idf_func(kg1, kg2):
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

    for r in r2idf:
        r2idf[r] = (r2idf[r] - min_idf) / (max_idf - min_idf)
    return r2idf


def get_weighted_adj(e, kg1, kg2):
    KG = kg1 + kg2
    r2f = func(KG)
    r2if = ifunc(KG)
    r2idf = idf_func(kg1, kg2)
    M = {}
    for tri in KG:
        # todo(zyj) 转成自环图
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]] * r2idf[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]] * r2idf[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]] * r2idf[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]] * r2idf[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def get_ae_input(attr):
    return sparse_to_tuple(sp.coo_matrix(attr))


def get_ent_list(pair):
    nppair = np.array([list(i) for i in pair])
    return nppair[:, 0], nppair[:, 1]


def get_all_attr(attr, sup_pairs):
    all_attr = attr.todense()
    one_list = [1.0 for i in range(all_attr.shape[1])]
    for e1, e2 in sup_pairs:
        all_attr[e1] = list(np.minimum(all_attr[e1] + all_attr[e2], one_list))
        all_attr[e2] = all_attr[e1]
    return sp.coo_matrix(all_attr)


def get_all_kg(kg1, kg2, train):
    left = train[:, 0]
    right = train[:, 1]
    all_kg1 = kg1
    all_kg2 = kg2
    for tri in kg1:
        if tri[0] in left:
            h = right[np.argwhere(left == tri[0])][0][0]
            all_kg1.append((h, tri[1], tri[2]))
        if tri[2] in left:
            t = right[np.argwhere(left == tri[2])][0][0]
            all_kg1.append((tri[0], tri[1], t))
        if tri[0] in left and tri[2] in left:
            all_kg1.append((h, tri[1], t))

    for tri in kg2:
        if tri[0] in right:
            h = left[np.argwhere(right == tri[0])][0][0]
            all_kg2.append((h, tri[1], tri[2]))
        if tri[2] in right:
            t = left[np.argwhere(right == tri[2])][0][0]
            all_kg2.append((tri[0], tri[1], t))
        if tri[0] in right and tri[2] in right:
            all_kg2.append((h, tri[1], t))
    return all_kg1, all_kg2


def load_ae_data(dataset_str, e, KG1, KG2, train):
    As = ['data/' + dataset_str + '/' + 'training_attrs_1', 'data/' + dataset_str + '/' + 'training_attrs_2']

    Es = ['data/' + dataset_str + '/' + 'ent_ids_1', 'data/' + dataset_str + '/' + 'ent_ids_2']

    ent2id = get_ent2id(Es)

    # 属性三元组 = 实体=>属性的稀疏矩阵
    attr = loadattr(As, e, ent2id)
    all_attr = get_all_attr(attr, train)

    # 拿 np 转了一下 ？
    # ae_input = get_ae_input(attr)
    ae_input = get_ae_input(all_attr)

    # 计算关系的邻接矩阵
    adj = get_weighted_adj(e, KG1, KG2)
    all_KG1, all_KG2 = get_all_kg(KG1, KG2, train)
    # adj = get_weighted_adj(e, all_KG1, all_KG2)

    support = [preprocess_adj(adj)]

    num_supports = 1

    return ae_input, support, num_supports, all_attr, all_KG1, all_KG2


def get_new_all_ae_data(e, base_attr, base_KG, left, right):
    train = []
    for i in range(len(left)):
        train.append(tuple([left[i], right[i]]))

    all_attr = get_all_attr(base_attr, train)
    ae_input = get_ae_input(all_attr)

    all_KG1, all_KG2 = get_all_kg(base_KG[0], base_KG[1], np.array(train))
    adj = get_weighted_adj(e, all_KG1, all_KG2)
    support = [preprocess_adj(adj)]

    return ae_input, support


def generate_new_triples(kg1, kg2, ref_ent_ids):
    print("generating new triples...")
    new_kg1_triples = []
    new_kg2_triples = []
    
    ref_12_dict = {}
    ref_21_dict = {}
    for ent1, ent2 in ref_ent_ids:
        ref_12_dict[ent1] = ent2
        ref_21_dict[ent2] = ent1
    
    for tri1 in kg1:
        if tri1[0] in ref_12_dict:
            new_kg1_triples.append([ref_12_dict[tri1[0]], tri1[1], tri1[2]])
        if tri1[2] in ref_12_dict:
            new_kg1_triples.append([tri1[0], tri1[1], ref_12_dict[tri1[2]]])
        if tri1[0] in ref_12_dict and tri1[2] in ref_12_dict:
            new_kg1_triples.append([ref_12_dict[tri1[0]], tri1[1], ref_12_dict[tri1[2]]])
    for tri2 in kg2:
        if tri2[0] in ref_21_dict:
            new_kg2_triples.append([ref_21_dict[tri2[0]], tri2[1], tri2[2]])
        if tri2[2] in ref_21_dict:
            new_kg2_triples.append([tri2[0], tri2[1], ref_21_dict[tri2[2]]])
        if tri2[0] in ref_21_dict and tri2[2] in ref_21_dict:
            new_kg2_triples.append([ref_21_dict[tri2[0]], tri2[1], ref_21_dict[tri2[2]]])
    print("kb1: " + str(len(kg1)) + "---+" + str(len(new_kg1_triples)))
    print("kb2: " + str(len(kg2)) + "---+" + str(len(new_kg2_triples))) 
    return new_kg1_triples, new_kg2_triples


def generate_negative(entity_dim_matrix, train_L, train_R, neg_nums_per_entity):
    l_edm = np.array([entity_dim_matrix[e1] for e1 in train_L])
    r_edm = np.array([entity_dim_matrix[e2] for e2 in train_R])
    sim = scipy.spatial.distance.cdist(l_edm, r_edm, metric = 'cityblock')
    neg_right = []
    for i in range(l_edm.shape[0]):
        rank = sim[i, :].argsort()
        l = 0
        for j in rank:
            if i == j:
                continue
            if l < neg_nums_per_entity:
                neg_right.append(train_R[j])
                l = l + 1
            else:
                break
    
    neg_left = []
    for i in range(r_edm.shape[0]):
        rank = sim[:, i].argsort()
        l = 0
        for j in rank:
            if i == j:
                continue
            if l < neg_nums_per_entity:
                neg_left.append(train_L[j])
                l = l + 1
            else:
                break
    return np.array(neg_right), np.array(neg_left)

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


def store_vecs(vecs_se, vecs_ae):
    print('#### writing vec file...')
    time_str = time.strftime("%m-%d-%H-%M", time.localtime())
    path = './data/' + Config.language + '/' + time_str
    if not os.path.exists(path):
        os.mkdir(path)
    fw = open('./data/' + Config.language + '/' + time_str + '/se_vec.txt', 'w')
    for vec in vecs_se:
        line = ''
        for v in vec:
            line += str(v) + ' '
        line += '\n'
        fw.write(line)
    fw.close()
    fw = open('./data/' + Config.language + '/' + time_str + '/ae_vec.txt', 'w')
    for vec in vecs_ae:
        line = ''
        for v in vec:
            line += str(v) + ' '
        line += '\n'
        fw.write(line)
    fw.close()
