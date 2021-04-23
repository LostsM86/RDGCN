import numpy as np
import scipy

import functools
print = functools.partial(print, flush=True)

def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))


# def get_combine_hits(se_vec, ae_vec, beta, test_pair, top_k=(1, 10, 50, 100)):
#     vec = np.concatenate([np.array(se_vec)*beta, np.array(ae_vec)*(1.0-beta)], axis=1)
#     get_hits(vec, test_pair, top_k)

def get_combine_hits(se_vec, ae_sim_mat, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([se_vec[e1] for e1, e2 in test_pair])
    Rvec = np.array([se_vec[e2] for e1, e2 in test_pair])
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    sim = sim * ae_sim_mat
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))


def get_hits_ae(sim_mat, test_pair, top_k=(1, 10, 50, 100)):
    for i in range(sim_mat.shape[0]):
        # rank = np.argpartition(-sim_mat[i, :], 100)
        rank = sim_mat[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        top_lr = [0] * len(top_k)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    print('For each left:')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
