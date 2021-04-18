import time
import itertools
import gc

import numpy as np
import igraph as ig
import networkx as nx
from include.Config import Config
import scipy

import functools
print = functools.partial(print, flush=True)

def bootstrapping(ents_embedding, ref_pairs, ref_ent1_list, ref_ent2_list, labeled_alignment):
    # sim_mat
    from sklearn.preprocessing import normalize
    if ents_embedding is None:
        return
    # 从ref_pairs中按对齐顺序取出实体的vec
    Lvec = np.array([ents_embedding[e1] for e1, e2 in ref_pairs])       # len(ref_pair) * 300
    Rvec = np.array([ents_embedding[e2] for e1, e2 in ref_pairs])
    # 实体相似度矩阵
    ref_sim_mat = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')      # len(ref_pair) * len(ref_pair)

    ref_sim_mat = 1.0 / np.exp(ref_sim_mat)
    # ref_sim_mat = np.matmul(Lvec, Rvec.transpose())
    # denom = np.linalg.norm(Lvec) * np.linalg.norm(Rvec)
    # ref_sim_mat = 0.5 + 0.5 * (ref_sim_mat / denom)
    # Lvec = normalize(Lvec, axis=1, norm='l2')
    # Rvec = normalize(Rvec, axis=1, norm='l2')
    # ref_sim_mat = np.matmul(Lvec, Rvec.transpose())


    th = Config.th
    n = ref_sim_mat.shape[0]
    # 这一轮找到的新的对齐实体
    print('>>>' + 'find_potential_alig')
    curr_labeled_alignment = find_potential_alignment(ref_sim_mat, th, Config.boot_K, n)
    print('<<<' + 'find_potential_alig_done')
    if curr_labeled_alignment is not None:
        print('>>>' + 'update_labeled_alignment')
        labeled_alignment = update_labeled_alignment(labeled_alignment, curr_labeled_alignment, ref_sim_mat, n)
        print('>>>' + 'update_labeled_alignment_label')
        labeled_alignment = update_labeled_alignment_label(labeled_alignment, ref_sim_mat, n)
        print('<<<' + 'update_labeled_alignment_label_done')
        del curr_labeled_alignment
    # labeled_alignment = curr_labeled_alignment
    if labeled_alignment is not None:
        ents1 = [ref_ent1_list[pair[0]] for pair in labeled_alignment]
        ents2 = [ref_ent2_list[pair[1]] for pair in labeled_alignment]
    else:
        ents1, ents2 = None, None
    del ref_sim_mat
    gc.collect()

    return labeled_alignment, ents1, ents2


def filter_mat(mat, threshold, greater=True, equal=False):
    if greater and equal:
        x, y = np.where(mat >= threshold)
    elif greater and not equal:
        x, y = np.where(mat > threshold)
    elif not greater and equal:
        x, y = np.where(mat <= threshold)
    else:
        x, y = np.where(mat < threshold)
    return set(zip(x, y))


def check_alignment(aligned_pairs, all_n, context="", is_cal=True):
    if aligned_pairs is None or len(aligned_pairs) == 0:
        print("{}, Empty aligned pairs".format(context))
        return
    num = 0
    for x, y in aligned_pairs:
        if x == y:
            num += 1
    print("{}, right alignment: {}/{}={:.3f}".format(context, num, len(aligned_pairs), num / len(aligned_pairs)))
    if is_cal:
        precision = round(num / len(aligned_pairs), 6)
        recall = round(num / all_n, 6)
        if recall > 1.0:
            recall = round(num / all_n, 6)
        f1 = 0
        if precision != 0 and recall != 0:
            f1 = round(2 * precision * recall / (precision + recall), 6)
        print("precision={}, recall={}, f1={}".format(precision, recall, f1))


def search_nearest_k(sim_mat, k):
    if k == 0:
        return None
    neighbors = set()
    ref_num = sim_mat.shape[0]
    for i in range(ref_num):
        # 前 k 大的下标
        rank = np.argpartition(-sim_mat[i, :], k)
        pairs = [j for j in itertools.product([i], rank[0:k])]
        neighbors |= set(pairs)
        # del rank
    assert len(neighbors) == ref_num * k
    return neighbors


def mwgm(pairs, sim_mat, func):
    return func(pairs, sim_mat)


def mwgm_graph_tool(pairs, sim_mat):
    from graph_tool.all import Graph, max_cardinality_matching
    if not isinstance(pairs, list):
        pairs = list(pairs)
    g = Graph()
    weight_map = g.new_edge_property("float")
    nodes_dict1 = dict()
    nodes_dict2 = dict()
    edges = list()
    for x, y in pairs:
        if x not in nodes_dict1.keys():
            n1 = g.add_vertex()
            nodes_dict1[x] = n1
        if y not in nodes_dict2.keys():
            n2 = g.add_vertex()
            nodes_dict2[y] = n2
        n1 = nodes_dict1.get(x)
        n2 = nodes_dict2.get(y)
        e = g.add_edge(n1, n2)
        edges.append(e)
        weight_map[g.edge(n1, n2)] = sim_mat[x, y]
    print("graph via graph_tool", g)
    res = max_cardinality_matching(g, heuristic=True, weight=weight_map, minimize=False, edges=True)
    # res = res.copy("double")
    # res.a = 2 * res.a + 2
    # graph_draw(g, edge_color=res, edge_pen_width=res, vertex_fill_color="grey",
    #            output="max_card_match.png")
    # exit()
    edge_index = np.where(res.get_array() == 1)[0].tolist()
    matched_pairs = set()
    for index in edge_index:
        matched_pairs.add(pairs[index])
    return matched_pairs


def mwgm_igraph(pairs, sim_mat):
    if not isinstance(pairs, list):
        pairs = list(pairs)
    index_id_dic1, index_id_dic2 = dict(), dict()
    index1 = set([pair[0] for pair in pairs])
    index2 = set([pair[1] for pair in pairs])
    for index in index1:
        index_id_dic1[index] = len(index_id_dic1)
    off = len(index_id_dic1)
    for index in index2:
        index_id_dic2[index] = len(index_id_dic2) + off
    assert len(index1) == len(index_id_dic1)
    assert len(index2) == len(index_id_dic2)
    edge_list = [(index_id_dic1[x], index_id_dic2[y]) for (x, y) in pairs]
    weight_list = [sim_mat[x, y] for (x, y) in pairs]
    leda_graph = ig.Graph(edge_list)
    leda_graph.vs["type"] = [0] * len(index1) + [1] * len(index2)
    leda_graph.es['weight'] = weight_list
    res = leda_graph.maximum_bipartite_matching(weights=leda_graph.es['weight'])
    print(res)
    selected_index = [e.index for e in res.edges()]
    matched_pairs = set()
    for index in selected_index:
        matched_pairs.add(pairs[index])
    return matched_pairs


def mwgm_networkx(pairs, sim_mat):
    def str_splice(prefix, index):
        return prefix + "_" + str(index)

    def remove_prefix(string):
        params = string.split('_')
        assert len(params) == 2
        return int(params[-1])

    prefix1 = 's'
    prefix2 = 't'
    graph = nx.Graph()
    for pair in pairs:
        graph.add_edge(str_splice(prefix1, pair[0]), str_splice(prefix2, pair[1]), weight=sim_mat[pair[0], pair[1]])
    edges = nx.max_weight_matching(graph, maxcardinality=False)
    matching_pairs = set()
    for v1, v2 in edges:
        if v1.startswith(prefix1):
            s = remove_prefix(v1)
            t = remove_prefix(v2)
        else:
            t = remove_prefix(v1)
            s = remove_prefix(v2)
        matching_pairs.add((s, t))
    return matching_pairs


def find_potential_alignment(sim_mat, sim_th, k, total_n):
    t = time.time()
    potential_aligned_pairs = generate_alignment(sim_mat, sim_th, k, total_n)
    if potential_aligned_pairs is None or len(potential_aligned_pairs) == 0:
        return None
    t1 = time.time()
    if Config.heuristic:
        selected_pairs = mwgm(potential_aligned_pairs, sim_mat, mwgm_graph_tool)
    else:
        selected_pairs = mwgm(potential_aligned_pairs, sim_mat, mwgm_igraph)
    check_alignment(selected_pairs, total_n, context="selected_pairs")
    del potential_aligned_pairs
    print("mwgm costs time: {:.3f} s".format(time.time() - t1))
    print("selecting potential alignment costs time: {:.3f} s".format(time.time() - t))
    return selected_pairs


def generate_alignment(sim_mat, sim_th, k, all_n):
    potential_aligned_pairs = filter_mat(sim_mat, sim_th)
    if len(potential_aligned_pairs) == 0:
        return None
    check_alignment(potential_aligned_pairs, all_n, context="after sim filtered")
    neighbors = search_nearest_k(sim_mat, k)
    # print("neighbors")
    # print(neighbors)
    # print("potential_aligned_pairs")
    # print(potential_aligned_pairs)
    if neighbors is not None:
        potential_aligned_pairs &= neighbors
        if len(potential_aligned_pairs) == 0:
            return None
        check_alignment(potential_aligned_pairs, all_n, context="after sim and neighbours filtered")
    del neighbors
    print(len(potential_aligned_pairs))
    return potential_aligned_pairs


def edit_alignment(alignment, prev_sim_mat, sim_mat, all_n):
    t = time.time()
    away_pairs = set()
    for i, j in alignment:
        if prev_sim_mat[i, j] > sim_mat[i, j]:
            away_pairs.add((i, j))
    check_alignment(away_pairs, all_n, "away pairs in selected pairs")
    edited_pairs = alignment - away_pairs
    check_alignment(edited_pairs, all_n, "after editing")
    print("editing costs time: {:.3f} s".format(time.time() - t))
    return alignment


def update_labeled_alignment(labeled_alignment, curr_labeled_alignment, sim_mat, all_n):
    # all_alignment = labeled_alignment | curr_labeled_alignment
    # check_alignment(labeled_alignment, all_n, context="before updating labeled alignment")
    # kb1中实体指向kb2中不同实体（多轮差异）
    labeled_alignment_dict = dict(labeled_alignment)
    n, n1 = 0, 0
    for i, j in curr_labeled_alignment:
        # 这一轮和上一轮打的标签不一致的数量
        if labeled_alignment_dict.get(i, -1) == i and j != i:
            n1 += 1
        if i in labeled_alignment_dict.keys():
            jj = labeled_alignment_dict.get(i)
            old_sim = sim_mat[i, jj]
            new_sim = sim_mat[i, j]
            if new_sim >= old_sim:
                # 原对齐为真实对齐，现对其为错误对齐的数目
                if jj == i and j != i:
                    n += 1
                labeled_alignment_dict[i] = j
        else:
            labeled_alignment_dict[i] = j
    print("update wrongly: ", n, "greedy update wrongly: ", n1)
    labeled_alignment = set(zip(labeled_alignment_dict.keys(), labeled_alignment_dict.values()))
    check_alignment(labeled_alignment, all_n, context="after editing labeled alignment (<-)")
    # selected_pairs = mwgm(all_alignment, sim_mat, mwgm_igraph)
    # check_alignment(selected_pairs, context="after updating labeled alignment with mwgm")
    return labeled_alignment


def update_labeled_alignment_label(labeled_alignment, sim_mat, all_n):
    # check_alignment(labeled_alignment, all_n, context="before updating labeled alignment label")
    labeled_alignment_dict = dict()
    updated_alignment = set()
    # kb1中多实体对齐kb2中一实体，择优
    for i, j in labeled_alignment:
        ents_j = labeled_alignment_dict.get(j, set())
        ents_j.add(i)
        labeled_alignment_dict[j] = ents_j
    for j, ents_j in labeled_alignment_dict.items():
        if len(ents_j) == 1:
            for i in ents_j:
                updated_alignment.add((i, j))
        else:
            max_i = -1
            max_sim = -10
            for i in ents_j:
                if sim_mat[i, j] > max_sim:
                    max_sim = sim_mat[i, j]
                    max_i = i
            updated_alignment.add((max_i, j))
    check_alignment(updated_alignment, all_n, context="after editing labeled alignment (->)")
    return updated_alignment


