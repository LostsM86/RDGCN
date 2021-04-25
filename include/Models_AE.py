import numpy as np
import math
from include.Utils_AE import *
import itertools
import collections
import random
from sklearn import preprocessing
import scipy


data_frequent_p = 0.95
batch_size = 2000
num_sampled_negs = 1000
num_train = 100
min_frequency = 5  # small data 5
min_props = 2
lr = 0.1
v = 2

embedding_size = 300


class CorrelationNN(object):
    def __init__(self, e):
        # 读取数据集
        self.init_width = 1.0 / math.sqrt(embedding_size)
        # self.prop_vec_file = rel_train_data_folder + 'attrs_vec'
        # self.prop_embeddings_file = rel_train_data_folder + 'attrs_embeddings'
        # self.meta_out_file = rel_train_data_folder + 'attrs_meta'

        # 变量声明
        self.e = e

        self.watch = {}

        # data load
        # self.base_data
        # self.data
        # self.sup_ents_pairs       # base训练实体对
        # self.new_ref_pairs       # 新增实体对
        # self.kb1_ids      # dict {(id, url), ...}
        # self.kb2_ids      # dict {(id, url), ...}
        # self.attrs_with_ids       # dict {(url,set(props_indx), ...}
        # self.range_vec        # list
        # self.reverse_props_ids        # {(prop:id), ...}
        # self.common_props_ids        # {(id:prop), ...}
        # self.attrs1       # training_attrs_1 + training_attrs_2
        # self.ent_vec_dict = {}        # ent 的属性list
        # self.reverse_dict     # ent:id
        # self.related_ents_dict1       # {h:(t1,t2,...),...}
        # self.related_ents_dict2       # {h:(t1,t2,...),...}

    def build(self):
        graph = tf.Graph()
        with graph.as_default():
            self.train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            self.range_vecs = tf.placeholder(tf.float32, shape=[batch_size, 1])

            with tf.variable_scope('embedding'):
                self.embeddings = tf.Variable(tf.random_uniform([self.props_size, embedding_size], -self.init_width, self.init_width))
                self.embeddings = tf.nn.l2_normalize(self.embeddings, 1)
                nce_weights = tf.Variable(tf.truncated_normal([self.props_size, embedding_size], stddev=self.init_width))
                nce_biases = tf.Variable(tf.zeros([self.props_size]))

            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            self.loss = nce_loss(nce_weights, nce_biases, self.train_labels, embed, num_sampled_negs, self.props_size, v=self.range_vecs)
            self.optimizer = tf.train.AdagradOptimizer(lr).minimize(self.loss)
            init = tf.global_variables_initializer()

            self.sess = tf.Session(graph=graph)
            self.sess.run(init)

    def mini_act(self, dic):
        feed_dict = {
            self.train_inputs: dic['train_inputs'],
            self.train_labels: dic['train_labels'],
            self.range_vecs: dic['range_vecs'],
        }

        output_feeds = [self.watch, self.optimizer, self.loss]

        results = self.sess.run(output_feeds, feed_dict)
        return results

    def act(self):
        results = self.sess.run(self.embeddings)
        return results

    def load_base_data(self, lang, train, ref_ent1_list, ref_ent2_list, related_ents_dict1, related_ents_dict2):
        attr_folder = './data/' + lang + '/'
        rel_train_data_folder = './data/' + lang + '/'
        attr_range_file = './data/' + lang + '/all_attrs_range'
        en_attr_range_file = './data/en_all_attrs_range'

        self.sup_ents_pairs = train
        self.all_id_list1, self.all_id_list2 = read_kg_ent_lists(attr_folder)
        self.ref_ent1_list = ref_ent1_list
        self.ref_ent2_list = ref_ent2_list
        self.related_ents_dict1 = related_ents_dict1
        self.related_ents_dict2 = related_ents_dict2
        # self.sup_ents_pairs = read_pair_ids(rel_train_data_folder + 'ref_ent_ids')      # 直接传入
        self.id_ent_dict1, self.ent_id_dict1, _, _ = read_ids(rel_train_data_folder + 'ent_ids_1')
        self.id_ent_dict2, self.ent_id_dict2, _, _ = read_ids(rel_train_data_folder + 'ent_ids_2')
        self.ent_id_dict = {**self.ent_id_dict1, **self.ent_id_dict2}
        props_set = set()
        props_list = []
        self.base_data = []

        # attrs data
        ent_attrs_dict = read_attrs(attr_folder + 'training_attrs_1')
        temp_ent_attrs = read_attrs(attr_folder + 'training_attrs_2')
        ent_attrs_dict = merge_dicts(ent_attrs_dict, temp_ent_attrs)
        for uri in ent_attrs_dict.keys():
            props_list.extend(list(ent_attrs_dict.get(uri)))
            props_set |= ent_attrs_dict.get(uri)
        del temp_ent_attrs

        # 去掉太高频和太低频的prop
        common_props2ids_dict = get_common(props_list, props_set)
        self.props_size = len(common_props2ids_dict)
        common_ids2props_dict = dict(zip(common_props2ids_dict.values(), common_props2ids_dict.keys()))

        self.ent_attrids_dict = dict()
        # prop_totals = []
        for uri in ent_attrs_dict.keys():
            props = ent_attrs_dict.get(uri)
            props_idxs = []
            for p in props:
                if p in common_props2ids_dict:
                    index_p = common_props2ids_dict[p]
                    props_idxs.append(index_p)
            if len(props_idxs) >= min_props:
                # prop_totals.append(len(props_idxs))
                self.ent_attrids_dict[uri] = set(props_idxs)
                # C(prop_idxs, 2) 某实体的其中两个属性的组合
                for p_id, context_p in itertools.combinations(props_idxs, 2):
                    if p_id != context_p:
                        self.base_data.append((p_id, context_p))

        # self.get_new_data(sup_ents_pairs)
        new_ref_pairs_dict = pair_2dict(self.sup_ents_pairs)
        for ent in new_ref_pairs_dict:
            kb2_ent = new_ref_pairs_dict.get(ent)
            ent_uri = self.id_ent_dict1[ent]
            kb2_ent_uri = self.id_ent_dict2[kb2_ent]
            if ent_uri in self.ent_attrids_dict and kb2_ent_uri in self.ent_attrids_dict:
                ent_props = self.ent_attrids_dict.get(ent_uri)
                ent2_props = self.ent_attrids_dict.get(kb2_ent_uri)
                # sup_ents_dict对应实体对的两个实体属性集合的笛卡尔乘积
                for p_id, context_p in itertools.product(ent_props, ent2_props):
                    if p_id != context_p:
                        # TODO(zyj) 这里在干嘛？
                        for i in range(10):
                            self.base_data.append((p_id, context_p))
                            self.base_data.append((context_p, p_id))
        self.data = self.base_data

        # range_vec
        # TODO(zyj) range_vec的顺序存疑
        range_dict = read_attrs_range(attr_range_file)
        en_range_dict = read_attrs_range(en_attr_range_file)
        self.range_vec = []
        for i in range(len(common_props2ids_dict)):
            assert i in common_ids2props_dict
            attr_uri = common_ids2props_dict[i]
            if attr_uri in range_dict:
                self.range_vec.append(range_dict.get(attr_uri))
            elif attr_uri in en_range_dict:
                self.range_vec.append(en_range_dict.get(attr_uri))
            else:
                self.range_vec.append(0)

    def get_new_data(self, new_ref_pairs):
        self.new_ref_pairs = new_ref_pairs
        new_ref_pairs_dict = pair_2dict(new_ref_pairs)
        self.data = self.base_data
        for ent in new_ref_pairs_dict:
            kb2_ent = new_ref_pairs_dict.get(ent)
            ent_uri = self.id_ent_dict1[ent]
            kb2_ent_uri = self.id_ent_dict2[kb2_ent]
            if ent_uri in self.ent_attrids_dict and kb2_ent_uri in self.ent_attrids_dict:
                ent_props = self.ent_attrids_dict.get(ent_uri)
                ent2_props = self.ent_attrids_dict.get(kb2_ent_uri)
                #sup_ents_dict对应实体对的两个实体属性集合的笛卡尔乘积
                for p_id, context_p in itertools.product(ent_props, ent2_props):
                    if p_id != context_p:
                        for i in range(10):
                            self.data.append((p_id, context_p))
                            self.data.append((context_p, p_id))

    def ent2vec(self, attr_embeddings):
        # self.ent_embedding = [[0.0]*embedding_size for i in range(self.e)]
        # for ent in self.ent_vec_dict.keys():
        #     prop_indexs = self.ent_vec_dict[ent]
        #     if len(prop_indexs) < 2:
        #         continue
        #     prop_indexs = list(prop_indexs)
        #     prop_embddings = self.embeddings[prop_indexs]
        #     self.ent_embedding[self.reverse_dict[ent]] = np.sum(prop_embddings, axis=0) / len(prop_indexs)
        # self.ent_embedding = preprocessing.normalize(self.ent_embedding)
        attr_embeddings = np.array(attr_embeddings)
        self.ent_embedding_list = [[0.0] * embedding_size for i in range(self.e)]
        for ent in self.ent_attrids_dict.keys():
            prop_indexs = self.ent_attrids_dict[ent]
            if len(prop_indexs) < 2:
                continue
            prop_indexs = list(prop_indexs)
            prop_embddings = attr_embeddings[prop_indexs]
            # self.ent_embedding_list index即id
            self.ent_embedding_list[self.ent_id_dict[ent]] = np.sum(prop_embddings, axis=0) / len(prop_indexs)
        del attr_embeddings

    def get_sim_mat(self):
        mat1 = np.array(self.ent_embedding_list)[self.all_id_list1]
        mat1 = preprocessing.normalize(np.matrix(mat1))
        mat2 = np.array(self.ent_embedding_list)[self.all_id_list2]
        mat2 = preprocessing.normalize(np.matrix(mat2))
        sim_mat = get_sim_mat_after_filter(mat1, mat2, is_sparse=False, is_filtered=False)
        del mat1
        del mat2
        print(sim_mat.min(), sim_mat.max(), sim_mat.mean())
        sim_mat = self.emhance_sim(sim_mat)
        print(sim_mat.min(), sim_mat.max(), sim_mat.mean())

    def emhance_sim(self, sim_mat, th=0.8):
        print("begin enhance_sim...")
        total_sim = 0
        related_pair = dict()
        for e1, e2 in self.sup_ents_pairs:
            if e1 in self.related_ents_dict1.keys() and e2 in self.related_ents_dict2.keys():
                e1_idx = self.all_id_list1.index(e1)
                e2_idx = self.all_id_list2.index(e2)
                total_sim += sim_mat[e1_idx, e1_idx]
                sim_mat[e1_idx, e2_idx] = 1
                # print("sim of sups", self.sim_mat[e1, e2])
                related_ents1 = to_ids(self.related_ents_dict1, self.all_id_list1)
                related_ents2 = to_ids(self.related_ents_dict2, self.all_id_list2)
                for r1 in related_ents1:
                    for r2 in related_ents2:
                        related_pair[(r1, r2)] = related_pair.get((r1, r2), 0) + 1
        print("related pairs", len(related_pair))
        avg_sim = total_sim / len(self.sup_ents_pairs)
        print("ava sim of sups", avg_sim)
        # sim_mat[sim_mat < th // 3] = 0.0
        print(len(related_pair))
        for r1, r2 in related_pair:
            sim_mat[r1, r2] *= (related_pair.get((r1, r2)) + 1) * 100  # big data
            # self.sim_mat[r1, r2] *= pow(3, related_pair.get((r1, r2)))  # small data
            sim_mat[r1, r2] = max(1, sim_mat[r1, r2])
            # if sim_mat[r1, r2] > 0.0001:
            #     sim_mat[r1, r2] = max(1.0, sim_mat[r1, r2])
        print("filtered by sim th:", th)
        sim_mat[sim_mat < th] = 0.0
        sim_mat = preprocessing.normalize(sim_mat, norm='l1')
        sim_mat = scipy.sparse.csr_matrix(sim_mat)
        return sim_mat


def to_ids(related_ent_ids, all_ent_list):
    ids = set()
    for id in related_ent_ids:
        assert id in all_ent_list
        ids.add(all_ent_list.index(id))
    assert len(ids) == len(related_ent_ids)
    return ids


def get_sim_mat_after_filter(mat11, mat22, is_sparse=True, is_filtered=True, th=0.8):
    sim = np.dot(mat11, mat22.T)
    if is_filtered:
        print("filtered by sim th:", th)
        sim[sim < th] = 0.0
    if is_sparse:
        print("begin sparse...")
        sim = scipy.sparse.lil_matrix(sim)
    return sim


def get_common(props_list, props_set):
    print("total props:", len(props_set))
    print("total prop frequency:", len(props_list))
    n = int(data_frequent_p * len(props_set))
    most_frequent_props = collections.Counter(props_list).most_common(n)
    print(most_frequent_props[0:10])
    most_frequent_props = most_frequent_props[len(props_set) - n:]
    common_props_ids = dict()
    for prop, freq in most_frequent_props:
        if freq >= min_frequency and prop not in common_props_ids:
            common_props_ids[prop] = len(common_props_ids)
    print('common props:', len(common_props_ids))
    return common_props_ids


def get_range_weight(range_vec, id1, id2):
    if range_vec[id1] == range_vec[id2]:
        return v
    return 1.0


def generate_batch_random(data_list, batch_size, range_vec):
    batch_data = random.sample(data_list, batch_size)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    range_type = np.ndarray(shape=(batch_size, 1), dtype=np.float32)
    for i in range(len(batch_data)):
        batch[i] = batch_data[i][0]
        labels[i, 0] = batch_data[i][1]
        range_type[i, 0] = get_range_weight(range_vec, batch[i], labels[i, 0])
    return batch, labels, range_type

