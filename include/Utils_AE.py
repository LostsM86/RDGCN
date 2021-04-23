import tensorflow as tf


def nce_loss(weights,
             biases,
             labels,
             inputs,
             num_sampled,
             num_classes,
             num_true=1,
             v=None):
    batch_size = int(labels.get_shape()[0])
    if v is None:
        v = tf.ones([batch_size, 1])

    true_logits, sampled_logits = compute_sampled_logits(
        weights=weights,
        biases=biases,
        labels=labels,
        inputs=inputs,
        num_sampled=num_sampled,
        num_classes=num_classes,
        num_true=num_true)
    true_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logits), logits=true_logits)
    sampled_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

    true_loss = tf.multiply(true_loss, v)
    return tf.div(tf.reduce_sum(true_loss) + tf.reduce_sum(sampled_loss), tf.constant(batch_size, dtype=tf.float32))


def compute_sampled_logits(weights,
                           biases,
                           labels,
                           inputs,
                           num_sampled,
                           num_classes,
                           num_true=1):
    if not isinstance(weights, list):
        weights = [weights]
    if labels.dtype != tf.int64:
        labels = tf.cast(labels, tf.int64)
    labels_flat = tf.reshape(labels, [-1])
    sampled_ids, true_expected_count, sampled_expected_count = tf.nn.log_uniform_candidate_sampler(
        true_classes=labels,
        num_true=num_true,
        num_sampled=num_sampled,
        unique=True,
        range_max=num_classes)

    true_w = tf.nn.embedding_lookup(weights, labels_flat)
    true_b = tf.nn.embedding_lookup(biases, labels_flat)
    sampled_w = tf.nn.embedding_lookup(weights, sampled_ids)
    sampled_b = tf.nn.embedding_lookup(biases, sampled_ids)
    dim = tf.shape(true_w)[1:2]
    new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
    row_wise_dots = tf.multiply(tf.expand_dims(inputs, 1), tf.reshape(true_w, new_true_w_shape))
    dots_as_matrix = tf.reshape(row_wise_dots, tf.concat([[-1], dim], 0))
    true_logits = tf.reshape(sum_rows(dots_as_matrix), [-1, num_true])
    true_b = tf.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_b_vec = tf.reshape(sampled_b, [num_sampled])
    sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True) + sampled_b_vec

    return true_logits, sampled_logits


def sum_rows(x):
    """Returns a vector summing up each row of the matrix x."""
    cols = tf.shape(x)[1]
    ones_shape = tf.stack([cols, 1])
    ones = tf.ones(ones_shape, x.dtype)
    return tf.reshape(tf.matmul(x, ones), [-1])


# data_utils
def read_pair_ids(file_path):
    file = open(file_path, 'r', encoding='utf8')
    pairs = list()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        pairs.append((int(params[0]), int(params[1])))
    file.close()
    return pairs


def read_ids(ids_file):
    file = open(ids_file, 'r', encoding='utf8')
    dic, reversed_dic, ids_set, uris_set = dict(), dict(), set(), set()
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        id = int(params[0])
        uri = params[1]
        dic[id] = uri
        reversed_dic[uri] = id
        ids_set.add(id)
        uris_set.add(uri)
    assert len(dic) == len(reversed_dic)
    assert len(ids_set) == len(uris_set)
    return dic, reversed_dic, ids_set, uris_set


def pair_2dict(pairs):
    d = dict()
    for pair in pairs:
        if pair[0] not in d:
            d[pair[0]] = pair[1]
        else:
            print("Error")
    return d


def read_attrs(attrs_file):
    attrs_dic = dict()
    with open(attrs_file, 'r', encoding='utf8') as file:
        for line in file:
            params = line.strip().strip('\n').split('\t')
            if len(params) >= 2:
                attrs_dic[params[0]] = set(params[1:])
            else:
                print(line)
    return attrs_dic


def merge_dicts(dict1, dict2):
    """dict1和dict2合并为dict2"""
    for k in dict1.keys():
        vs = dict1.get(k)
        dict2[k] = dict2.get(k, set()) | vs
    return dict2


def read_lines(file_path):
    if file_path is None:
        return []
    file = open(file_path, 'r', encoding='utf8')
    return file.readlines()


def read_attrs_range(file_path):
    dic = dict()
    lines = read_lines(file_path)
    for line in lines:
        line = line.strip()
        params = line.split('\t')
        assert len(params) == 2
        dic[params[0]] = int(params[1])
    return dic

def read_kg_ent_lists(file_path):
    fr = open(file_path + 'ent_ids_1', 'r')
    ent_1 = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        ent_1.append(int(line[0]))
    fr.close()
    fr = open(file_path + 'ent_ids_2', 'r')
    ent_2 = []
    for line in fr.readlines():
        line = line.strip().split('\t')
        ent_2.append(int(line[0]))
    fr.close()
    return ent_1, ent_2



