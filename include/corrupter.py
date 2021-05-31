import tensorflow as tf
from collections import defaultdict
import numpy as np


class BernCorrupter:
    def __init__(self, data, n_ent, n_rel):
        self.bern_prob = self.get_bern_prob(data, n_ent, n_rel)
        self.n_ent = n_ent

    def corrupt(self, head, tail, rela):
        prob = self.bern_prob[rela]
        selection = tf.compat.v1.distributions.Bernoulli(props=prob, dtype='int64')
        ent_random = np.random.choice(self.n_ent, len(head))
        head_out = (1 - selection) * head.numpy() + selection * ent_random
        tail_out = selection * tail.numpy() + (1 - selection) * ent_random
        return tf.variable(head_out, trainable=False), tf.variable(tail_out, trainable=False)

    def get_bern_prob(self, data, n_ent, n_rel):
        head, rela, tail = data
        edges = defaultdict(lambda: defaultdict(lambda: set()))
        rev_edges = defaultdict(lambda: defaultdict(lambda: set()))
        for h, t, r in zip(head, tail, rela):
            edges[r][h].add(t)
            rev_edges[r][t].add(h)
        bern_prob = tf.zeros(n_rel)
        for k in edges.keys():
            right = sum(len(tails) for tails in edges[k].values()) / len(edges[k])
            left = sum(len(heads) for heads in rev_edges[k].values()) / len(rev_edges[k])
            bern_prob[k] = right / (right + left)
        return bern_prob
