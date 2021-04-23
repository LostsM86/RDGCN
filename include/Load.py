import numpy as np


# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1):
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


def get_ent_list(pair):
    nppair = np.array([list(i) for i in pair])
    return nppair[:, 0], nppair[:, 1]


def generate_related_ents(triples):
    related_ents_dict = dict()
    for h, r, t in triples:
        add_dict_kv(related_ents_dict, h, t)
    return related_ents_dict


def add_dict_kv(dic, k, v):
    vs = dic.get(k, set())
    vs.add(v)
    dic[k] = vs
