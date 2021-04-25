import tensorflow as tf
from include.Config import Config
from include.Model import build, training
from include.Test import *
from include.Load import *
from include.Models_AE import *

import warnings
warnings.filterwarnings("ignore")

import functools
print = functools.partial(print, flush=True)
'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

if __name__ == '__main__':
    print('CONFIG')
    print(Config.__dict__)
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))

    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = np.array(ILL[illL // 10 * Config.seed:])
    all_ent1_list, all_ent2_list = get_ent_list(np.array(ILL))
    ref_ent1_list, ref_ent2_list = get_ent_list(test)

    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    related_ents_dict1 = generate_related_ents(KG1)
    related_ents_dict2 = generate_related_ents(KG2)

    tf.reset_default_graph()

    # ae build
    print('#### AE building')
    model_ae = CorrelationNN(e)
    model_ae.load_base_data(Config.language, train, ref_ent1_list, ref_ent2_list, related_ents_dict1, related_ents_dict2)
    model_ae.build()

    # se build
    print('#### SE building')
    output_layer_se, loss_se, train_step_se, sess_se \
        = build(0.001, Config.dim, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.k, Config.language[0:2], e, KG1, KG2)

    # se\ae
    vec, J = training(output_layer_se, loss_se, train_step_se, sess_se,
                      model_ae,
                      Config.epochs, Config.k, train, test, all_ent1_list, all_ent2_list, ref_ent1_list, ref_ent2_list)

    # print('loss:', J)
    # print('Result:')
    # get_hits(vec, test)
