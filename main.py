import tensorflow as tf
from include.Config import Config
from include.Model import build, training
from include.Test import get_hits
from include.Load import *
from include.utils import *
from include.models import *

import warnings
warnings.filterwarnings("ignore")

'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

if __name__ == '__main__':

    # load data: e(实体数) + train + (kg1 + kg2)
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))

    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]
    all_ent1_list, all_ent2_list = get_ent_list(np.array(ILL))

    # ae need: 实体-属性矩阵, 实体-实体矩阵, 1
    ae_input, support, num_supports = load_ae_data(Config.lang)

    # se need
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3) 

    # init tf 环境 == clear
    tf.reset_default_graph()

    # build ae obj
    sess_ae, model_ae = ae_session_init(train, ae_input, support, num_supports)

    # build se obj
    sess_se, op_se, output_layer_se, loss_se = se_session_init(e, train, KG1, KG2)

    # train
    training(sess_se, output_layer_se, loss_se, op_se,
          sess_ae, model_ae, ae_input, support, num_supports, 
          Config.epochs, Config.neg_K, train, test, all_ent1_list, all_ent2_list)
