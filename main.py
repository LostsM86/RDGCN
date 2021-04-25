import tensorflow as tf
from include.Config import Config
from include.Model import build, training
from include.Test import get_hits
from include.Load import *
from include.utils import *

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
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))

    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]

    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    # Load data
    ae_input = load_ae_data(Config.lang)

    model_func = GCN_Align

    # 实体的数量
    e = ae_input[2][0]

    ph_ae = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        # tf.placeholder(tf.float32),
        'features': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder_with_default(0, shape=())
    }
    # TODO(zdh) 校验参数
    model_ae = model_func(ph_ae, input_dim=ae_input[2][1], output_dim=Config.ae_dim,
                          ILL=train, sparse_inputs=True, featureless=False, logging=True)


    feed_dict_ae = construct_feed_dict(ae_input, support, ph_ae)
    feed_dict_ae.update({ph_ae['dropout']: FLAGS.dropout})

    output_layer, loss = build(
        Config.dim, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.k, Config.language[0:2], e, train, KG1 + KG2)
    vec, J = training(output_layer, loss, 0.001,
                      Config.epochs, train, e, Config.k, test)
    print('loss:', J)
    print('Result:')
    get_hits(vec, test)
