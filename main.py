from include.models import *

import warnings
warnings.filterwarnings("ignore")

import functools
print = functools.partial(print, flush=True)

seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

if __name__ == '__main__':
    print(Config.__dict__)

    # load data: e(实体数) + train + (kg1 + kg2)
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))

    ILL = loadfile(Config.ill, 2)
    illL = len(ILL)
    # np.random.shuffle(ILL)
    train = np.array(ILL[:illL // 10 * Config.seed])
    test = ILL[illL // 10 * Config.seed:]
    all_ent1_list, all_ent2_list = get_ent_list(np.array(ILL))

    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)

    # ae need: 实体-属性矩阵, 实体-实体矩阵, 1
    ae_input, support, num_supports, all_attr, all_KG1, all_KG2 = load_ae_data(Config.language, e, KG1, KG2, train)

    # init tf 环境 == clear
    tf.reset_default_graph()

    # build ae obj
    sess_ae, ph_ae, model_ae = ae_session_init(train, ae_input, num_supports)

    # build se obj
    sess_se, op_se, output_layer_se, loss_se= se_session_init(e, train, KG1, KG2)

    # train
    training(sess_se, output_layer_se, loss_se, op_se, Config.se_neg_K,
             sess_ae, model_ae, ae_input, support, ph_ae, Config.ae_neg_K, all_attr, [all_KG1, all_KG2],
             e, Config.epochs, train, test, [all_ent1_list, all_ent2_list],
             Config.th, Config.boot_K)
