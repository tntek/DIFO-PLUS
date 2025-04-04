import os
import numpy as np
import torch
import random
import time
import src.methods.net.difoplus as difoplus
import src.methods.net.source as SOURCE
from conf import cfg, load_cfg_from_args


if __name__ == "__main__":
    load_cfg_from_args()
    print("+++++++++++++++++++IB+++++++++++++++++++++")
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    cfg.type = cfg.domain
    cfg.t_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
    cfg.test_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.T] + '_list.txt'
    cfg.s_dset_path = cfg.FOLDER + cfg.SETTING.DATASET + '/' + cfg.domain[cfg.SETTING.S] + '_list.txt'
    cfg.savename = cfg.MODEL.METHOD
    start =time.time()
    torch.manual_seed(cfg.SETTING.SEED)
    torch.cuda.manual_seed(cfg.SETTING.SEED)
    np.random.seed(cfg.SETTING.SEED)
    random.seed(cfg.SETTING.SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    if cfg.SETTING.DATASET == 'office-home':
        if cfg.DA == 'pda':
            cfg.class_num = 65
            cfg.src_classes = [i for i in range(65)]
            cfg.tar_classes = [i for i in range(25)]

            
    elif cfg.MODEL.METHOD == "difoplus":
        print("using difoplus method")
        acc = difoplus.train_target(cfg)
    elif cfg.MODEL.METHOD == "source":
        print("training source model")
        acc = SOURCE.train_source(cfg)

    end = time.time()
    all_time = 'Running time: %s Seconds'%(round(end-start, 2))
    print(all_time)
