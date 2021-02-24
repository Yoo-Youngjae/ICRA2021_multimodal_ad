#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.


#  this code is for hsr slip detection
import torch
import numpy as np

from torch import optim
from ignite.engine import Engine, Events

from utils.metric import *
from utils.data_loaders import get_loaders, get_input_size
from matplotlib import pyplot as plt
import pandas as pd
import time

class NoveltyDetecter():

    def __init__(self, config):
        self.config = config

    def show_detect_slip(self, test_x, test_y, model, batch_size=171): # test_x length = 30
        model.eval()

        if isinstance(test_x, np.ndarray):
            test_x = torch.tensor(test_x)
        test_x = test_x.view(batch_size,1,self.config.input_size)
        loss_fn = torch.nn.MSELoss(reduction="sum")
        x_plot = []
        y_plot = []

        true_plot = []
        false_plot = []

        label_plot = []
        for i, (x, y) in enumerate(zip(test_x, test_y)):
            model.eval()
            x = x.to(next(model.parameters()).device).float()
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            print(i, loss.item(), y)
            x_plot.append(i)
            y_plot.append(loss.item())
            if bool(y) == True: # is np true
                label_plot.append(1)
                true_plot.append(x)
            else:
                label_plot.append(0)
                false_plot.append(x)

        # plt.subplot(2,1,1)
        # plt.plot(x_plot, y_plot)
        # plt.title('RaPP Loss')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(x_plot, label_plot)
        # plt.title('Label')
        # plt.tight_layout()
        plt.hist(x_plot, bins=int(180/5))
        plt.show()






def get_config():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=0)

    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--verbose', type=int, default=1)

    p.add_argument('--data', type=str, default='steel')
    p.add_argument('--unimodal_normal', action='store_true', default=False)
    p.add_argument('--target_class', type=str, default=1)

    p.add_argument('--novelty_ratio', type=float, default=.0)
    p.add_argument('--models', type=str, default='ae')
    p.add_argument('--btl_size', type=int, default=20)
    p.add_argument('--n_layers', type=int, default=10)

    p.add_argument('--use_rapp', action='store_true', default=True)
    p.add_argument('--start_layer_index', type=int, default=0)
    p.add_argument('--end_layer_index', type=int, default=-1)

    p.add_argument('--n_trials', type=int, default=1)

    config = p.parse_args()

    return config



def load_data():
    return 0
if __name__ == '__main__':
    from model_builder import get_model

    config = get_config()
    config.input_size = 3648
    model = get_model(config)
    model.load_state_dict(torch.load("/home/yjyoo/PycharmProjects/Rapp/savedRaPP/savedRaPP_All_0515.pt"))
    from reconstruction_aggregation import get_diffs

    dset_manager, train_loader, valid_loader, test_loader = get_loaders(config)
    _train_x, _ = dset_manager.get_transformed_data(train_loader)
    _valid_x, _ = dset_manager.get_transformed_data(valid_loader)
    train_diff_on_layers = get_diffs(_train_x, model)
    valid_diff_on_layers = get_diffs(_valid_x, model)
    detecter = NoveltyDetecter(config)
    start_time = time.time()



    # while time.time() - start_time <= 10:
    _test_x = load_data()
    detecter.show_detect_slip(_test_x,model, batch_size=171)


