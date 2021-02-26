
#  this code is for hsr slip detection
import torch
import numpy as np
from utils.data_loaders import TabularDatasetManager, get_input_size
from matplotlib import pyplot as plt
from utils.normalize import Standardizer, Rotater
import pandas as pd
import math
import time

def get_norm(x, norm_type=2):
    return abs(x)**norm_type

def get_d_norm_loss(test_diffs,
                    config,
                    start_layer_index=0,
                    end_layer_index=None,
                    gpu_id=-1,
                    norm_type=2
                   ):

    if end_layer_index is None:
        end_layer_index = len(test_diffs) + 1

    if start_layer_index > len(test_diffs) - 1:
        start_layer_index = len(test_diffs) - 1

    if end_layer_index - start_layer_index < 1:
        end_layer_index = start_layer_index + 1


    train_diffs = torch.load(config.train_diffs)


    test_diffs = torch.cat([torch.from_numpy(i) for i in test_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()

    # |test_diffs| = (batch_size, dim * config.n_layers)

    rotater = Rotater()
    stndzer = Standardizer()

    rotater.fit(train_diffs, gpu_id=gpu_id)
    stndzer.fit(rotater.run(train_diffs, gpu_id=gpu_id))

    nap_time = time.time()
    test_rotateds = stndzer.run(rotater.run(test_diffs, gpu_id=gpu_id))
    print('nap_time', time.time() - nap_time)
    score = get_norm(test_rotateds, norm_type).mean(axis=1)


    return score

class NoveltyDetecter():

    def __init__(self, config):
        self.config = config

    def show_detect_slip(self, test_x, test_y, model): # test_x length = 30
        model.eval()

        if isinstance(test_x, np.ndarray):
            test_x = torch.tensor(test_x)
        test_x = test_x.view(-1,1,self.config.input_size)
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

        plt.subplot(2,1,1)
        plt.plot(x_plot, y_plot)
        plt.title('RaPP Loss')

        plt.subplot(2, 1, 2)
        plt.plot(x_plot, label_plot)
        plt.title('Label')
        plt.tight_layout()
        # plt.hist(x_plot, bins=int(180/5))
        plt.show()


    def test(self,
             model,
             dset_manager,
             test_loader,
             config
             ):
        from reconstruction_aggregation import get_diffs

        model.eval()

        with torch.no_grad():

            _test_x, _test_y = dset_manager.get_transformed_data(test_loader)

            # _test_x = _test_x.view(171,1,2048)[0]
            # _test_y = _test_y[0]

            if self.config.unimodal_normal:
                _test_y = np.where(np.isin(_test_y, [self.config.target_class]), False, True)
            else:
                _test_y = np.where(np.isin(_test_y, [self.config.target_class]), True, False)

            show_mode = False
            if show_mode:
                self.show_detect_slip(_test_x, _test_y, model)

            test_diff_time = time.time()
            test_diff_on_layers = get_diffs(_test_x, model)
            print('test_diff_time', time.time() - test_diff_time)

        score = get_d_norm_loss(
            test_diff_on_layers,
            config,
            gpu_id=self.config.gpu_id,
            start_layer_index=self.config.start_layer_index,
            end_layer_index=self.config.n_layers + 1 - self.config.end_layer_index,
            norm_type=2

        )

        return score



def get_loaders(config, dir_name):
    import json
    with open('datasets/data_config.json', 'r') as f:
        data_config = json.load(f)
        data_config = data_config[config.data]

        # split labels
        class_list = data_config['labels']
        seen_labels, unseen_labels = [], []
        if config.target_class not in class_list:
            config.target_class = class_list[1]

        for i in class_list:
            if i != config.target_class:
                if config.unimodal_normal:
                    unseen_labels += [i]
                else:
                    seen_labels += [i]
            else:
                if config.unimodal_normal:
                    seen_labels += [i]
                else:
                    unseen_labels += [i]


        from sklearn.preprocessing import StandardScaler
        dset_manager = TabularDatasetManager(
            dataset_name=config.data,
            config=config,
            csv_num=0,
            full_test=dir_name
            # transform=StandardScaler(),
        )
        # balance ratio of loaders

        seen_index_list = dset_manager.get_indexes(labels=seen_labels, ratios=[0, 0, 1])
        unseen_index_list = dset_manager.get_indexes(labels=unseen_labels)

        if config.verbose >= 1:
            print(
                'After balancing:\t|train|=%d |valid|=%d |test_normal|=%d |test_novelty|=%d |novelty_ratio|=%.4f' % (
                    len(seen_index_list[0]),
                    len(seen_index_list[1]),
                    len(seen_index_list[2]),
                    len(unseen_index_list[0]),
                    len(unseen_index_list[0]) / (len(unseen_index_list[0]) + len(seen_index_list[2]))
                )
            )

        indexes_list = [
            seen_index_list[0],
            seen_index_list[1],
            seen_index_list[2] + unseen_index_list[0]
        ]

        train_loader, valid_loader, test_loader = dset_manager.get_loaders(
            batch_size=config.batch_size, indexes_list=indexes_list, ratios=None
        )

        return dset_manager, train_loader, valid_loader, test_loader


def get_config():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--batch_size', type=int, default=7000)
    p.add_argument('--slicing_size', type=int, default=56000)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--data', type=str, default='hsr_objectdrop')
    p.add_argument('--unimodal_normal', action='store_true', default=False)
    p.add_argument('--target_class', type=str, default=1)

    p.add_argument('--novelty_ratio', type=float, default=.0)
    p.add_argument('--btl_size', type=int, default=100) # 100, 10
    p.add_argument('--n_layers', type=int, default=5) # 5, 3

    p.add_argument('--start_layer_index', type=int, default=0)
    p.add_argument('--end_layer_index', type=int, default=-1)
    p.add_argument('--from', type=str, default="youngjae")


    p.add_argument('--folder_name', type=str, default="hsr_objectdrop/")
    p.add_argument('--models', type=str, default='ae')
    p.add_argument('--save_mode', action='store_true', default=False)

    p.add_argument('--data_folder_name', type=str, default="/data_ssd/hsr_dropobject/")
    p.add_argument('--file_name', type=str, default="data_sum") # data_sum_motion, data_sum_free
    p.add_argument('--sensor', type=str, default="All")  # All hand_camera force_torque head_depth mic LiDAR
    p.add_argument('--saved_name', type=str, default="datasets/All_100.pt")
    p.add_argument('--saved_data', type=str, default="All")
    p.add_argument('--saved_result', type=str, default="1_26/All_sec")
    p.add_argument('--object_select_mode', action='store_true', default=False)
    p.add_argument('--train_diffs', type=str, default='datasets/All_train_diffs.pt')

    config = p.parse_args()

    if config.file_name is not 'data_sum':
        config.slicing_size = 7000

    return config

def main(config):
    from model_builder import get_model
    config.input_size = get_input_size(config)
    model = get_model(config)
    print(model)
    model.load_state_dict(torch.load(config.saved_name))
    detecter = NoveltyDetecter(config)



    dir_name = 'datasets/caltime_test.csv'
    df = pd.read_csv(dir_name)
    config.batch_size = len(df)
    dset_manager, train_loader, valid_loader, test_loader = get_loaders(config, dir_name)
    score = detecter.test(
            model,
            dset_manager,
            test_loader,
            config
        )
    plt.plot(score, color='r')
    plt.show()
    # logscore = [math.log(i) for i in score]
    # for i in score:
    #     print(i)
    # print('max, min',max(score), min(score))


if __name__ == '__main__':
    config = get_config()
    main(config)
