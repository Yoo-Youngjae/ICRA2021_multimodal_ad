import torch
import numpy as np
from collections import Iterable
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
import sklearn
import librosa
import decimal


def get_input_size(config):
    if config.sensor == 'All':
        val = 1728
        return val
    elif config.sensor == 'hand_camera':
        return 1024
    elif config.sensor == 'force_torque':
        return 64 # 64
    elif config.sensor == 'head_depth':
        return 512
    elif config.sensor == 'LiDAR':
        return 2048
    elif config.sensor == 'mic':
        return 128

def get_balance(seen_index_list, unseen_index_list, novelty_ratio=.5):
    if novelty_ratio <= 0.:
        return seen_index_list, unseen_index_list

    current_ratio = len(unseen_index_list) / (len(seen_index_list) + len(unseen_index_list))

    if current_ratio < novelty_ratio:
        target_seen_cnt = int(len(unseen_index_list) / novelty_ratio - len(unseen_index_list))
        new_seen_index_list = list(np.random.choice(seen_index_list, target_seen_cnt, replace=False))

        return new_seen_index_list, unseen_index_list
    elif current_ratio > novelty_ratio:
        target_unseen_cnt = int((len(seen_index_list) * novelty_ratio) / (1 - novelty_ratio))
        new_unseen_index_list = list(np.random.choice(unseen_index_list, target_unseen_cnt, replace=False))

        return seen_index_list, new_unseen_index_list
    else:
        return seen_index_list, unseen_index_list

def get_loaders(config, use_full_class=False):
    # get hsr_objectdrop config for Tabular dataset
    import json
    with open('datasets/data_config.json', 'r') as f:
        data_config = json.load(f)
    if config.data not in data_config.keys():
        raise ValueError('no dataset config for'+ config.data)
    data_config = data_config[config.data]

    # split labels 
    class_list = data_config['labels']
    seen_labels, unseen_labels = [], []

    if config.target_class not in class_list:
        if config.data == 'hsr_objectdrop':
            config.target_class = class_list[1]
        else:
            config.target_class = class_list[0]
    
    for i in class_list:
        if use_full_class:
            seen_labels += [i]
            continue
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

    if data_config['from'] in ['youngjae']:

        dset_manager = TabularDatasetManager(
            config=config

        )

    # balance ratio of loaders
    if use_full_class:
        seen_index_list = dset_manager.get_indexes(labels=seen_labels, ratios=[0.6, 0.2, 0.2])
        indexes_list = [
            seen_index_list[0],
            seen_index_list[1],
            seen_index_list[2]
        ]
    else:
        seen_index_list = dset_manager.get_indexes(labels=seen_labels, ratios=[0.6, 0.2, 0.2])
        unseen_index_list = dset_manager.get_indexes(labels=unseen_labels)

        if config.verbose >= 2:
            print(
                'Before balancing:\t|train|=%d |valid|=%d |test_normal|=%d |test_novelty|=%d |novelty_ratio|=%.4f' % (
                    len(seen_index_list[0]),
                    len(seen_index_list[1]),
                    len(seen_index_list[2]),
                    len(unseen_index_list[0]),
                    len(unseen_index_list[0])/(len(unseen_index_list[0])+len(seen_index_list[2]))
                )
            )
        seen_index_list[2], unseen_index_list[0] = get_balance(seen_index_list[2],
                                                               unseen_index_list[0],
                                                               config.novelty_ratio
                                                               )
        if config.verbose >= 1:
            print(
                'After balancing:\t|train|=%d |valid|=%d |test_normal|=%d |test_novelty|=%d |novelty_ratio|=%.4f' % (
                    len(seen_index_list[0]),
                    len(seen_index_list[1]),
                    len(seen_index_list[2]),
                    len(unseen_index_list[0]),
                    len(unseen_index_list[0])/(len(unseen_index_list[0])+len(seen_index_list[2]))
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


class SequentialIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# writen by chungyeon_lee
class HSR_Net(nn.Module):
    def __init__(self, unimodal, config):
        super(HSR_Net, self).__init__()
        self.conv1r = nn.Conv2d( 3, 16, kernel_size=2, stride=2)
        self.conv2r = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3r = nn.Conv2d(16, 16, kernel_size=2, stride=2)

        self.conv1d = nn.Conv2d( 1,  8, kernel_size=2, stride=2)
        self.conv2d = nn.Conv2d( 8,  8, kernel_size=3, stride=1, padding=1)
        self.conv3d = nn.Conv2d( 8,  8, kernel_size=2, stride=2)
        self.batch_size = config.slicing_size
        self.config = config


        # 1d version
        self.conv1l = nn.Conv1d( 1,  8, kernel_size=18, stride=9, padding=9)
        self.conv2l = nn.Conv1d( 8, 16, kernel_size=2, stride=2)
        self.conv3l = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        self.conv4l = nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=3)
        self.conv5l = nn.Conv1d(16, 32, kernel_size=2, stride=2)


        self.conv1m = nn.Conv1d(1, 12, kernel_size=2, stride=1)
        self.conv2m = nn.Conv1d(12, 8, kernel_size=2, stride=2, padding=2)
        self.unimodal = unimodal



    def forward(self, r, d, l, t, m):
        out = torch.Tensor().cuda(self.config.gpu_id)

        for i in range(self.batch_size):

            if r is not None:
                rr = F.relu(self.conv1r(r[i]))
                rr = F.relu(self.conv2r(rr))
                rr = F.relu(self.conv3r(rr))
                if self.unimodal:
                    result = rr

            if d is not None:
                dd = F.relu(self.conv1d(d[i]))
                dd = F.relu(self.conv2d(dd))
                dd = F.relu(self.conv3d(dd))
                if self.unimodal:
                    result = dd


            if l is not None:
                # 1d version
                ll = F.relu(self.conv1l(l[i]))
                ll = F.relu(self.conv2l(ll))
                ll = F.relu(self.conv3l(ll))
                ll = F.relu(self.conv4l(ll))
                ll = F.relu(self.conv5l(ll))
                ll = ll.view(-1, 32, 8, 1).repeat(1, 1, 1, 8)
                if self.unimodal:
                    result = ll

            if t is not None:
                # Broadcast
                tt = t[i].repeat(1, 1, 8, 8) # (1, 2, 8, 8)
                if self.unimodal:
                    result = tt

            if m is not None:
                mm = F.relu(self.conv1l(m[i]))
                mm = F.relu(self.conv2l(mm))
                mm = mm.view(-1, 2, 8, 1).repeat(1, 1, 1, 8)
                if self.unimodal:
                    result = mm

            # Concatenate
            if not self.unimodal:
                result = torch.cat((rr, dd, tt, mm), dim=1)

            out = torch.cat((out, result), 0)
        return out



class TabularDataset(Dataset):
    def __init__(self, config, transform=None, target_transform=None):
        All = False
        hand_camera = False
        force_torque = False
        head_depth = False

        mic = False
        unimodal = True


        if config.sensor == 'All':
            All = True
            unimodal = False
        elif config.sensor == 'hand_camera':
            hand_camera = True
        elif config.sensor == 'force_torque':
            force_torque = True
        elif config.sensor == 'head_depth':
            head_depth = True
        elif config.sensor == 'mic':
            mic = True



        data_sum_file = config.file_name
        if data_sum_file is not 'data_sum':
            df_datasum = pd.read_csv(config.data_folder_name+data_sum_file+'0.csv')
        elif config.object_select_mode:
            df_datasum = pd.read_csv(config.data_folder_name+data_sum_file+'0.csv')
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'1.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'2.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'3.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'4.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'5.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'6.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'7.csv'), ignore_index=True)
            df_objectlist = pd.read_csv(config.data_folder_name+'objectsplit.csv')
            print(config.object_type)
            df_objectlist = df_objectlist[config.object_type]           # book only mode !!! # cracker doll metalcup eraser cookies book plate bottle
            object_dir_list = df_objectlist.to_list()
            df_datasum = df_datasum[df_datasum['data_dir'].isin(object_dir_list)]
            df_datasum.index = [i for i in range(len(df_datasum.index))]
            df_datasum = df_datasum.loc[:config.slicing_size - 1]
            df_datasum = sklearn.utils.shuffle(df_datasum)  # shuffle
        else:
            df_datasum = pd.read_csv(config.data_folder_name+data_sum_file+'0.csv')
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'1.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'2.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'3.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'4.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'5.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'6.csv'), ignore_index=True)
            df_datasum = df_datasum.append(pd.read_csv(config.data_folder_name+data_sum_file+'7.csv'), ignore_index=True)

            df_datasum = sklearn.utils.shuffle(df_datasum)
            df_datasum.index = [i for i in range(len(df_datasum.index))]
            df_datasum = df_datasum.loc[:config.slicing_size - 1]

        depth_series = df_datasum['cur_depth_id']
        hand_series = df_datasum['cur_hand_id']
        hand_weight_series = df_datasum['cur_hand_weight']
        data_dir = df_datasum['data_dir']
        label_series = df_datasum['label']

        ## essential erase
        data = df_datasum.drop(columns=['data_dir'])
        data = data.drop(columns=['now_timegap'])
        data = data.drop(columns=['label'])
        data = data.drop(columns=['id'])
        data = data.loc[:, ~data.columns.str.match('Unnamed')]

        if All:
            for i in range(963):
                if i < 10:
                    data = data.drop(columns='LiDAR00' + str(i))
                elif i< 100:
                    data = data.drop(columns='LiDAR0' + str(i))
                else:
                    data = data.drop(columns='LiDAR' + str(i))
        elif hand_camera:
            data = hand_series.to_frame() # pd.concat([,label_series], axis=1)
        elif force_torque:
            data = hand_weight_series.to_frame() #pd.concat([, label_series], axis=1)
        elif head_depth:
            data = depth_series.to_frame() #pd.concat([, label_series], axis=1)

        if mic or All:
            mic_df = None
            for i in range(13):
                if i == 0:
                    mic_df = data['mfcc00']
                elif i <10:
                    mic_df = pd.concat([mic_df, data['mfcc0' + str(i)]], axis=1)
                else:
                    mic_df = pd.concat([mic_df, data['mfcc' + str(i)]], axis=1)
            if mic:
                data = mic_df



        base_depth_arr = np.array([])
        base_hand_arr = np.array([])
        firstRow = True
        if hand_camera or head_depth or All:
            for idx, data_dir_str in tqdm(zip(data.index,data_dir)):
                nowdf = data.loc[idx]

                if hand_camera or All:
                    hand_dir = '/data_ssd/hsr_dropobject/data/' + data_dir_str + '/data/img/hand/' + str(
                        int(nowdf['cur_hand_id'])) + '.png'
                    hand_im = Image.open(hand_dir).resize((32, 24))
                    hand_arr = np.array(hand_im)
                    # print(hand_arr.shape)
                    hand_arr = hand_arr.reshape(1, -1)
                if head_depth or All:
                    depth_dir = '/data_ssd/hsr_dropobject/data/' + data_dir_str + '/data/img/d/' + str(
                        int(nowdf['cur_depth_id'])) + '.png'
                    depth_im = Image.open(depth_dir).resize((32, 24))
                    depth_arr = np.array(depth_im).reshape(1, -1)
                else:
                    pass

                #reshape for 1d
                if firstRow:
                    firstRow = False
                    if hand_camera or All:
                        base_hand_arr = hand_arr
                    if head_depth or All:
                        base_depth_arr = depth_arr
                else:
                    if hand_camera or All:
                        base_hand_arr = np.concatenate((base_hand_arr, hand_arr), axis=0)
                    if head_depth or All:
                        base_depth_arr = np.concatenate((base_depth_arr, depth_arr), axis=0)


        if hand_camera or All:
            r = self.norm_vec_np(base_hand_arr)
            r = torch.from_numpy(r.astype(np.float32))
            r = r.view(-1, 1, 3, 24, 32).squeeze()
            r = F.interpolate(r, 32).view(-1,1,3,32,32)
            r = r.cuda(config.gpu_id)
            print(r.shape)
        if head_depth or All:
            d = self.norm_vec_np(base_depth_arr)
            d = torch.from_numpy(d.astype(np.float32))
            d = d.view(-1, 1, 24, 32)
            d = F.interpolate(d, 32).view(-1, 1, 1, 32, 32)
            d = d.cuda(config.gpu_id)
            print(d.shape)

        if (force_torque or All) :
            t = self.norm_vec_np(hand_weight_series.to_numpy())
            t = torch.from_numpy(t.astype(np.float32))
            t = t.view(-1, 1)
            t = t.cuda(config.gpu_id)
            print(t.shape)
        if mic or All:
            mic_df = mic_df.to_numpy()
            m = self.norm_vec_np(mic_df)
            m = torch.from_numpy(m.astype(np.float32))
            m = m.view(-1, 1, 1, 13)
            m = m.cuda(config.gpu_id)
            print(m.shape)


        if All:
            hsr_net = HSR_Net(unimodal, config).cuda(config.gpu_id)
            data = hsr_net(r, d, None, t, m)
            print(data.shape)
            data = data.view(-1, 1728)
        if hand_camera:
            hsr_net = HSR_Net(unimodal, config).cuda(config.gpu_id)
            data = hsr_net(r, None, None, None, None)
            print(data.shape)
            data = data.view(-1, 1024)
        if force_torque:
            hsr_net = HSR_Net(unimodal, config).cuda(config.gpu_id)
            data = hsr_net(None, None, None, t, None)
            print(data.shape)
            data = data.view(-1, 64) # 64 128
        if head_depth:
            hsr_net = HSR_Net(unimodal, config).cuda(config.gpu_id)
            data = hsr_net(None, d, None, None, None)
            print(data.shape)
            data = data.view(-1, 512)
        if mic:
            hsr_net = HSR_Net(unimodal, config).cuda(config.gpu_id)
            data = hsr_net(None, None, None, None, m)
            print(data.shape)
            data = data.view(-1, 128)



        self.transform = transform
        self.target_transform = target_transform

        self.data = data
        print(self.data.shape)
        self.targets = torch.from_numpy(label_series.to_numpy().astype(np.float32))


    def norm_vec(self,v, range_in=None, range_out=None):
        if range_out is None:
            range_out = [0.0,1.0]
        if range_in is None:
            range_in = [torch.min(v, 0), torch.max(v, 0)] #range_in = [np.min(v,0), np.max(v,0)]
        r_out = range_out[1] - range_out[0]
        r_in = range_in[1] - range_in[0]
        v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
        v = np.nan_to_num(v, nan=0.0)
        return v

    def norm_vec_np(self,v, range_in=None, range_out=None):
        if range_out is None:
            range_out = [0.0,1.0]
        if range_in is None:
            range_in = [np.min(v,0), np.max(v,0)]
        r_out = range_out[1] - range_out[0]
        r_in = range_in[1] - range_in[0]
        v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
        v = np.nan_to_num(v) # , nan=0.0
        return v

    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class TabularDatasetManager:
    
    def __init__(self, config, transform=None, target_transform=None,
        shuffle=False, data_size=0):

        data_list = []
        targets_list = []

        self.train_dataset = self._get_dataset(config=config)
        if self.train_dataset:
            data, targets = self.train_dataset.data, self.train_dataset.targets
            data_list.append(data)
            targets_list.append(targets)

        if type(data[0]) == np.ndarray:
            self.total_x = np.concatenate(data_list)[-data_size:] if data_list else None
            self.total_y = np.concatenate(targets_list)[-data_size:] if targets_list else None
        elif type(data[0]) == torch.Tensor:
            self.total_x = torch.cat(data_list)[-data_size:] if data_list else None
            self.total_y = torch.cat(targets_list)[-data_size:] if targets_list else None
        else:
            raise NotImplementedError

        if shuffle:
            from numpy.random import permutation
            shuffled_indices = permutation(len(self.total_x))
            self.total_x = self.total_x[shuffled_indices]
            self.total_y = self.total_y[shuffled_indices]

        self.total_size = len(self.total_x)

        self.train_dataset.data = self.total_x
        self.train_dataset.targets = self.total_y
        self.train_dataset.transform = transform
        self.train_dataset.target_transform = target_transform
        
    def _get_dataset(self, config):

        dataset = TabularDataset(config)

        return dataset

    def get_indexes(self, ratios=None, labels=None):
        if labels is not None: 
            if not isinstance(labels, Iterable):
                labels = [labels]
            indexes = list(np.where(np.isin(self.total_y, labels))[0])
        else:
            indexes = list(range(self.total_size))

        if ratios:
            assert sum(ratios) == 1
            if len(ratios) == 1:
                return indexes
            else:
                ratios = np.array(ratios)
                indexes_list = np.split(indexes, [int(e) for e in (ratios.cumsum()[:-1] * len(indexes))])
                indexes_list = [list(indexes) for indexes in indexes_list]
        else:
            indexes_list = [indexes]

        return indexes_list

    def get_transformed_data(self, data_loader):
        """
        Multi indexing support
        """
        x = []
        y = []
        for i in data_loader.sampler:
            # indexing -> __getitem__ -> applying transform
            _x, _y = data_loader.dataset[i]
            x.append(_x)
            y.append(_y)

        if type(_x) == np.ndarray:
            x = np.stack(x)
        elif type(_x) == torch.Tensor:
            x = torch.stack(x)
        else:
            raise NotImplementedError

        if type(_y) == np.ndarray:
            y = np.array(y)
        elif type(_y) == torch.Tensor:
            y = torch.tensor(y)

        return x, y
    
    def get_loaders(self, batch_size, ratios=None, indexes_list=None, use_gpu=False):
        if ratios and indexes_list:
            raise Exception("Only either `ratios` or `indexes_list` is allowed")
        elif ratios:
            indexes_list = self.get_indexes(ratios=ratios)

        if self.train_dataset.transform is not None:
            self.train_dataset.transform.fit(self.train_dataset.data[indexes_list[0]])

        if len(indexes_list) == 2:
            return [
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SubsetRandomSampler(indexes_list[0]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[1]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
            ]
        elif len(indexes_list) == 3:
            return [
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SubsetRandomSampler(indexes_list[0]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[1]),
                    pin_memory=use_gpu,
                    num_workers=0,
                ),
                DataLoader(
                    self.train_dataset, batch_size,
                    sampler=SequentialIndicesSampler(indexes_list[2]),
                    pin_memory=use_gpu,
                    num_workers=0,
                )
            ]


class Multisensory_module(nn.Module):
    def __init__(self, config, unimodal=False):
        super(Multisensory_module, self).__init__()
        self.batch_size = config.batch_size
        self.config = config
        self.unimodal = unimodal

        self.conv1r = nn.Conv2d( 3, 16, kernel_size=2, stride=2)
        self.conv2r = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3r = nn.Conv2d(16, 16, kernel_size=2, stride=2)

        self.conv1d = nn.Conv2d( 1,  8, kernel_size=2, stride=2)
        self.conv2d = nn.Conv2d( 8,  8, kernel_size=3, stride=1, padding=1)
        self.conv3d = nn.Conv2d( 8,  8, kernel_size=2, stride=2)


        # 1d version
        self.conv1l = nn.Conv1d( 1,  8, kernel_size=18, stride=9, padding=9)
        self.conv2l = nn.Conv1d( 8, 16, kernel_size=2, stride=2)
        self.conv3l = nn.Conv1d(16, 32, kernel_size=2, stride=2)
        self.conv4l = nn.Conv1d(32, 16, kernel_size=3, stride=2, padding=3)
        self.conv5l = nn.Conv1d(16, 32, kernel_size=2, stride=2)


        self.conv1m = nn.Conv1d(1, 12, kernel_size=2, stride=1)
        self.conv2m = nn.Conv1d(12, 8, kernel_size=2, stride=2, padding=2)


    def forward(self, r, d, t, m):
        out = torch.Tensor().cuda(self.config.gpu_id)
        for i in range(self.batch_size):
            if r is not None:
                rr = F.relu(self.conv1r(r[i]))
                rr = F.relu(self.conv2r(rr))
                rr = F.relu(self.conv3r(rr))
                if self.unimodal:
                    result = rr
            if d is not None:
                dd = F.relu(self.conv1d(d[i]))
                dd = F.relu(self.conv2d(dd))
                dd = F.relu(self.conv3d(dd))
                if self.unimodal:
                    result = dd

            # if l is not None:
            #     # 1d version
            #     ll = F.relu(self.conv1l(l[i]))
            #     ll = F.relu(self.conv2l(ll))
            #     ll = F.relu(self.conv3l(ll))
            #     ll = F.relu(self.conv4l(ll))
            #     ll = F.relu(self.conv5l(ll))
            #     ll = ll.view(-1, 32, 8, 1).repeat(1, 1, 1, 8)
            #     if self.unimodal:
            #         result = ll

            if t is not None:
                # Broadcast
                tt = t[i].repeat(1, 1, 8, 8) # (1, 2, 8, 8)
                if self.unimodal:
                    result = tt

            if m is not None:
                mm = F.relu(self.conv1l(m[i]))
                mm = F.relu(self.conv2l(mm))
                mm = mm.view(-1, 2, 8, 1).repeat(1, 1, 1, 8)
                if self.unimodal:
                    result = mm

            # Concatenate
            if not self.unimodal:
                result = torch.cat((rr, dd, tt, mm), dim=1)

            out = torch.cat((out, result), 0)
        return out

def save_mfcc_from_wav(mic_q, config, length, window_size=0.1, stride=0.1):
    # load wav

    y = np.frombuffer(b''.join(mic_q),dtype=np.int16)
    sr = 44100 # rate
    # check if more than 'length' sec
    if len(y) < sr * length:
        print('length of wav file must be over ' + str(length) + ' seconds')

    # cut wav to exactly 'length' seconds
    length = round(length, 1)
    slice_idx = int(round(sr * length))
    # y = y[:slice_idx]

    # apply MFCC
    nfft = int(round(sr * window_size))
    hop = int(round(sr * stride))
    S = librosa.feature.melspectrogram(y.astype('float32'), sr=sr, n_mels=128, n_fft=nfft, hop_length=hop)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)  # n_mfcc => number of features

    # reform [n_mfcc x length*(1/stride)] -> [length*(1/stride) x n_mfcc]
    mfcc = mfcc.T
    mfcc = mfcc[:config.batch_size,:]
    print('mfcc.shape', mfcc.shape)

    return mfcc

def norm_vec(v, range_in=None, range_out=None):
    if range_out is None:
        range_out = [-1, 1]
    if range_in is None:
        range_in = [torch.min(v), torch.max(v)]

    r_out = range_out[1] - range_out[0]
    r_in = range_in[1] - range_in[0]
    v = (r_out * (v - range_in[0]) / r_in) + range_out[0]
    return v

def HsrDataset(config, force_q, hand_q, depth_q, mic_q):
    t = torch.tensor(force_q, dtype=torch.float32)
    r = torch.tensor(hand_q, dtype=torch.float32).view(-1, 1, 3, 32, 32)
    d = torch.tensor(depth_q, dtype=torch.float32).view(-1, 1, 1, 32, 32)
    m = torch.tensor(mic_q, dtype=torch.float32).view(-1, 1, 1, 13)
    r = norm_vec(r, range_in=[0, 255])
    d = norm_vec(d, range_in=[0, 255])
    m = norm_vec(m)
    t = norm_vec(t, range_in=[0, 400])

    multisensory_module = Multisensory_module(config).cuda(config.gpu_id)

    fusion_representation = multisensory_module(r.cuda(config.gpu_id),
                                                d.cuda(config.gpu_id),
                                                t.cuda(config.gpu_id),
                                                m.cuda(config.gpu_id))

    return fusion_representation.view(config.batch_size, -1)


def get_realtime_dataloader(config, force_q, hand_q, depth_q, mic_q):
    mic_q = save_mfcc_from_wav(mic_q, config, length=decimal.Decimal(config.batch_size/10), window_size=0.1, stride=0.1)
    fusion_representation = HsrDataset(config, force_q, hand_q, depth_q, mic_q)
    return fusion_representation

