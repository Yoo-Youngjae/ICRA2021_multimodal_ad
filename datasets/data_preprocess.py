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
import pandas as pd
import numpy as np
import os


def calc_time_bound(df_rgb, df_depth, df_LiDAR, df_hand_weight, df_microphone):
    rgb_timegap = df_rgb['timegap']
    depth_timegap = df_depth['timegap']
    LiDAR_timegap = df_LiDAR['timegap']
    hand_weight_timegap = df_hand_weight['timegap']
    microphone_timegap = df_microphone['timegap']

    start_time = min([rgb_timegap.values[0], depth_timegap.values[0], LiDAR_timegap.values[0], hand_weight_timegap.values[0], microphone_timegap.values[0]])
    end_time = max([rgb_timegap.values[-1], depth_timegap.values[-1], LiDAR_timegap.values[-1], hand_weight_timegap.values[-1], microphone_timegap.values[-1]])
    start_time = round(start_time +1, 1)
    end_time  = round(end_time -1, 1)
    print("== Time Bound ==")
    print(start_time)
    print(end_time)
    print("=====")
    return start_time, end_time

def find_rgb_id(now_time, df_rgb, rgb_id, data_frequency):
    for _, time, id, timegap in df_rgb.values[rgb_id:-1]:
        if now_time <= timegap and now_time + data_frequency >= timegap:
            rgb_id = id
            return id, rgb_id

def find_depth_id(now_time, df_depth, depth_id, data_frequency):
    for _, time, id, timegap in df_depth.values[depth_id:-1]:
        if now_time <= timegap and now_time + data_frequency >= timegap:
            depth_id = id
            return id, depth_id

def find_LiDAR_data(now_time, df_LiDAR, LiDAR_id, data_frequency):
    for _, data, time, timegap in df_LiDAR.values[LiDAR_id:-1]:
        if now_time <= timegap and now_time+ data_frequency >= timegap:
            data = data[1: -1]
            data = data.split(',')
            data = [float(i) for i in data]
            LiDAR_id = _
            return data, LiDAR_id

def find_hand_weight(now_time, df_hand_weight, hand_weight_id, data_frequency):
    for _, datetime, timegap, weight in df_hand_weight.values[hand_weight_id:-1]:
        if now_time <= timegap and now_time+ data_frequency >= timegap:
            hand_weight_id = _
            return weight, hand_weight_id


def find_hand_id(now_time, df_hand, hand_id, data_frequency, dir_name):
    for _, time, id, timegap in df_hand.values[hand_id:-1]:
        if now_time <= timegap and now_time + data_frequency >= timegap:
            hand_id = id
            return id, hand_id

def find_mic_data(now_time, df_microphone, mic_id, data_frequency):
    for _, data, time, timegap in df_microphone.values[mic_id:-1]:
        if now_time <= timegap and now_time+ data_frequency >= timegap:
            data = data[1: -1]
            data = data.split(',')
            data = [float(i) for i in data]
            Mic_id = _
            return data, Mic_id

def hsr_preprocess(data_dir, folder_name, file_name, root):
    filenames = os.listdir(root+'/'+folder_name+'data')
    data_df = pd.DataFrame(
        [{'now_timegap': 0, 'cur_rgb_id': 0, 'cur_depth_id': 0,
          'cur_hand_id': 0,'cur_hand_weight': 0, 'data_dir' : 0}])

    for data_dir_item in filenames:
        dir_name = root+'/'+folder_name+'data/'+data_dir_item+'/data'
        df_rgb = pd.read_csv(dir_name+'/rgb.csv')[1:]
        df_depth = pd.read_csv(dir_name+'/depth.csv')[1:]
        df_LiDAR = pd.read_csv(dir_name+'/LiDAR.csv')[1:]
        df_hand = pd.read_csv(dir_name + '/hand.csv')[1:]
        df_hand_weight = pd.read_csv(dir_name+'/hand_weight.csv')[1:]
        df_microphone = pd.read_csv(dir_name + '/Microphone.csv')[1:]
        df_drop_time = pd.read_csv(dir_name + '/drop_time.csv')
        _, drop_end, drop_start  = df_drop_time.values[0]


        rgb_id = 0
        depth_id = 0
        hand_id = 0
        LiDAR_id = 0
        hand_weight_id = 0
        mic_id = 0
        data_frequency = 0.1  # 10 frames
        drop_duration = 0.5  # 0.5 sec

        start_time, end_time = calc_time_bound(df_rgb, df_depth, df_LiDAR, df_hand_weight, df_microphone)
        now_time = start_time


        while now_time <= drop_start+0.5:
            cur_rgb_id, rgb_id =        find_rgb_id(now_time, df_rgb, rgb_id, data_frequency)
            cur_depth_id, depth_id =    find_depth_id(now_time, df_depth, depth_id, data_frequency)
            cur_hand_id, hand_id =      find_hand_id(now_time, df_hand, hand_id, data_frequency, dir_name+'/img/hand/')
            cur_LiDAR_data, LiDAR_id =  find_LiDAR_data(now_time, df_LiDAR, LiDAR_id, data_frequency)
            cur_hand_weight, hand_weight_id = find_hand_weight(now_time, df_hand_weight, hand_weight_id, data_frequency)
            cur_mic, mic_id =           find_mic_data(now_time, df_microphone, mic_id, data_frequency)
            for li_idx, li_data in enumerate(cur_LiDAR_data):
                if li_idx == 0:
                    total_lidar_df = pd.DataFrame([{'LiDAR00'+str(li_idx): li_data}])
                else:
                    num = li_idx
                    str_num = str(li_idx)
                    if num < 10:
                        str_num = '00' + str_num
                    elif num < 100:
                        str_num = '0' + str_num
                    li_df = pd.DataFrame([{'LiDAR'+str_num: li_data}])
                    total_lidar_df = pd.concat([total_lidar_df, li_df], axis=1)

            for mi_idx, mi_data in enumerate(cur_mic):
                if mi_idx == 0:
                    total_mic_df = pd.DataFrame([{'Mic000'+str(mi_idx): mi_idx}])
                else:
                    num = mi_idx
                    str_num = str(mi_idx)
                    if num < 10:
                        str_num = '000' + str_num
                    elif num < 100:
                        str_num = '00' + str_num
                    elif num < 1000:
                        str_num = '0' + str_num
                    mi_df = pd.DataFrame([{'Mic'+str_num: mi_data}])
                    total_mic_df = pd.concat([total_mic_df, mi_df], axis=1)


            temp_df = pd.DataFrame([{'now_timegap': now_time, 'cur_rgb_id': cur_rgb_id, 'cur_depth_id': cur_depth_id,
                                     'cur_hand_id': cur_hand_id, 'cur_hand_weight': cur_hand_weight , 'data_dir' : data_dir_item}])
            temp_df = pd.concat([temp_df, total_lidar_df], axis=1)
            temp_df = pd.concat([temp_df, total_mic_df], axis=1)
            if now_time >= drop_start and now_time <= drop_start + drop_duration: # drop
                label_df = pd.DataFrame([{'label': 1}])
                temp_df = pd.concat([temp_df, label_df], axis=1)
            else:                                       # normal
                label_df = pd.DataFrame([{'label': 0}])
                temp_df = pd.concat([temp_df, label_df], axis=1)


            data_df = data_df.append(temp_df, ignore_index=True)


            now_time += data_frequency

        data_df[1:].to_csv(root + '/' + folder_name + 'data_sum.csv')

    print('hsr dataset : data_sum.csv made')



def get_preprocess(data_name, data_config, root):
    data_dir = '/data_ssd/hsr_dropobject/data'
    if data_name == 'hsr_objectdrop':
        return hsr_preprocess(data_dir, data_config['folder_name'], data_config['file_name'], root)
    else:
        return None
