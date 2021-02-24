import numpy as np
import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import time
import decimal

# source_path   path of audio file
# save_path     path of mfcc csv file to be saved
# length        desirable length of wav file (sec) -> 13
# window_size   It will extract features in 'window_size' seconds -> 0.1
# stride        It will extract features per 'stride' seconds -> 0.1
def save_mfcc_from_wav(source_path, save_path, length=decimal.Decimal(0.0), window_size=0.1, stride=0.1, start_time=0):
    # load wav
    y, sr = librosa.load(source_path)

    # check if more than 'length' sec
    if len(y) < sr * length:
        print('length of wav file must be over ' + str(length) + ' seconds')

    # cut wav to exactly 'length' seconds
    length = round(length,1)
    slice_idx = round(sr*length)
    y = y[:slice_idx]

    # apply MFCC
    nfft = int(round(sr * window_size))
    hop = int(round(sr * stride))
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, n_fft=nfft, hop_length=hop) #
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)     # n_mfcc => number of features

    # reform [n_mfcc x length*(1/stride)] -> [length*(1/stride) x n_mfcc]
    mfcc = mfcc.T
    column_name = []
    # save results

    np.savetxt(save_path, mfcc, delimiter=',')

    for i in range(13):
        if i < 10:
            column_name.append('mfcc0' + str(i))
        else:
            column_name.append('mfcc' + str(i))
    df = pd.DataFrame(mfcc)
    df.columns = column_name
    return df

def calc_time_bound(df_depth, df_LiDAR, df_hand_weight, df_microphone):
    depth_timegap = df_depth['timegap']
    LiDAR_timegap = df_LiDAR['timegap']
    hand_weight_timegap = df_hand_weight['timegap']
    # microphone_timegap = df_microphone['timegap']

    start_time = max([depth_timegap.values[0], LiDAR_timegap.values[0], hand_weight_timegap.values[0]])

    start_time = round(start_time, 1)
    print('start_time : ',start_time)

    return start_time

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


def find_hand_id(now_time, df_hand, hand_id, data_frequency):
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


def hsr_preprocess(data_dir, origin_df, filenames, last_folder_df, df_count):
    item_count = 0
    for data_dir_item in filenames:
        item_count += 1
        print(item_count, df_count)
        data_df = None
        dir_name = data_dir+'/'+data_dir_item+'/data'
        df_depth = pd.read_csv(dir_name+'/depth.csv')[1:]
        df_LiDAR = pd.read_csv(dir_name+'/LiDAR.csv')[1:]
        df_hand = pd.read_csv(dir_name + '/hand.csv')[1:]
        df_hand_weight = pd.read_csv(dir_name+'/hand_weight.csv')[1:]
        df_microphone = pd.read_csv(dir_name + '/Microphone.csv')[1:]
        df_drop_time = pd.read_csv(dir_name + '/drop_time.csv')
        _, drop_end, drop_start  = df_drop_time.values[0]

        depth_id = 0
        hand_id = 0
        LiDAR_id = 0
        hand_weight_id = 0
        mic_id = 0
        data_frequency = 0.1  # 10 frames
        drop_duration = 0.5  # 0.5 sec

        start_time = calc_time_bound(df_depth, df_LiDAR, df_hand_weight, df_microphone)
        now_time = start_time
        time_cnt = decimal.Decimal(0.0)

        while now_time <= drop_start+0.5:
            time_cnt += decimal.Decimal(0.1)
            cur_depth_id, depth_id =    find_depth_id(now_time, df_depth, depth_id, data_frequency)
            cur_hand_id, hand_id =      find_hand_id(now_time, df_hand, hand_id, data_frequency)
            cur_LiDAR_data, LiDAR_id =  find_LiDAR_data(now_time, df_LiDAR, LiDAR_id, data_frequency)
            cur_hand_weight, hand_weight_id = find_hand_weight(now_time, df_hand_weight, hand_weight_id, data_frequency)
            # cur_mic, mic_id =           find_mic_data(now_time, df_microphone, mic_id, data_frequency)
            lidar_column_name = []
            for num in range(963):
                if num < 10:
                    lidar_column_name.append('LiDAR00'+str(num))
                elif num < 100:
                    lidar_column_name.append('LiDAR0'+str(num))
                else:
                    lidar_column_name.append('LiDAR' + str(num))
            lidar_df = pd.DataFrame([cur_LiDAR_data], columns=lidar_column_name)


            temp_df = pd.DataFrame([{'now_timegap': now_time, 'cur_depth_id': cur_depth_id,
                                     'cur_hand_id': cur_hand_id, 'cur_hand_weight': cur_hand_weight , 'data_dir' : data_dir_item}])
            temp_df = pd.concat([temp_df, lidar_df], axis=1)
            if now_time >= drop_start and now_time <= drop_start + drop_duration: # drop
                label_df = pd.DataFrame([{'label': 1}])
                temp_df = pd.concat([temp_df, label_df], axis=1)
            else:                                       # normal
                label_df = pd.DataFrame([{'label': 0}])
                temp_df = pd.concat([temp_df, label_df], axis=1)

            if data_df is None:
                data_df = temp_df
            else:
                data_df = data_df.append(temp_df, ignore_index=True)


            now_time += data_frequency

        sound_df = save_mfcc_from_wav(dir_name+'/sound/output.wav', dir_name+'/sound/mfcc.csv', length=time_cnt,
                                      window_size=0.1, stride=0.1, start_time=start_time)
        # mfcc concat
        data_df = pd.concat([sound_df, data_df], axis=1)
        if origin_df is None:
            origin_df = data_df
        else:
            origin_df = origin_df.append(data_df, ignore_index=True)


        origin_df.to_csv('/data_ssd/hsr_dropobject/data_sum'+str(df_count)+'.csv')
        if item_count >= 120:
            item_count = 0
            df_count += 1
            origin_df = None

        # last folder update
        temp_df = pd.DataFrame([{'name': data_dir_item}])
        if last_folder_df is None:
            last_folder_df = temp_df
        else:
            last_folder_df = last_folder_df.append(temp_df, ignore_index=True)
        last_folder_df.to_csv('/data_ssd/hsr_dropobject/last_folder.csv')

    print('hsr dataset made')

if __name__ == '__main__':
    data_dir = '/data_ssd/hsr_dropobject/data'
    # slice the folder
    filenames = os.listdir(data_dir)
    filenames.sort()
    df_count = 7

    try:
        last_folder_df = pd.read_csv('/data_ssd/hsr_dropobject/last_folder.csv')
        last_last_folder = last_folder_df['name'][len(last_folder_df)-1]
        start_idx = filenames.index(last_last_folder)
        filenames = filenames[start_idx+1:]

        origin_df = pd.read_csv('/data_ssd/hsr_dropobject/data_sum'+str(df_count)+'.csv')
    except: # when start first
        last_folder_df = None
        origin_df = None

    start = time.time()

    hsr_preprocess(data_dir, origin_df, filenames, last_folder_df, df_count)
