import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

df_rgb = pd.read_csv('/home/yjyoo/PycharmProjects/Rapp/youngjae/data/rgb.csv')[1:]
df_depth = pd.read_csv('/home/yjyoo/PycharmProjects/Rapp/youngjae/data/depth.csv')[1:]
df_LiDAR = pd.read_csv('/home/yjyoo/PycharmProjects/Rapp/youngjae/data/LiDAR.csv')[1:]
df_hand_weight = pd.read_csv('/home/yjyoo/PycharmProjects/Rapp/youngjae/data/hand_weight.csv')[1:]

depth_id = 0
LiDAR_id = 0
hand_weight_id = 0

def find_depth_id(now_timegap, next_timegap):
    global df_depth, depth_id
    for _, time, id, timegap in df_depth.values[depth_id:-1]:
        if now_timegap <= timegap and next_timegap >= timegap:
            depth_id = id - 1
            return id

def find_LiDAR_data(now_timegap, next_timegap):
    global df_LiDAR, LiDAR_id
    for _, data, time, timegap in df_LiDAR.values[LiDAR_id:-1]:
        if now_timegap <= timegap and next_timegap >= timegap:
            data = data[1: -1]
            data = data.split(',')
            LiDAR_id = _ -1
            return data

def find_hand_weight(now_timegap, next_timegap):
    global df_hand_weight, hand_weight_id
    for _, datetime, timegap, weight in df_hand_weight.values[hand_weight_id:-1]:
        if now_timegap <= timegap and next_timegap >= timegap:
            hand_weight_id = _ -1
            return weight

def get_dataset():
    global df_rgb, df_depth, df_LiDAR, df_hand_weight
    data_df = pd.DataFrame([{'now_timegap': 0, 'next_timegap': 0, 'cur_rgb_id' : 0, 'cur_depth_id' : 0, 'cur_LiDAR_data' : 0, 'cur_hand_weight' : 0}])
    # print(df_rgb.head())
    # print(df_depth.head())
    # print(df_LiDAR.head())
    # print(df_hand_weight.head())

    print('df_rgb len : ' + str(len(df_rgb)))
    print('df_depth len : ' + str(len(df_depth)))
    print('df_LiDAR len : ' + str(len(df_LiDAR)))
    print('df_hand_weight len : ' + str(len(df_hand_weight)))

    # time sync
    for _, time, id, timegap in df_rgb.values[:-1]:
        now_timegap = timegap
        next_timegap = df_rgb.values[id][3]

        cur_depth_id = find_depth_id(now_timegap, next_timegap)
        cur_LiDAR_data =find_LiDAR_data(now_timegap, next_timegap)
        cur_hand_weight = find_hand_weight(now_timegap, next_timegap)
        print(cur_depth_id, cur_hand_weight, cur_LiDAR_data)
        temp_df = pd.DataFrame([{'now_timegap': now_timegap, 'next_timegap': next_timegap, 'cur_rgb_id': id, 'cur_depth_id': cur_depth_id,
                                 'cur_LiDAR_data': cur_LiDAR_data, 'cur_hand_weight': cur_hand_weight}])
        data_df = data_df.append(temp_df, ignore_index=True)

    data_df.to_csv('data_sum.csv')

    ### rgb img
    # rgb_path = '/home/yjyoo/PycharmProjects/Rapp/youngjae/hsr_objectdrop/img/rgb/'
    # img_list = []
    # for _, time, id, timegap in df_rgb.values:
    #     if id <= 30:
    #         im = Image.open(rgb_path+str(id)+'.png')
    #         img_list.append(np.array(im))
    # rgb_image_array = np.array(img_list)
    # print('rgb img array shape : ',rgb_image_array.shape)
    #
    # ### depth img
    # depth_path = '/home/yjyoo/PycharmProjects/Rapp/youngjae/hsr_objectdrop/img/d/'
    # img_list = []
    # for _, time, id, timegap in df_depth.values:
    #     if id <= 30:
    #         im = Image.open(depth_path + str(id) + '.png')
    #         img_list.append(np.array(im))
    # depth_image_array = np.array(img_list)
    # print('depth img array shape : ', depth_image_array.shape)

    # for _, hsr_objectdrop, time, timegap in df_LiDAR.values:
    #     hsr_objectdrop = hsr_objectdrop[1 : -1]
    #     hsr_objectdrop = hsr_objectdrop.split(',')
    #     if _ == 1:
    #         print(time)
    #         print(timegap)




