import math
import sys

from geometry_msgs.msg import WrenchStamped
import rospy
import pandas as pd
import time
from datetime import datetime

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import wave
from collections import deque

import torch
from utils.data_loaders import get_input_size
import cv2



_CONNECTION_TIMEOUT = 10.0



class ForceSensorCapture(object):
    """Subscribe and hold force sensor data"""

    def __init__(self, now, maxlen):
        self._force_data_x = 0.0
        self._force_data_y = 0.0
        self._force_data_z = 0.0
        self.queue = deque(maxlen=maxlen)


        self.weight_df = pd.DataFrame([{'weight': 0, 'datetime': datetime.now()}])

        # Subscribe force torque sensor data from HSRB
        ft_sensor_topic = '/hsrb/wrist_wrench/raw'
        self._wrist_wrench_sub = rospy.Subscriber(
            ft_sensor_topic, WrenchStamped, self.__ft_sensor_cb)

        # Wait for connection
        try:
            rospy.wait_for_message(ft_sensor_topic, WrenchStamped,
                                   timeout=_CONNECTION_TIMEOUT)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

        self.starttime = now
        self.first_force_list = self.get_current_force()



    def get_current_force(self):
        return [self._force_data_x, self._force_data_y, self._force_data_z]

    def compute_difference(self, pre_data_list, post_data_list):
        if (len(pre_data_list) != len(post_data_list)):
            raise ValueError('Argument lists differ in length')
        # Calcurate square sum of difference
        square_sums = sum([math.pow(b - a, 2)
                           for (a, b) in zip(pre_data_list, post_data_list)])
        return math.sqrt(square_sums)

    def __ft_sensor_cb(self, data):
        try:
            self._force_data_x = data.wrench.force.x
            self._force_data_y = data.wrench.force.y
            self._force_data_z = data.wrench.force.z

            force_difference = self.compute_difference(self.first_force_list, self.get_current_force())
            self.weight = round((force_difference * 1000) / 9.81, 1)
            self.queue.append(self.weight)
        except:
            pass




class VisionController(object):
    def __init__(self, now, maxlen):
        self.rgb_img = None
        self.depth_img = None
        self.hand_img = None

        self.bridge = CvBridge()
        self.starttime = now

        self.hand_queue = deque(maxlen=maxlen)
        self.depth_queue = deque(maxlen=maxlen)



        depth_topic = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'
        # hand_topic = '/snu/hand_camera_image_raw'
        hand_camera_topic = '/camera/color/image_raw'


        # self._rgb_sub = rospy.Subscriber(rgb_topic, Image, self._rgb_callback)
        self._hand_sub = rospy.Subscriber(hand_camera_topic, Image, self._hand_callback)
        self._depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_callback)
        try:
            # rospy.wait_for_message(rgb_topic, CompressedImage, timeout=_CONNECTION_TIMEOUT)
            rospy.wait_for_message(hand_camera_topic, Image, timeout=_CONNECTION_TIMEOUT)
            rospy.wait_for_message(depth_topic, Image, timeout=_CONNECTION_TIMEOUT)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)


    def _depth_callback(self, data):
        self.depth_img = self.bridge.imgmsg_to_cv2(data,"32FC1")
        self.depth_img = cv2.resize(self.depth_img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
        self.depth_queue.append(self.depth_img)

    def _hand_callback(self, data):
        self.hand_img = self.bridge.imgmsg_to_cv2(data,"bgr8") #bgr8
        self.hand_img = cv2.resize(self.hand_img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
        self.hand_queue.append(self.hand_img)


    def _rgb_callback(self, data):
        self.rgb_img = self.bridge.imgmsg_to_cv2(data,"bgr8") #bgr8



class LiDARController(object):
    """Control arm and gripper"""

    def __init__(self, now, maxlen):
        self.save_mode = True
        LiDAR_topic = '/hsrb/base_scan'  # # rgb_topic = '/depthcloud_encoded/compressed'
        self.LiDAR_num = 0
        self.LiDAR_df = pd.DataFrame([{'datetime': datetime.now()}])
        self._LiDAR_sub = rospy.Subscriber(LiDAR_topic, LaserScan, self._LiDAR_callback)
        self.starttime = now
        self.queue = deque(maxlen=maxlen)

        # Wait for connection
        try:
            rospy.wait_for_message(LiDAR_topic, LaserScan, timeout=_CONNECTION_TIMEOUT)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

    def _LiDAR_callback(self, data):
        self.LiDAR_num += 1
        # print(data.ranges)
        tempdf = pd.DataFrame([{'datetime' : datetime.now(), 'data' : data.ranges, 'timegap' : time.time() - self.starttime}]) # 'data': data.ranges,
        self.LiDAR_df = self.LiDAR_df.append(tempdf, ignore_index=True)


class MicController(object):
    """Control arm and gripper"""

    def __init__(self, now, maxlen):
        self.save_mode = True
        mic_topic = '/snu/microphone_send'

        self.mic_num = 0
        self.CHANNELS = 2
        self.RATE = 44100
        self.frames = []
        self._mic_sub = rospy.Subscriber(mic_topic, String, self._mic_callback)
        self.starttime = now
        self.queue = deque(maxlen=maxlen*3)


        # Wait for connection
        try:
            rospy.wait_for_message(mic_topic, String, timeout=_CONNECTION_TIMEOUT)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

    def _mic_callback(self, data):
        self.frames.append(data.data)
        self.queue.append(data.data)


    def save(self, save_path):
        tempdf = pd.DataFrame([{'endtime': datetime.now()}])
        self.mic_df = pd.concat([self.mic_df, tempdf], axis=1)
        self.mic_df.to_csv(save_path + '/data/Microphone.csv')
        # make wave file
        WAVE_OUTPUT_FILENAME = save_path + '/data/sound/output.wav'
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(2)  # p.get_sample_size(FORMAT) == 2
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

def get_config():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument('--gpu_id', type=int, default=0)

    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--data', type=str, default='hsr_objectdrop')
    p.add_argument('--target_class', type=str, default=1)

    p.add_argument('--novelty_ratio', type=float, default=.0)

    p.add_argument('--btl_size', type=int, default=100) # 100
    p.add_argument('--n_layers', type=int, default=5) # 5

    p.add_argument('--use_rapp', action='store_true', default=True)
    p.add_argument('--start_layer_index', type=int, default=0)
    p.add_argument('--end_layer_index', type=int, default=-1)
    p.add_argument('--n_trials', type=int, default=1)
    p.add_argument('--from', type=str, default="youngjae")


    p.add_argument('--folder_name', type=str, default="hsr_objectdrop/")

    p.add_argument('--batch_size', type=int, default=36)

    p.add_argument('--sensor', type=str, default="All")  # All hand_camera force_torque head_depth mic LiDAR
    p.add_argument('--saved_name', type=str, default="datasets/All_100.pt")
    p.add_argument('--saved_data', type=str, default="All")

    p.add_argument('--object_select_mode', action='store_true', default=False)
    p.add_argument('--object_type', type=str, default="book") # cracker doll metalcup eraser cookies book plate bottle
    p.add_argument('--train_diffs', type=str, default='datasets/All_train_diffs.pt')

    config = p.parse_args()

    return config

if __name__ == '__main__':
    from model_builder import get_model
    from NoveltyDetecter import NoveltyDetecter
    from utils.data_loaders import get_realtime_dataloader
    config = get_config()

    rospy.init_node('hsr_realtime_anomaly_detection')
    now = time.time()

    maxlen = config.batch_size

    # for data stream
    force_sensor_capture = ForceSensorCapture(now, maxlen)
    vision_controller = VisionController(now, maxlen)
    lidar_controller = LiDARController(now, maxlen)
    mic_controller = MicController(now, maxlen)

    # get model
    config.input_size = get_input_size(config)
    model = get_model(config)
    print(model)
    model.load_state_dict(torch.load(config.saved_name))

    rospy.sleep(10)
    detecter = NoveltyDetecter(config)
    force_q = force_sensor_capture.queue
    hand_q = vision_controller.hand_queue
    depth_q = vision_controller.depth_queue
    mic_q = mic_controller.queue

    fusion_representation = get_realtime_dataloader(config, force_q, hand_q, depth_q, mic_q)
    score = detecter.test(
        model,
        fusion_representation,
        config
    )
    print(score)
