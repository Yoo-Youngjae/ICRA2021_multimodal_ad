import math
import os
import sys

import actionlib
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import WrenchStamped
import rospy
import pandas as pd
import time
from datetime import datetime
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

from tmc_control_msgs.msg import (
    GripperApplyEffortAction,
    GripperApplyEffortGoal
)
from sensor_msgs.msg import LaserScan
from tmc_manipulation_msgs.srv import (
    SafeJointChange,
    SafeJointChangeRequest
)
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge

import cv2
import atexit

# for mic_controller
import numpy as np

# for base_controller
import control_msgs.msg
import controller_manager_msgs.srv
import geometry_msgs.msg
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import trajectory_msgs.msg
from hsrb_interface import Robot
from hsrb_interface import geometry

from tf import TransformListener, Transformer, transformations
from tf.transformations import quaternion_from_euler
from std_srvs.srv import Empty


# for miccontroller
import pyaudio
import wave

_CONNECTION_TIMEOUT = 10.0



class ForceSensorCapture(object):
    """Subscribe and hold force sensor data"""

    def __init__(self, now):
        self._force_data_x = 0.0
        self._force_data_y = 0.0
        self._force_data_z = 0.0
        self.save_mode = True


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
            weight = round((force_difference * 1000) / 9.81, 1)
            if self.save_mode:
                tempdf = pd.DataFrame([{'weight' : weight, 'datetime' : datetime.now(), 'timegap' : time.time() - self.starttime}])
                self.weight_df = self.weight_df.append(tempdf, ignore_index=True)
        except:
            pass

    def save(self, save_path):
        self.weight_df.to_csv(save_path+'/data/hand_weight.csv')

    def delete(self):
        self.save_mode = False



class VisionController(object):
    def __init__(self, now, save_path):
        self.save_mode = True
        self.rgb_img = None
        self.depth_img = None
        self.hand_img = None

        self.rgb_num = 0
        self.depth_num = 0
        self.hand_num = 0

        self.bridge = CvBridge()
        self.starttime = now
        self.save_path = save_path

        # rgb_topic = '/hsrb/head_rgbd_sensor/rgb/image_rect_color' # # rgb_topic = '/depthcloud_encoded/compressed'
        depth_topic = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'
        hand_topic = '/snu/hand_camera_image_raw'

        # self.rgb_df = pd.DataFrame([{'id': self.rgb_num, 'datetime': datetime.now()}])
        self.hand_df = pd.DataFrame([{'id': 0, 'datetime': datetime.now()}])
        self.depth_df = pd.DataFrame([{'id': 0, 'datetime': datetime.now()}])


        # self._rgb_sub = rospy.Subscriber(rgb_topic, Image, self._rgb_callback)
        self._hand_sub = rospy.Subscriber(hand_topic, Image, self._hand_callback)
        self._depth_sub = rospy.Subscriber(depth_topic, Image, self._depth_callback)
        try:
            # rospy.wait_for_message(rgb_topic, CompressedImage, timeout=_CONNECTION_TIMEOUT)
            rospy.wait_for_message(hand_topic, Image, timeout=_CONNECTION_TIMEOUT)
            rospy.wait_for_message(depth_topic, Image, timeout=_CONNECTION_TIMEOUT)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)


    def _rgb_callback(self, data):
        if self.save_mode:
            self.rgb_num += 1
            self.rgb_img = self.bridge.imgmsg_to_cv2(data,"bgr8") #bgr8
            cv2.imwrite(self.save_path + '/data/img/rgb/'+str(self.rgb_num)+'.png', self.rgb_img)
            tempdf = pd.DataFrame([{'id': self.rgb_num, 'datetime' : datetime.now(), 'timegap' : time.time() - self.starttime}])
            self.rgb_df = self.rgb_df.append(tempdf, ignore_index=True)


    def _depth_callback(self, data):
        if self.save_mode or time.time() - self.starttime <= 7:
            self.depth_num += 1
            self.depth_img = self.bridge.imgmsg_to_cv2(data,"32FC1")
            cv2.imwrite(self.save_path + '/data/img/d/'+str(self.depth_num)+'.png', self.depth_img)
            tempdf = pd.DataFrame([{'id': self.depth_num, 'datetime' : datetime.now(), 'timegap' : time.time() - self.starttime}])
            self.depth_df = self.depth_df.append(tempdf, ignore_index=True)

    def _hand_callback(self, data):
        if self.save_mode or time.time() - self.starttime <=7 :
            self.hand_num += 1
            self.hand_img = self.bridge.imgmsg_to_cv2(data,"bgr8") #bgr8
            cv2.imwrite(self.save_path + '/data/img/hand/'+str(self.hand_num)+'.png',self.hand_img)
            tempdf = pd.DataFrame([{'id': self.hand_num, 'datetime' : datetime.now(), 'timegap' : time.time() - self.starttime}])
            self.hand_df = self.hand_df.append(tempdf, ignore_index=True)


    def save(self, save_path):
        # self.rgb_df.to_csv(save_path+'/data/rgb.csv')
        self.depth_df.to_csv(save_path+'/data/depth.csv')
        self.hand_df.to_csv(save_path + '/data/hand.csv')

    def delete(self):
        self.save_mode = False





class JointController(object):
    """Control arm and gripper"""

    def __init__(self):
        joint_control_service = '/safe_pose_changer/change_joint'
        grasp_action = '/hsrb/gripper_controller/grasp'
        self._joint_control_client = rospy.ServiceProxy(
            joint_control_service, SafeJointChange)

        self._gripper_control_client = actionlib.SimpleActionClient(
            grasp_action, GripperApplyEffortAction)

        # Wait for connection
        try:
            self._joint_control_client.wait_for_service(
                timeout=_CONNECTION_TIMEOUT)
            if not self._gripper_control_client.wait_for_server(rospy.Duration(
                    _CONNECTION_TIMEOUT)):
                raise Exception(grasp_action + ' does not exist')
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

    def move_to_joint_positions(self, goal_joint_states):
        """Joint position control"""
        try:
            req = SafeJointChangeRequest(goal_joint_states)
            res = self._joint_control_client(req)
        except rospy.ServiceException as e:
            rospy.logerr(e)
            return False
        # return res.success

    def grasp(self, effort):
        """Gripper torque control"""
        goal = GripperApplyEffortGoal()
        goal.effort = effort

        # Send message to the action server
        if (self._gripper_control_client.send_goal_and_wait(goal) ==
                GoalStatus.SUCCEEDED):
            return True
        else:
            return False




class LiDARController(object):
    """Control arm and gripper"""

    def __init__(self, now):
        self.save_mode = True
        LiDAR_topic = '/hsrb/base_scan'  # # rgb_topic = '/depthcloud_encoded/compressed'
        self.LiDAR_num = 0
        self.LiDAR_df = pd.DataFrame([{'datetime': datetime.now()}])
        self._LiDAR_sub = rospy.Subscriber(LiDAR_topic, LaserScan, self._LiDAR_callback)
        self.starttime = now

        # Wait for connection
        try:
            rospy.wait_for_message(LiDAR_topic, LaserScan, timeout=_CONNECTION_TIMEOUT)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

    def _LiDAR_callback(self, data):
        if self.save_mode:
            self.LiDAR_num += 1
            # print(data.ranges)
            tempdf = pd.DataFrame([{'datetime' : datetime.now(), 'data' : data.ranges, 'timegap' : time.time() - self.starttime}]) # 'data': data.ranges,
            self.LiDAR_df = self.LiDAR_df.append(tempdf, ignore_index=True)

    def save(self, save_path):
        self.LiDAR_df.to_csv(save_path+'/data/LiDAR.csv')

    def delete(self):
        self.save_mode = False

class MicController(object):
    """Control arm and gripper"""

    def __init__(self, now, save_path):
        self.save_mode = True
        mic_topic = '/snu/microphone_send'
        self.save_path = save_path
        self.mic_num = 0
        self.CHANNELS = 2
        self.RATE = 44100
        self.frames = []
        self.mic_df = pd.DataFrame([{'starttime': datetime.now()}])
        self._mic_sub = rospy.Subscriber(mic_topic, String, self._mic_callback)
        self.starttime = now


        # Wait for connection
        try:
            rospy.wait_for_message(mic_topic, String, timeout=_CONNECTION_TIMEOUT)
        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

    def _mic_callback(self, data):
        if self.save_mode:
            self.frames.append(data.data)


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

    def delete(self):
        self.save_mode = False


class BaseController(object):
    def __init__(self):

        # initialize action client
        self.cli = actionlib.SimpleActionClient(
            '/hsrb/omni_base_controller/follow_joint_trajectory',
            control_msgs.msg.FollowJointTrajectoryAction)

        # wait for the action server to establish connection
        self.cli.wait_for_server()

        # make sure the controller is running
        rospy.wait_for_service('/hsrb/controller_manager/list_controllers')
        self.list_controllers = rospy.ServiceProxy(
            '/hsrb/controller_manager/list_controllers',
            controller_manager_msgs.srv.ListControllers)
        running = False
        while running is False:
            rospy.sleep(0.1)
            for c in self.list_controllers().controller:
                if c.name == 'omni_base_controller' and c.state == 'running':
                    running = True

    def send_goal(self, goal):
        self.cli.send_goal(goal)
# class HSR_Nav(object):
#     def __init__(self, params):
#         # Navigation
#         self.nav_as = actionlib.SimpleActionClient('/hsrb/impedance_control/follow_joint_trajectory_with_config', MoveBaseAction)
#         self.nav_as.wait_for_server()
#         self.transform = TransformListener()
#         self.transformer = Transformer(True, rospy.Duration(10.0))
#     # Get the current position
#     def get_current_position(self, p=np.array([0, 0, 0]), o=np.array([0, 0, 0, 1]), source='base_footprint',
#                              target='map'):
#         pp = PoseStamped()
#         pp.pose.position.x = p[0]
#         pp.pose.position.y = p[1]
#         pp.pose.position.z = p[2]
#         pp.pose.orientation.x = o[0]
#         pp.pose.orientation.y = o[1]
#         pp.pose.orientation.z = o[2]
#         pp.pose.orientation.w = o[3]
#         pp.header.frame_id = source  # 'CameraDepth_frame'
#         print('before wait')
#         self.transform.waitForTransform(target, source, time=rospy.Time(), timeout=rospy.Duration(3.0))
#         pp.header.stamp = self.transform.getLatestCommonTime(target, source)
#         print('ater wait')
#
#         result = self.transform.transformPose(target, pp)
#         print('get result')
#         result_p = np.array([result.pose.position.x, result.pose.position.y, result.pose.position.z])
#         result_o = np.array([result.pose.orientation.x, result.pose.orientation.y,
#                              result.pose.orientation.z, result.pose.orientation.w])
#         return result_p, result_o
#
#     def go_to_goal(self, x, y, theta, clear_costmap=False):  # z=yaw in angle
#         if x == 0 and y == 0: return 1
#         if clear_costmap: self.map_clear_srv()
#
#         print('[NAVI] Moving to ' + str(x) + ',' + str(y))
#         nav_goal = self.set_nav_goal(x, y, theta)
#         self.nav_as.send_goal(nav_goal)
#         self.nav_as.wait_for_result()
#         nav_res = self.nav_as.get_result()
#         nav_state = self.nav_as.get_state()
#         if nav_state == 3:
#             print("[NAVI] Arrived at " + str(x) + ',' + str(y) + ',' + str(theta))
#             return 0
#         elif nav_state == 4:
#             print("[NAVI] Failed to move to " + str(x) + ',' + str(y) + ',' + str(theta))
#             return 1
#         elif nav_state == 5:
#             print("[NAVI] Destination is not attainable!")
#             return 1
#         else:
#             return 1
#
#     # Makes the robot navigate to a relative position
#     def go_in_meter(self, x, y):
#         goal_p, goal_o = self.get_current_position(np.array([x, y, 0]))
#         goal_yaw = self.quat_to_yaw(goal_o)
#         print('[NAVI] Moving to ' + str(goal_p[0]) + ',' + str(goal_p[1]))
#         self.go_to_goal(goal_p[0], goal_p[1], goal_yaw, clear_costmap=False)
#         return None
#
#     # Transform quaternion to euler
#     def quat_to_yaw(self, quat):
#         rpy = transformations.euler_from_quaternion((quat[0], quat[1], quat[2], quat[3]))
#         yaw = rpy[2]
#         return yaw
#
#     # Set the navigation goal
#     def set_nav_goal(self, x, y, theta):
#         mb_goal = MoveBaseGoal()
#         mb_goal.target_pose.header.frame_id = '/map'  # Note: the frame_id must be map
#         mb_goal.target_pose.pose.position.x = x
#         mb_goal.target_pose.pose.position.y = y
#         mb_goal.target_pose.pose.position.z = 0.0  # z must be 0.0 (2d-map)
#
#         # Orientation of the robot is expressed in the yaw value of euler angles
#         angle = math.radians(theta)  # angles are expressed in radians
#         quat = quaternion_from_euler(0.0, 0.0, angle)  # roll, pitch, yaw
#         mb_goal.target_pose.pose.orientation = Quaternion(*quat.tolist())
#         return mb_goal


if __name__ == '__main__':
    rospy.init_node('hsrb_subsribe_and_record_data')
    robot = Robot()
    base = robot.try_get('omni_base')
    # hsr_nav = HSR_Nav(0)
    # print('hsr nav start')
    # hsr_nav.go_in_meter(1, 1)
    # print('go end')
    front_cnt = 0
    back_cnt = 0
    side_cnt = 0
    rotate_cnt = 0
    while True:
        mode = raw_input('travel type : ')
        if mode not in ['front', 'f', 'back', 'b', 'side', 's', 'rotate', 'r', 'free', 'p', 'quit', 'q']:
            print("please type in 'front', 'f', 'back', 'b', 'side', 's', 'rotate', 'r', 'free', 'p' 'quit', 'q'")
            continue
        if mode == 'quit' or mode == 'q':
            print('record end')
            break
        now = datetime.now()
        today = now.strftime("%m_%d_%H:%M:%S")
        os.makedirs('data/' + today)
        os.makedirs('data/' + today+'/data')
        os.makedirs('data/' + today + '/data/sound')
        os.makedirs('data/' + today + '/data/img')
        # os.makedirs('data/' + today + '/data/img/rgb')
        os.makedirs('data/' + today + '/data/img/d')
        os.makedirs('data/' + today + '/data/img/hand')
        save_path = 'data/' + today


        joint_controller = JointController()
        initial_position = JointState()

        initial_position.name.extend(['arm_lift_joint', 'arm_flex_joint',
                                      'arm_roll_joint', 'wrist_flex_joint',
                                      'wrist_roll_joint', 'head_pan_joint',
                                      'head_tilt_joint', 'hand_motor_joint'])
        initial_position.position.extend([0.0, 0.0, 0.0, -1.57,
                                          0.0, 0.0, -0.7, 1.2])
        joint_controller.move_to_joint_positions(initial_position)
        print('Begin grab')
        rospy.sleep(1.5)
        print(3)
        rospy.sleep(0.5)
        print(2)
        rospy.sleep(0.5)
        print(1)
        rospy.sleep(0.5)
        joint_controller.grasp(-0.1)
        rospy.sleep(3.0)
        print('start travel')

        ## data recoding start
        now = time.time()
        force_sensor_capture = ForceSensorCapture(now)
        vision_controller = VisionController(now, save_path)
        lidar_controller = LiDARController(now)
        mic_controller = MicController(now, save_path)
        # base_controller = BaseController()

        ################################## start moving
        # rospy version

        # hsrb_interface version
        if mode == 'front' or mode == 'f':
            front_cnt += 1
            try:
                base.go_rel(2.0, 0.0, 0.0, 5.0)
            except:
                rospy.logerr('Fail go')
        elif mode == 'back' or mode == 'b':
            back_cnt += 1
            for i in range(50):
                try:
                    base.go_rel(-0.5, 0.0, 0.0, 0.1)
                except:
                    continue
        elif mode == 'side' or mode == 's':
            side_cnt += 1
            for i in range(50):
                try:
                    base.go_rel(0.0,1.0, 0.0, 0.1)
                except:
                    continue
        elif mode == 'rotate' or mode == 'r':
            rotate_cnt += 1
            for i in range(50):
                try:
                    base.go_rel(0.0, 0.0, 2.0, 0.1)
                except:
                    continue
        else:   # free
            rospy.sleep(5)

        ################################### reach end
        print('drop')
        ################################# Drop


        initial_position = JointState()
        initial_position.name.extend(['arm_lift_joint', 'arm_flex_joint',
                                      'arm_roll_joint', 'wrist_flex_joint',
                                      'wrist_roll_joint', 'head_pan_joint',
                                      'head_tilt_joint', 'hand_motor_joint'])
        initial_position.position.extend([0.0, 0.0, 0.0, -1.57,
                                          0.0, 0.0, -0.7, 0.2])
        drop_start = (time.time() - now) + 0.3
        print('drop time',drop_start)
        joint_controller.move_to_joint_positions(initial_position)
        drop_end = time.time() - now
        print('go back')
        ################################## drop end
        force_sensor_capture.save(save_path)
        vision_controller.save(save_path)
        lidar_controller.save(save_path)
        mic_controller.save(save_path)
        force_sensor_capture.delete()
        lidar_controller.delete()
        mic_controller.delete()
        vision_controller.delete()
        ## start moving


        # hsrb_interface version
        if mode == 'front' or mode == 'f':
            for i in range(50):
                try:
                    base.go_rel(-0.5, 0.0, 0.0, 0.1)
                except:
                    continue
        elif mode == 'back' or mode == 'b':
            try:
                base.go_rel(2.0, 0.0, 0.0, 5.0)
            except:
                rospy.logerr('Fail go')
        elif mode == 'side' or mode == 's':
            for i in range(50):
                try:
                    base.go_rel(0.0, -1.0, 0.0, 0.1)
                except:
                    continue
        elif mode == 'rotate' or mode == 'r':
            for i in range(50):
                try:
                    base.go_rel(0.0, 0.0, 2.0, 0.1)
                except:
                    continue
        else: # free
            rospy.sleep(5)

        ################################### start moving
        # before exit, save dataframes

        drop_df = pd.DataFrame([{'drop_start': drop_start, 'drop_end': drop_end}])
        drop_df.to_csv(save_path+'/data/drop_time.csv')
        initial_position.position.extend([0.0, 0.0, 0.0, 0,
                                          0.0, 0.0, 0.0, 1.2])


        print('[STATUS] f : ',front_cnt,' b :',back_cnt,' s :',side_cnt,' r :',rotate_cnt)
        # del base_controller



