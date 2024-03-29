import rospy
import pandas as pd
import time
import math
import os
from sensor_msgs.msg import Image, CompressedImage, PointCloud2, JointState
from cv_bridge import CvBridge
from geometry_msgs.msg import WrenchStamped
from datetime import datetime
import sys
import cv2
from hsrb_interface import Robot
import trajectory_msgs.msg
from tmc_manipulation_msgs.srv import (
    SafeJointChange,
    SafeJointChangeRequest
)

_CONNECTION_TIMEOUT = 10.0

class VisionController(object):
    def __init__(self, start_time, save_path):
        self.start_time = start_time
        self.save_path = save_path
        self.save_mode = True
        self.bridge = CvBridge()

        self.rgb_img = None
        self.depth_img = None

        self.rgb_num = 0
        self.depth_num = 0

        self.head_rgb_num = 0
        self.head_depth_num = 0

        self.hand_df = pd.DataFrame([{'id': 0, 'datetime': datetime.now()}])
        self.depth_df = pd.DataFrame([{'id': 0, 'datetime': datetime.now()}])

        self.head_rgb_df = pd.DataFrame([{'id': 0, 'datetime': datetime.now()}])
        self.head_depth_df = pd.DataFrame([{'id': 0, 'datetime': datetime.now()}])


        hand_rgb_topic = '/camera/color/image_raw'
        hand_depth_topic = '/camera/depth/image_rect_raw'

        head_rgb_topic = '/hsrb/head_rgbd_sensor/rgb/image_rect_color'
        head_depth_topic = '/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw'


        self._rgb_sub = rospy.Subscriber(hand_rgb_topic, Image, self._rgb_callback)
        self._depth_sub = rospy.Subscriber(hand_depth_topic, Image, self._depth_callback)
        self.head_rgb_sub = rospy.Subscriber(head_rgb_topic, Image, self.head_rgb_callback)
        self.head_depth_sub = rospy.Subscriber(head_depth_topic, Image, self.head_depth_callback)


    def _rgb_callback(self, img_msg):
        if self.save_mode:
            self.rgb_num += 1
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            rgb_img = cv_image
            cv2.imwrite(self.save_path + '/data/img/hand/' + str(self.rgb_num) + '.png', rgb_img)
            tempdf = pd.DataFrame(
                [{'id': self.rgb_num, 'datetime': datetime.now(), 'timegap': time.time() - self.start_time}])
            self.hand_df = self.hand_df.append(tempdf, ignore_index=True)

    def _depth_callback(self, data):
        if self.save_mode:
            self.depth_num += 1
            depth_img = self.bridge.imgmsg_to_cv2(data, "32FC1")
            cv2.imwrite(self.save_path + '/data/img/d/' + str(self.depth_num) + '.png', depth_img)
            tempdf = pd.DataFrame(
                [{'id': self.depth_num, 'datetime': datetime.now(), 'timegap': time.time() - self.start_time}])
            self.depth_df = self.depth_df.append(tempdf, ignore_index=True)


    def head_rgb_callback(self, img_msg):
        if self.save_mode:
            self.head_rgb_num += 1
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "passthrough")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            rgb_img = cv_image
            cv2.imwrite(self.save_path + '/data/img/head_rgb/' + str(self.head_rgb_num) + '.png', rgb_img)
            tempdf = pd.DataFrame(
                [{'id': self.head_rgb_num, 'datetime': datetime.now(), 'timegap': time.time() - self.start_time}])
            self.head_rgb_df = self.head_rgb_df.append(tempdf, ignore_index=True)

    def head_depth_callback(self, data):
        if self.save_mode:
            self.head_depth_num += 1
            depth_img = self.bridge.imgmsg_to_cv2(data, "32FC1")
            cv2.imwrite(self.save_path + '/data/img/head_d/' + str(self.head_depth_num) + '.png', depth_img)
            tempdf = pd.DataFrame(
                [{'id': self.head_depth_num, 'datetime': datetime.now(), 'timegap': time.time() - self.start_time}])
            self.head_depth_df = self.head_depth_df.append(tempdf, ignore_index=True)


    def save(self, save_path):
        self.depth_df[1:].to_csv(save_path+'/data/depth.csv')
        self.hand_df[1:].to_csv(save_path + '/data/hand.csv')
        self.head_rgb_df[1:].to_csv(save_path + '/data/head_rgb.csv')
        self.head_depth_df[1:].to_csv(save_path + '/data/head_depth.csv')
        self.save_mode = False

class ForceSensorController(object):
    """Subscribe and hold force sensor data"""

    def __init__(self, start_time):
        self.start_time = start_time
        self.save_mode = True
        self.init_force = None
        self.cur_force = None

        self.weight_df = pd.DataFrame([{'weight': 0, 'datetime': datetime.now()}])

        ft_sensor_topic = '/hsrb/wrist_wrench/raw'
        self._wrist_wrench_sub = rospy.Subscriber(ft_sensor_topic, WrenchStamped, self.__ft_sensor_cb)

    def get_current_weight(self):
        force_difference = self.compute_difference(self.init_force, self.cur_force)
        weight = round((force_difference * 1000) / 9.81, 1)
        return weight

    def compute_difference(self, pre_data_list, post_data_list):
        if (len(pre_data_list) != len(post_data_list)):
            raise ValueError('Argument lists differ in length')
        # Calcurate square sum of difference
        square_sums = sum([math.pow(b - a, 2) for (a, b) in zip(pre_data_list, post_data_list)])
        return math.sqrt(square_sums)

    def __ft_sensor_cb(self, data):
        if self.init_force is None:
            self._force_data_x = data.wrench.force.x
            self._force_data_y = data.wrench.force.y
            self._force_data_z = data.wrench.force.z
            self.init_force = [data.wrench.force.x, data.wrench.force.y, data.wrench.force.z]
        self.cur_force = [data.wrench.force.x, data.wrench.force.y, data.wrench.force.z]
        if self.save_mode:
            weight = self.get_current_weight()
            tempdf = pd.DataFrame(
                [{'weight': weight, 'datetime': datetime.now(), 'timegap': time.time() - self.start_time}])
            self.weight_df = self.weight_df.append(tempdf, ignore_index=True)

    def save(self, save_path):
        self.weight_df.to_csv(save_path+'/data/hand_weight.csv')
        self.save_mode = False

class JointController(object):
    """Control arm and gripper"""

    def __init__(self):
        joint_control_service = '/safe_pose_changer/change_joint'
        grasp_action = '/hsrb/gripper_controller/grasp'
        self._joint_control_client = rospy.ServiceProxy(joint_control_service, SafeJointChange)
        self.gripper_pub = rospy.Publisher('/hsrb/gripper_controller/command', trajectory_msgs.msg.JointTrajectory, queue_size=10)



        # Wait for connection
        try:
            self._joint_control_client.wait_for_service(
                timeout=_CONNECTION_TIMEOUT)

        except Exception as e:
            rospy.logerr(e)
            sys.exit(1)

    def move_to_joint_positions(self, goal_pose):
        if goal_pose == 'initial_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint'])
            goal_joint_states.position.extend([-2.4, 0, 0.7])
        elif goal_pose == 'head_down_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['head_pan_joint', 'head_tilt_joint'])
            goal_joint_states.position.extend([0, -0.6])
        elif goal_pose == 'go_to_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['arm_flex_joint', 'wrist_flex_joint'])
            goal_joint_states.position.extend([0, -1.57])
        elif goal_pose == 'place_position':
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['arm_flex_joint', 'wrist_flex_joint'])
            goal_joint_states.position.extend([-1.2, -0.4])
        else:
            goal_joint_states = JointState()
            goal_joint_states.name.extend(['arm_flex_joint', 'wrist_flex_joint'])
            goal_joint_states.position.extend([-1.2, -0.4])


        """Joint position control"""
        try:
            req = SafeJointChangeRequest(goal_joint_states)
            res = self._joint_control_client(req)
        except rospy.ServiceException as e:
            rospy.logerr(e)
            return False
        return res.success

    def grasp(self, position):
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = ["hand_motor_joint"]
        p = trajectory_msgs.msg.JointTrajectoryPoint()
        p.positions = [position]
        p.velocities = [0]
        p.effort = [0.1]
        p.time_from_start = rospy.Time(3)
        traj.points = [p]

        self.gripper_pub.publish(traj)


class GripperDegreeController(object):
    """Control arm and gripper"""

    def __init__(self, start_time):
        self.start_time = start_time
        self.save_mode = True

        joint_states_topic = '/hsrb/robot_state/joint_states'

        self.joint_states_sub = rospy.Subscriber(joint_states_topic, JointState, self._joint_state_callback)
        self.gripper_df = pd.DataFrame([{'gripper_radian': 0, 'datetime': datetime.now()}])

    def _joint_state_callback(self, data):
        gripper_joints = []
        for i, name in enumerate(data.name):
            if name in ["hand_l_proximal_joint", "hand_l_spring_proximal_joint",
                        "hand_l_distal_joint", "hand_l_mimic_distal_joint",
                        "hand_r_proximal_joint", "hand_r_spring_proximal_joint",
                        "hand_r_distal_joint", "hand_r_mimic_distal_joint"]:
                gripper_joints.append(data.position[i])
        if self.save_mode:
            tempdf = pd.DataFrame(
                [{'gripper_radian': gripper_joints, 'datetime': datetime.now(), 'timegap': time.time() - self.start_time}])
            self.gripper_df = self.gripper_df.append(tempdf, ignore_index=True)

    def save(self, save_path):
        self.gripper_df[1:].to_csv(save_path+'/data/gripper_degree.csv')
        self.save_mode = False

def make_dir(obj_name):
    now = datetime.now()
    today = now.strftime("%m_%d_%H:%M:%S") + '_'+obj_name
    os.makedirs('data/' + today)
    os.makedirs('data/' + today + '/data')
    # os.makedirs('data/' + today + '/data/sound')
    os.makedirs('data/' + today + '/data/img')
    # os.makedirs('data/' + today + '/data/img/rgb')
    os.makedirs('data/' + today + '/data/img/d')
    os.makedirs('data/' + today + '/data/img/hand')
    os.makedirs('data/' + today + '/data/img/head_rgb')
    os.makedirs('data/' + today + '/data/img/head_d')
    save_path = 'data/' + today
    return save_path


box_position = [-0.136, -2.515, -0.512]
if __name__ == '__main__':
    # start setup
    rospy.init_node('hsr_tidyup_main')

    # hsr python api
    robot = Robot()
    tts = robot.try_get('default_tts')
    tts.language = tts.ENGLISH
    omni_base = robot.try_get('omni_base')
    whole_body = robot.try_get('whole_body')
    gripper = robot.try_get('gripper')
    joint_controller = JointController()

    while True:
        # 1. initial pose
        print('1. initial pose')
        joint_controller.move_to_joint_positions('initial_position')
        joint_controller.grasp(1.0)

        # 2. pick
        print('2. start pick')
        while True:
            is_picked = raw_input('Press object name : ')
            if is_picked == 'q':
                print('terminate')
                sys.exit()
            else:
                obj_name = is_picked
                break

        save_path = make_dir(obj_name)
        start_time = time.time()
        gripper_degree_controller = GripperDegreeController(start_time)
        vision_controller = VisionController(start_time, save_path)
        force_sensor_controller = ForceSensorController(start_time)

        gripper.apply_force(1.0) # pick

        # 3. go_to pose
        print('3. go to pose')
        #whole_body.move_to_joint_positions({'head_pan_joint' : 0, 'head_tilt_joint' : -0.5})
        joint_controller.move_to_joint_positions('head_down_position')
        joint_controller.move_to_joint_positions('go_to_position')

        # 4. go to box
        print('4. go to box')
        while True:
            is_picked = raw_input('Press any key if place the object into the box : ')
            if is_picked == 'q':
                print('terminate')
                sys.exit()
            else:
                break

        # 5. place the object
        print('5. place the object')
        joint_controller.move_to_joint_positions('place_position')
        rospy.sleep(2)
        # save the csv before place
        vision_controller.save(save_path)
        force_sensor_controller.save(save_path)
        gripper_degree_controller.save(save_path)
        rospy.sleep(3)
        rospy.sleep(1)
        joint_controller.grasp(1.0)


        # 6. go to pose
        joint_controller.move_to_joint_positions('go_to_position')

        while True:
            is_picked = raw_input('Press any key if you continue : ')
            if is_picked == 'q':
                print('terminate')
                sys.exit()
            else:
                break