import gym
import rospy
import roslaunch
import time
import numpy as np
import cv2
import sys
import os
import random
from hector_uav_msgs.srv import EnableMotors
from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from geometry_msgs.msg import Twist, Pose, PoseStamped
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from gym.utils import seeding
from cv_bridge import CvBridge, CvBridgeError
import math

point = Pose()


class GazeboWorldHectorLaserCamera(gazebo_env.GazeboEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Launch the simulation with the given launchfile name
        gazebo_env.GazeboEnv.__init__(self, "/home/yari/Projects/catkin_ws/src/hector/launch/slam.launch")
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
        self.enable_motors = rospy.ServiceProxy('enable_motors', EnableMotors)

        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        self.reward_range = (-np.inf, np.inf)

        self.seed()

        self.last50actions = [0] * 50

        self.img_rows = 32
        self.img_cols = 32
        self.img_channels = 1

    def calculate_observation(self, data):
        min_range = 0.21
        done = False
        for i, item in enumerate(data.ranges):
            if (min_range > data.ranges[i] > 0):
                done = True
        return done

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        moveBindings = {
            0: (1, 0, 0, 0),
            1: (1, 0, 0, -1),
            2: (0, 0, 0, 1),
            3: (0, 0, 0, -1),
            4: (1, 0, 0, 1),
            5: (-1, 0, 0, 0),
            6: (-1, 0, 0, 1),
            7: (-1, 0, 0, -1),
            8: (1, -1, 0, 0),
            9: (1, 0, 0, 0),
            10: (0, 1, 0, 0),
            11: (0, -1, 0, 0),
            12: (1, 1, 0, 0),
            13: (-1, 0, 0, 0),
            14: (-1, -1, 0, 0),
            15: (-1, 1, 0, 0),
            16: (0, 0, 1, 0),
            17: (0, 0, -1, 0),
        }
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")
        twist = Twist()
        twist.linear.x = moveBindings[action][0]
        twist.linear.y = moveBindings[action][1]
        twist.linear.z = moveBindings[action][2]
        twist.angular.z = moveBindings[action][3]
        self.cmd_pub.publish(twist)

        '''
        if action == 0:  # FORWARD AND UP
            vel_cmd = Twist()
            vel_cmd.linear.x = 2
            vel_cmd.linear.y = 0
            vel_cmd.linear.z = 1
            vel_cmd.angular.x = 0
            vel_cmd.angular.y = 0
            vel_cmd.angular.z = 0
            self.cmd_pub.publish(vel_cmd)
        elif action == 1:  # LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.linear.z = 1
            vel_cmd.linear.y = 2
            vel_cmd.angular.x = 0
            vel_cmd.angular.y = 0
            vel_cmd.angular.z = 0
            self.cmd_pub.publish(vel_cmd) # BE91 7340 4422 9076
        elif action == 2:  # RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.linear.z = 1
            vel_cmd.linear.y = -2
            vel_cmd.angular.x = 0
            vel_cmd.angular.y = 0
            vel_cmd.angular.z = 0
            self.cmd_pub.publish(vel_cmd)
        elif action == 3:  # BACK
            vel_cmd = Twist()
            vel_cmd.linear.x = -2
            vel_cmd.linear.z = 1
            vel_cmd.linear.y = 0
            vel_cmd.angular.x = 0
            vel_cmd.angular.y = 0
            vel_cmd.angular.z = 0
            self.cmd_pub.publish(vel_cmd)
        elif action == 4:  # DOWN
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.linear.z = -2
            vel_cmd.linear.y = 0
            vel_cmd.angular.x = 0
            vel_cmd.angular.y = 0
            vel_cmd.angular.z = 0
            self.cmd_pub.publish(vel_cmd)
        elif action == 5:  # TURN LEFT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.linear.z = 0
            vel_cmd.linear.y = 0
            vel_cmd.angular.x = 0
            vel_cmd.angular.y = 0
            vel_cmd.angular.z = 2
            self.cmd_pub.publish(vel_cmd)
        elif action == 6:  # TURN RIGHT
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.linear.z = 0
            vel_cmd.linear.y = 0
            vel_cmd.angular.x = 0
            vel_cmd.angular.y = 0
            vel_cmd.angular.z = -2
            self.cmd_pub.publish(vel_cmd)
        elif action == 7:  # UP
            vel_cmd = Twist()
            vel_cmd.linear.x = 0
            vel_cmd.linear.z = 2
            vel_cmd.linear.y = 0
            vel_cmd.angular.x = 0
            vel_cmd.angular.y = 0
            vel_cmd.angular.z = 0
            self.cmd_pub.publish(vel_cmd)
        '''
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=5)
            except:
                pass

        done = self.calculate_observation(data)

        image_data = None
        success = False
        cv_image = None

        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/front_cam/camera/image', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                # temporal fix, check image is not corrupted
                if not (cv_image[h / 2, w / 2, 0] == 178 and cv_image[h / 2, w / 2, 1] == 178 and cv_image[
                    h / 2, w / 2, 2] == 178):
                    success = True
                else:
                    pass
                    # print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        self.last50actions.pop(0)  # remove oldest
        if action == 0:
            self.last50actions.append(0)
        else:
            self.last50actions.append(1)

        action_sum = sum(self.last50actions)
        '''
        # Add center of the track reward
        # len(data.ranges) = 100
        laser_len = len(data.ranges)
        tup = []
        for value in data.ranges:
            if value != float("inf"):
                tup.append(value)
        left_sum = sum(tuple(tup[laser_len - (laser_len / 5):laser_len - (laser_len / 10)])) # 80-90
        right_sum = sum(tuple(tup[(laser_len / 10):(laser_len / 5)]))  # 10-20

        center_detour = abs(right_sum - left_sum) / 5

        # 3 actions
        if not done:
            if action == 0:
                reward = 1 / float(center_detour + 1)
            elif action_sum > 45:  # L or R looping
                reward = -0.5
            else:  # L or R no looping
                reward = 0.5 / float(center_detour + 1)
        else:
            reward = -1
        
        
        x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''

        rospy.Subscriber("/pose", PoseStamped, get_location)

        # distance = math.sqrt((20 - point.position.x)**2 + (20 - point.position.y)**2)
        distance = math.sqrt(math.pow((50 - point.position.y), 2))

        reward = 50 - distance
        reward = reward * 10
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        '''
        cv2.imshow("Image window", cv_image)
        cv2.waitKey(3)
        '''

        if point.position.x > 10 or point.position.x < -10 or point.position.y < -5 or point.position.z > 20:
            reward = -1000
            print("not on path or to high!")

        # cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        # cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))
        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state, reward, done, {}

        # test STACK 4
        # cv_image = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        # self.s_t = np.append(cv_image, self.s_t[:, :3, :, :], axis=1)
        # return self.s_t, reward, done, {} # observation, reward, done, info



    def _reset(self):

        self.last50actions = [0] * 50  # used for looping avoidance

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service('/gazebo/reset_simulation')
        try:
            # reset_proxy.call()
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print ("/gazebo/reset_simulation service call failed")

        # Unpause simulation to make observation
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            # resp_pause = pause.call()
            self.unpause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/unpause_physics service call failed")

        image_data = None
        success = False
        cv_image = None
        while image_data is None or success is False:
            try:
                image_data = rospy.wait_for_message('/front_cam/camera/image', Image, timeout=5)
                h = image_data.height
                w = image_data.width
                cv_image = CvBridge().imgmsg_to_cv2(image_data, "bgr8")
                # temporal fix, check image is not corrupted
                if not (cv_image[h / 2, w / 2, 0] == 178 and cv_image[h / 2, w / 2, 1] == 178 and cv_image[
                    h / 2, w / 2, 2] == 178):
                    success = True
                else:
                    pass
                    # print("/camera/rgb/image_raw ERROR, retrying")
            except:
                pass

        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            # resp_pause = pause.call()
            self.pause()
        except (rospy.ServiceException) as e:
            print ("/gazebo/pause_physics service call failed")

        '''x_t = skimage.color.rgb2gray(cv_image)
        x_t = skimage.transform.resize(x_t,(32,32))
        x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))'''

        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image = cv2.resize(cv_image, (self.img_rows, self.img_cols))
        # cv_image = cv_image[(self.img_rows/20):self.img_rows-(self.img_rows/20),(self.img_cols/10):self.img_cols] #crop image
        # cv_image = skimage.exposure.rescale_intensity(cv_image,out_range=(0,255))

        state = cv_image.reshape(1, 1, cv_image.shape[0], cv_image.shape[1])
        return state

        # test STACK 4
        # self.s_t = np.stack((cv_image, cv_image, cv_image, cv_image), axis=0)
        # self.s_t = self.s_t.reshape(1, self.s_t.shape[0], self.s_t.shape[1], self.s_t.shape[2])
        # return self.s_t

def get_location(data):
    point.position.x, point.position.y, point.position.z = data.pose.position.x, data.pose.position.y,data.pose.position.z