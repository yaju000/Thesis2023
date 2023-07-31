#!/usr/bin/env python

import math
import numpy as np
import rospy
import matplotlib.pyplot as plt
from pandas import DataFrame
from mavros_msgs.msg import State
from sensor_msgs.msg import Imu, NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from mavros_msgs.msg import PositionTarget
from mavros_msgs.srv import SetMode
from mavros_msgs.srv import ParamSet
from std_msgs.msg import Float64
from math import *
from datetime import datetime
import time
import heapq

from AOA_v2 import AOA
from LeastQ_v1 import least_square
from hsv_fw import hsv

AOA = AOA()
LQ = least_square()
hsv = hsv()

enu_pos = []
azimuth = []
azimuth_deg = []
elevation = []
elevation_deg = []
ob_points = []
est_vector = []
est_list = []
P_img_list = []
uav_pose = []
RMSE_list = []
GDOP_list = []

class aoa_info(object):
    def __init__(self):
        self.imu_msg = Imu()
        self.gps_msg = Odometry()

        self.gps_pose = [0,0,0]
        self.ned_pose = [0,0,0]
        self.imu_x = 0
        self.lamda = 0
        self.roll, self.pitch, self.yaw = 0,0,0
        self.vision_roll, self.vision_pitch, self.vision_yaw = 0, 0, 0
        self.hd_deg = 0
        self.quat = [0,0,0,0]
        self.u, self.v = 0, 0
        self.u_u, self.v_v = 0, 0
        self.P_img_x, self.P_img_y, self.P_img_z = 0, 0, 0
        self.angle_a_w = 0
        self.angle_e_w = 0
        self.angle_a = [0, 0]
        self.angle_e = [0, 0]
        self.pos_hat = [0, 0]
        self.dd = 0

        self.est_position = [0, 0, 0]
        self.ob_point = [0, 0, 0]
        self.est_n, self.est_e, self.est_d = 0, 0, 0
        self.vector_n, self.vector_e, self.vector_d = 0, 0, 0
        self.last_req = rospy.Time.now()

        rospy.Subscriber("/plane_cam_0/mavros/imu/data", Imu, self.imu_callback)
        rospy.Subscriber("/plane_cam_0/mavros/global_position/local", Odometry, self.gps_callback) 
        rospy.Subscriber("/plane_cam_0/mavros/global_position/compass_hdg", Float64, self.hdg_callback)   

    def gps_callback(self, msg):
        self.gps_msg = msg
        self.gps_pose[0] = msg.pose.pose.position.x
        self.gps_pose[1] = msg.pose.pose.position.y
        self.gps_pose[2] = msg.pose.pose.position.z

        self.ned_pose[0], self.ned_pose[1], self.ned_pose[2] = self.ENU_to_NED(self.gps_pose[0], self.gps_pose[1], self.gps_pose[2])   
        enu_pos.append([self.gps_pose[0], self.gps_pose[1], self.gps_pose[2]])

    def imu_callback(self, msg):
        self.imu_msg = msg
        self.quat[0] = msg.orientation.w
        self.quat[1] = msg.orientation.x
        self.quat[2] = msg.orientation.y
        self.quat[3] = msg.orientation.z

        ## ENU ##
        self.roll, self.pitch, self.yaw = self.euler_from_quaternion(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        
        yaw = self.yaw

        if 0 < yaw:
            if (np.pi/2) >=yaw:
                psi = np.pi/2-yaw
                # print('1')
            else:
                psi = -(yaw-np.pi/2)+2*np.pi
                # print('4')

        elif yaw < 0:
            if (-np.pi/2) <=yaw:
                # print('2')
                psi = -yaw+np.pi/2
            else:
                # print('3')
                psi = -yaw-3*np.pi/2+2*np.pi
        else:
            psi = yaw

        # print('yaw',self.yaw)
        # print('psi =',psi*180/np.pi)
        # print('roll =',self.roll)
        # print('pitch =',self.pitch)
        # print('yaw =',self.yaw)

    def hdg_callback(self, msg): #enu # pi~(-pi)
        self.hdg_msg = msg
        heading_angle = msg.data #ned
        self.hd_deg = heading_angle #ned # degree
        self.heading = 90-heading_angle 

        if self.heading <= -180:
            self.heading = self.heading + 360
        else:
            self.heading = self.heading

        self.heading = np.deg2rad(self.heading) #enu

        # print('heading angle(deg) = ')
        # print(heading_angle)
        # print('heading angle(rad) = ')
        # print(self.heading)

    def ENU_to_NED(self, x, y, z):
  
        R = [[0, 1, 0],[1, 0, 0],[0, 0, -1]]
        q = [x, y, z]
        ned = np.matmul(R,q)
        a = ned[0]
        b = ned[1]
        c = ned[2]
      
        return a, b, c

    def NED_to_ENU(self, x, y, z):
  
        R = [[0, 1, 0],[1, 0, 0],[0, 0, -1]]
        q = [x, y, z]
        ned = np.matmul(R,q)
        a = ned[0]
        b = ned[1]
        c = ned[2]
      
        return a, b, c

    def cal_aoa_info(self):

        if [self.u, self.v]!=[None, None]:

            ##position_vector
            size_u = 640
            size_v = 360
            u_0 = size_u/2
            v_0 = size_v/2
            # focal length
            f = 277.191356
          
            ## 1 ##
            self.P_img_x = u_0 - self.u
            self.P_img_y = v_0 - self.v
            self.P_img_z = f

            ## 2 ##
            # self.P_img_x = v_0 - self.v
            # self.P_img_y = self.u - u_0 
            # self.P_img_z = f

            P_img = [self.P_img_x, self.P_img_y, self.P_img_z]
            # print("u, v = ")
            # print(self.u, self.v)
            # print('------------')
            # print("P_img = ")
            # print(P_img)

            ## 1 ##
            self.angle_a_w, self.angle_e_w, self.angle_a, self.angle_e, self.est_n, self.est_e, self.est_d, self.vector_n, self.vector_e, self.vector_d, self.dd = AOA.AOA_v2(self.ned_pose[0], self.ned_pose[1], self.ned_pose[2], self.roll, self.pitch, self.yaw, self.P_img_x, self.P_img_y, self.P_img_z)
            
        else:
            pass

    def euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def cal_dop(self, a, d):  
        # a = ob_points
        # b = next ob_point 
        # d = estimated position
        # print('--------------------')
        H = []

        Node_number = len(a)

        for i in range(Node_number):

            r = np.square(d[0]-a[i][0])+np.square(d[1]-a[i][1])
            H_i = [-(d[1]-a[i][1])/r, (d[0]-a[i][0])/r] #1*2
            H.append(H_i)

        QQ = np.matmul(np.transpose(H), H) #2*2
        Q = np.linalg.inv(QQ)
        GDOP = np.sqrt(np.trace(Q))
        #print("GDOP = ",GDOP)
        #print('--------------------')
        return GDOP

    def iteration(self, event):

        self.u, self.v, self.bb_area = hsv.value_callback()
        self.cal_aoa_info()
        ob_point = [self.ned_pose[0], self.ned_pose[1], self.ned_pose[2]]

        angle_a_w = round(self.angle_a_w,2)
        angle_e_w = round(self.angle_e_w,2)
        
        # if angle_a_w >3.14 :
        #     angle_a_w = angle_a_w - 3.14*2
        # elif angle_a_w < -1.57:
        #     angle_a_w = 3.14*2 + angle_a_w

        print('Get!')
        azimuth.append(angle_a_w)
        azimuth_deg.append(angle_a_w*180/np.pi)
        elevation.append(angle_e_w)
        elevation_deg.append(angle_e_w*180/np.pi)
        ob_points.append(ob_point)
        est_vector.append([self.vector_n, self.vector_e, self.vector_d])
        uav_pose.append([self.roll, self.pitch, self.yaw])

        # if  len(azimuth) >= 2:
        #     Est_n,Est_e,Est_d = LQ.LeastQ_m(ob_points, azimuth)
        #     Est_position = [Est_n,Est_e,Est_d]
        #     print('Est_n,Est_e,Est_d = ')
        #     print(Est_position)
        #     est_list.append(Est_position)
        #     error = np.sqrt(np.square(Est_n)+np.square(Est_e-50))
        #     print('error =', error)
        #     RMSE_list.append(error)
        #     GDOP = self.cal_dop(ob_points, [est_list[-1][0],est_list[-1][1],0])
        #     GDOP_list.append(GDOP)
        #     print('GDOP =', GDOP)


if __name__ == '__main__':
    rospy.init_node('aoa_info_only', anonymous=True)
    dt = 1.0/20
    pathplan_run = aoa_info()
    rospy.Timer(rospy.Duration(dt), pathplan_run.iteration)
    rospy.spin()


    df = DataFrame({'enu_pos': enu_pos})
    df.to_excel('fw_path.xlsx', sheet_name='sheet1', index=False)
    dq = DataFrame({'azimuth':azimuth,'azimuth_deg':azimuth_deg,'elevation':elevation,'elevation_deg':elevation_deg})
    dq.to_excel('fw_measurement.xlsx', sheet_name='sheet1', index=False)
    dt = DataFrame({'ob_points': ob_points,'est_vector':est_vector,'uav_pose':uav_pose})
    dt.to_excel('fw_ob.xlsx', sheet_name='sheet1', index=False)
    # dp = DataFrame({'est_position':est_list})
    # dp.to_excel('fw_est.xlsx', sheet_name='sheet1', index=False)

