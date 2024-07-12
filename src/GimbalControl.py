#!/usr/bin/env python3

import rospy
import numpy as np
import os
from geometry_msgs.msg import Vector3Stamped, QuaternionStamped
from dji_osdk_ros.msg import Gimbal
from PID import PID

def enu2ned(vec):
    T = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    return T @ vec

def quaternion2euler(q):
    [q0, q1, q2, q3] = q
    roll = np.arctan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    pitch = np.arcsin(2 * (q0 * q2 - q3 * q1))
    yaw = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return np.array([roll, pitch, yaw])

def euler2quaternion(attitudeAngle):
    quaternion = [0] * 4
    cosHalfRoll = np.cos(attitudeAngle[0] / 2)
    cosHalfPitch = np.cos(attitudeAngle[1] / 2)
    cosHalfYaw = np.cos(attitudeAngle[2] / 2)
    sinHalfRoll = np.sin(attitudeAngle[0] / 2)
    sinHalfPitch = np.sin(attitudeAngle[1] / 2)
    sinHalfYaw = np.sin(attitudeAngle[2] / 2)

    quaternion[0] = cosHalfRoll * cosHalfPitch * cosHalfYaw + sinHalfRoll * sinHalfPitch * sinHalfYaw
    quaternion[1] = sinHalfRoll * cosHalfPitch * cosHalfYaw - cosHalfRoll * sinHalfPitch * sinHalfYaw
    quaternion[2] = cosHalfRoll * sinHalfPitch * cosHalfYaw + sinHalfRoll * cosHalfPitch * sinHalfYaw
    quaternion[3] = cosHalfRoll * cosHalfPitch * sinHalfYaw - sinHalfRoll * sinHalfPitch * cosHalfYaw
    return quaternion

def rpyENU2NED(rpyRadENU):
    return np.array([rpyRadENU[0], -rpyRadENU[1], yawRadENU2NED(rpyRadENU[2])])

def yawRadENU2NED(yawRadENU):
    return (np.pi / 2 - yawRadENU) % (2 * np.pi)

def rpyRadString(rpyRad):
    rpyDeg = 180 / np.pi * rpyRad
    return f'(roll: {rpyDeg[0]:.2f}, pitch: {rpyDeg[1]:.2f}, yaw: {rpyDeg[2]:.2f})'

def rpyDegString(rpyDeg):
    return f'(roll: {rpyDeg[0]:.2f}, pitch: {rpyDeg[1]:.2f}, yaw: {rpyDeg[2]:.2f})'

def vector2rpy(vector):
    return np.array([vector.x, vector.y, vector.z])

def rpyVectorRadString(rpyVectorRad):
    rpyDeg = 180 / np.pi * vector2rpy(rpyVectorRad)
    return f'(roll: {rpyDeg[0]:.2f}, pitch: {rpyDeg[1]:.2f}, yaw: {rpyDeg[2]:.2f})'

def rpyVectorDegString(rpyVectorDeg):
    rpyDeg = vector2rpy(rpyVectorDeg)
    return f'(roll: {rpyDeg[0]:.2f}, pitch: {rpyDeg[1]:.2f}, yaw: {rpyDeg[2]:.2f})'

def gimbalAngleString(gimbalAngleRad: Gimbal):
    rpyRad = np.array([gimbalAngleRad.roll, gimbalAngleRad.pitch, gimbalAngleRad.yaw])
    return rpyRadString(rpyRad)

def vectorString(vector):
    return f'({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f})'

def normalizeRad(angleRad):
    return np.arctan2(np.sin(angleRad), np.cos(angleRad))

class GimbalControlNode:
    def __init__(self, uav_name):
        self.mode = rospy.get_param("mode", "stabilize")
        
        guidanceLength = rospy.get_param("guidanceLength")
        takeoffHeight = rospy.get_param("takeoffHeight")

        targetHeight = rospy.get_param("targetHeight")

        self.cameraPitchENUDeg = np.degrees(np.arctan((takeoffHeight - targetHeight) / guidanceLength))
        self.gimbalTargetRPYNEDDeg = np.array([0.0, -self.cameraPitchENUDeg, 0.0])

        self.uav_name = uav_name

        self.current_atti = None
        self.meRPYRadENU = None
        self.meRPYRadNED = None
        self.initYawNEDDeg = None
        self.meGimbalRPYNEDRad = None
        self.meGimbalRPYNEDDeg = None

        self.init_start_time = rospy.Time.now()
        self.initialized = False
        self.elapsed_time = 0

        self.rpySpeedPID = [
            PID(kp=1, ki=0, kd=0.0, intMax=np.pi, intMin=-np.pi),
            PID(kp=1, ki=0, kd=0.0, intMax=np.pi, intMin=-np.pi),
            PID(kp=1, ki=0, kd=0.0, intMax=np.pi, intMin=-np.pi),
        ]

        print(f"Initializing GimbalControlNode for UAV: {uav_name} in {self.mode} mode")

        rospy.Subscriber(uav_name + "/dji_osdk_ros/attitude", QuaternionStamped, self.attitude_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/gimbal_angle", Vector3Stamped, self.gimbal_angle_callback)
        if self.mode == "stabilize":
            rospy.Subscriber(uav_name + "/control_angle", Vector3Stamped, self.inertial_control_angle_callback)
        
        self.gimbal_angle_cmd_publisher = rospy.Publisher(uav_name + "/dji_osdk_ros/gimbal_angle_cmd", Gimbal, queue_size=10)
        self.gimbal_speed_cmd_publisher = rospy.Publisher(uav_name + "/dji_osdk_ros/gimbal_speed_cmd", Vector3Stamped, queue_size=10)
        
        rospy.Timer(rospy.Duration(0.1), self.main_loop)

    def attitude_callback(self, msg):
        self.current_atti = msg
        q = [msg.quaternion.w, msg.quaternion.x, msg.quaternion.y, msg.quaternion.z]
        self.meRPYRadENU = quaternion2euler(q)
        self.meRPYRadNED = normalizeRad(rpyENU2NED(self.meRPYRadENU))

    def gimbal_angle_callback(self, msg):
        self.meGimbalRPYNEDDeg = np.array([msg.vector.y, msg.vector.x, msg.vector.z])
        self.meGimbalRPYNEDRad = np.radians(self.meGimbalRPYNEDDeg)
        if self.initYawNEDDeg is None and self.initialized:
            self.initYawNEDDeg = msg.vector.z
            print(f"Initial Gimbal Yaw set to: {self.initYawNEDDeg:.2f} degs")

    def inertial_control_angle_callback(self, msg: Vector3Stamped):
        self.gimbalTargetRPYNEDDeg = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def initialize_gimbal(self):
        init_gimbal_angle = Gimbal()
        init_gimbal_angle.header.stamp = rospy.Time.now()
        init_gimbal_angle.ts = 2
        init_gimbal_angle.mode = 1
        init_gimbal_angle.roll = 0
        init_gimbal_angle.pitch = 0
        init_gimbal_angle.yaw = 0
        self.gimbal_angle_cmd_publisher.publish(init_gimbal_angle)
        print("Publishing gimbal 0, 0, 0 degrees")

    def publish_track_gimbal_angle(self):
        gimbal_angle = Gimbal()
        gimbal_angle.header.stamp = rospy.Time.now()
        gimbal_angle.ts = 1
        gimbal_angle.mode = 1

        if self.meRPYRadNED is None or self.initYawNEDDeg is None:
            return
        roll, pitch, yaw = self.meRPYRadNED
        initYawNEDRad = np.radians(self.initYawNEDDeg)
        gimbal_angle.roll = normalizeRad(pitch)
        gimbal_angle.pitch = normalizeRad(roll)
        gimbal_angle.yaw = normalizeRad(yaw - initYawNEDRad)
        print(f'Expected gimbal at {gimbalAngleString(gimbal_angle)}')

        self.gimbal_angle_cmd_publisher.publish(gimbal_angle)

    def publish_track_gimbal_speed(self):
        gimbal_angle = Vector3Stamped()
        gimbal_angle.header.stamp = rospy.Time.now()

        if self.meRPYRadNED is None or self.initYawNEDDeg is None:
            return
        
        expectedRPYRadNED = self.meRPYRadNED + np.radians(self.gimbalTargetRPYNEDDeg)
        nowRPYRadNED = self.meGimbalRPYNEDRad

        speed = normalizeRad(expectedRPYRadNED - nowRPYRadNED)
        
        print(f'Expected gimbal RPY at {rpyRadString(expectedRPYRadNED)}')
        print(f'Expected gimbal speed at {vectorString(np.degrees(speed))}')

        speed = [self.rpySpeedPID[i].compute(speed[i]) for i in range(3)]

        gimbal_angle.vector.x = speed[0]
        gimbal_angle.vector.y = speed[1]
        gimbal_angle.vector.z = speed[2]

        self.gimbal_speed_cmd_publisher.publish(gimbal_angle)

    def publish_stablize_gimbal_angle(self):
        gimbal_angle = Gimbal()
        gimbal_angle.header.stamp = rospy.Time.now()
        gimbal_angle.ts = 1
        gimbal_angle.mode = 1

        if self.gimbalTargetRPYNEDDeg is None or self.initYawNEDDeg is None:
            return
        gimbalTargetRPYNEDRad = np.radians(self.gimbalTargetRPYNEDDeg)
        gimbal_angle.roll = normalizeRad(gimbalTargetRPYNEDRad[0]) 
        gimbal_angle.pitch = normalizeRad(gimbalTargetRPYNEDRad[1]) 
        gimbal_angle.yaw = normalizeRad(gimbalTargetRPYNEDRad[2]) 
        print(f'Expected gimbal at {gimbalAngleString(gimbal_angle)}')
        
        self.gimbal_angle_cmd_publisher.publish(gimbal_angle)

    def publish_gimbal_angle(self):
        if self.mode == "track":
            self.publish_track_gimbal_speed()
        elif self.mode == "stabilize":
            self.publish_stablize_gimbal_angle()

    def main_loop(self, event):
        self.elapsed_time = (rospy.Time.now() - self.init_start_time).to_sec()

        os.system('clear')

        print(f"UAV: {self.uav_name} | Mode: {self.mode} | Time: {self.elapsed_time:.2f}")
        if self.initYawNEDDeg is not None:
            print(f"Initial Gimbal Yaw NED Deg: {self.initYawNEDDeg:.2f}")

        if self.meRPYRadNED is not None:
            print(f"UAV RPY NED Deg: {rpyRadString(self.meRPYRadNED)}")

        if self.meGimbalRPYNEDRad is not None:
            print(f"Raw gimbal RPY NED Deg: {rpyRadString(self.meGimbalRPYNEDRad)}")

        if self.meGimbalRPYNEDRad is not None and self.initYawNEDDeg is not None:
            gimbal_angle_corrected = self.meGimbalRPYNEDRad - np.radians(np.array([0, 0, self.initYawNEDDeg]))
            print(f"Corrected gimbal RPY NED Deg: {rpyRadString(gimbal_angle_corrected)}")

        if self.gimbalTargetRPYNEDDeg is not None:
            print(f"Target Gimbal RPY NED Deg: {rpyDegString(self.gimbalTargetRPYNEDDeg)}")

        if self.elapsed_time < 5:
            self.initialize_gimbal()
            print("Initialization incomplete. Recording initial gimbal angle.")
        else:
            if not self.initialized:
                self.initialized = True
                print("Initialization complete. Recording initial gimbal angle.")
            self.publish_gimbal_angle()

if __name__ == "__main__":
    rospy.init_node("gimbal_control_node")
    
    gimbal_control_node = GimbalControlNode('suav')
    rospy.spin()