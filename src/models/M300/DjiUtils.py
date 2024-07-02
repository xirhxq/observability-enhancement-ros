#!/usr/bin/env python3

import rospy
import ctypes
import math
import os
import time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import (
    PoseStamped, Twist, TwistStamped, Vector3, Quaternion, Vector3Stamped, QuaternionStamped, PointStamped
)
from std_msgs.msg import (
    Float32, Int16, Int8, String, UInt8, 
    Float32MultiArray, Float64MultiArray, Int8MultiArray, Empty
)
Point = Vector3
from sensor_msgs.msg import (
    Imu, NavSatFix, Joy, TimeReference, BatteryState, Image
)
from nmea_msgs.msg import Sentence

from dji_osdk_ros.msg import (
    Gimbal, MobileData, PayloadData, FlightAnomaly, 
    VOPosition, FCTimeInUTC, GPSUTC
)
from dji_osdk_ros.srv import (
    Activation, CameraAction, DroneArmControl, 
    DroneTaskControl, DroneTaskControlRequest, 
    MFIOConfig, MFIOSetValue, 
    SDKControlAuthority, SDKControlAuthorityRequest,
    SetLocalPosRef, 
    SendMobileData, SendPayloadData, QueryDroneVersion
)

try:
    from dji_osdk_ros.srv import (
        Stereo240pSubscription, StereoDepthSubscription, 
        StereoVGASubscription, SetupCameraStream
    )
except ImportError:
    # These services are only available if ADVANCED_SENSING is defined
    pass

from spirecv_msgs.msg import TargetsInFrame, Target

def get_ros_package_path(package_name):
    import rospkg
    rospack = rospkg.RosPack()
    return rospack.get_path(package_name)

package_path = get_ros_package_path('observability_enhancement')
lib_path = os.path.abspath(os.path.join(package_path, '..', '..', 'devel', 'lib', 'libconstants.so'))

lib = ctypes.CDLL(lib_path)

lib.get_constant.restype = ctypes.c_int
lib.get_constant.argtypes = [ctypes.c_char_p]

def get_constant(name):
    return lib.get_constant(name.encode('utf-8'))

class DJISDK:
    class Control:
        HORIZONTAL_ANGLE = get_constant('DJISDK::Control::HORIZONTAL_ANGLE')
        HORIZONTAL_VELOCITY = get_constant('DJISDK::Control::HORIZONTAL_VELOCITY')
        HORIZONTAL_POSITION = get_constant('DJISDK::Control::HORIZONTAL_POSITION')
        HORIZONTAL_ANGULAR_RATE = get_constant('DJISDK::Control::HORIZONTAL_ANGULAR_RATE')
        VERTICAL_VELOCITY = get_constant('DJISDK::Control::VERTICAL_VELOCITY')
        VERTICAL_POSITION = get_constant('DJISDK::Control::VERTICAL_POSITION')
        VERTICAL_THRUST = get_constant('DJISDK::Control::VERTICAL_THRUST')
        YAW_ANGLE = get_constant('DJISDK::Control::YAW_ANGLE')
        YAW_RATE = get_constant('DJISDK::Control::YAW_RATE')
        HORIZONTAL_GROUND = get_constant('DJISDK::Control::HORIZONTAL_GROUND')
        HORIZONTAL_BODY = get_constant('DJISDK::Control::HORIZONTAL_BODY')
        STABLE_DISABLE = get_constant('DJISDK::Control::STABLE_DISABLE')
        STABLE_ENABLE = get_constant('DJISDK::Control::STABLE_ENABLE')
        
    class DisplayMode:
        MODE_MANUAL_CTRL = get_constant('DJISDK::DisplayMode::MODE_MANUAL_CTRL')
        MODE_ATTITUDE = get_constant('DJISDK::DisplayMode::MODE_ATTITUDE')
        MODE_P_GPS = get_constant('DJISDK::DisplayMode::MODE_P_GPS')
        MODE_HOTPOINT_MODE = get_constant('DJISDK::DisplayMode::MODE_HOTPOINT_MODE')
        MODE_ASSISTED_TAKEOFF = get_constant('DJISDK::DisplayMode::MODE_ASSISTED_TAKEOFF')
        MODE_AUTO_TAKEOFF = get_constant('DJISDK::DisplayMode::MODE_AUTO_TAKEOFF')
        MODE_AUTO_LANDING = get_constant('DJISDK::DisplayMode::MODE_AUTO_LANDING')
        MODE_NAVI_GO_HOME = get_constant('DJISDK::DisplayMode::MODE_NAVI_GO_HOME')
        MODE_NAVI_SDK_CTRL = get_constant('DJISDK::DisplayMode::MODE_NAVI_SDK_CTRL')
        MODE_FORCE_AUTO_LANDING = get_constant('DJISDK::DisplayMode::MODE_FORCE_AUTO_LANDING')
        MODE_SEARCH_MODE = get_constant('DJISDK::DisplayMode::MODE_SEARCH_MODE')
        MODE_ENGINE_START = get_constant('DJISDK::DisplayMode::MODE_ENGINE_START')
        
    class FlightStatus:
        STATUS_STOPPED = get_constant('DJISDK::FlightStatus::STATUS_STOPPED')
        STATUS_ON_GROUND = get_constant('DJISDK::FlightStatus::STATUS_ON_GROUND')
        STATUS_IN_AIR = get_constant('DJISDK::FlightStatus::STATUS_IN_AIR')
        
    class M100FlightStatus:
        M100_STATUS_ON_GROUND = get_constant('DJISDK::M100FlightStatus::M100_STATUS_ON_GROUND')
        M100_STATUS_TAKINGOFF = get_constant('DJISDK::M100FlightStatus::M100_STATUS_TAKINGOFF')
        M100_STATUS_IN_AIR = get_constant('DJISDK::M100FlightStatus::M100_STATUS_IN_AIR')
        M100_STATUS_LANDING = get_constant('DJISDK::M100FlightStatus::M100_STATUS_LANDING')
        M100_STATUS_FINISHED_LANDING = get_constant('DJISDK::M100FlightStatus::M100_STATUS_FINISHED_LANDING')

def print_all_constants():
    for attr_name in dir(DJISDK):
        attr = getattr(DJISDK, attr_name)
        if isinstance(attr, type):
            for const_name in dir(attr):
                if not const_name.startswith("__"):
                    const_value = getattr(attr, const_name)
                    rospy.loginfo(f"{attr_name}::{const_name}: {const_value}")

def talker():
    pub = rospy.Publisher('chatter', Int8, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)

    print_all_constants()

    while not rospy.is_shutdown():
        horizontal_angle = DJISDK.Control.HORIZONTAL_ANGLE
        horizontal_velocity = DJISDK.Control.HORIZONTAL_VELOCITY
        rospy.loginfo(f"HORIZONTAL_ANGLE: {horizontal_angle}")
        rospy.loginfo(f"HORIZONTAL_VELOCITY: {horizontal_velocity}")
        pub.publish(horizontal_angle)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass