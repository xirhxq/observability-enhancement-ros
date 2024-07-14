#!/usr/bin/env python
import rospy
import rosbag
import random
import numpy as np
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import NavSatFix

def localOffsetFromGpsOffset(target, origin):
    deltaLon = target.longitude - origin.longitude
    deltaLat = target.latitude - origin.latitude
    C_EARTH = 6378137.0
    return [
        np.deg2rad(deltaLon) * C_EARTH * np.cos(np.deg2rad(target.latitude)),
        np.deg2rad(deltaLat) * C_EARTH,
        target.altitude - origin.altitude
    ]

def create_rosbag(use_rtk, bag_filename):
    bag = rosbag.Bag(bag_filename, 'w')

    try:
        # Initialize ROS node
        rospy.init_node('rosbag_creator', anonymous=True)

        # Local frame reference (only one message)
        local_frame_ref = NavSatFix()
        local_frame_ref.latitude = 40.0
        local_frame_ref.longitude = -74.0
        local_frame_ref.altitude = 10.0
        local_frame_ref.header.stamp = rospy.Time.now()
        local_frame_ref.header.frame_id = "world"
        
        bag.write('/suav/dji_osdk_ros/local_frame_ref', local_frame_ref, local_frame_ref.header.stamp)

        # Simulate GPS and RTK positions (continuous trajectories with slight differences)
        start_time = rospy.Time.now()
        for i in range(100):
            gps_position = NavSatFix()
            gps_position.header.stamp = start_time + rospy.Duration(i * 0.1)
            gps_position.latitude = 40.0 + 0.0001 * i + random.uniform(-0.00001, 0.00001)
            gps_position.longitude = -74.0 + 0.0001 * i + random.uniform(-0.00001, 0.00001)
            gps_position.altitude = 10.0 + random.uniform(-0.1, 0.1)
            gps_position.header.frame_id = "gps"
            
            bag.write('/suav/dji_osdk_ros/gps_position', gps_position, gps_position.header.stamp)
            
            rtk_position = NavSatFix()
            rtk_position.header.stamp = start_time + rospy.Duration(i * 0.1)
            rtk_position.latitude = 40.0 + 0.0001 * i + random.uniform(-0.000005, 0.000005)
            rtk_position.longitude = -74.0 + 0.0001 * i + random.uniform(-0.000005, 0.000005)
            rtk_position.altitude = 10.0 + random.uniform(-0.05, 0.05)
            rtk_position.header.frame_id = "rtk"
            
            bag.write('/suav/dji_osdk_ros/rtk_position', rtk_position, rtk_position.header.stamp)

            # Calculate local position
            local_position = PointStamped()
            local_position.header.stamp = start_time + rospy.Duration(i * 0.1)
            local_position.header.frame_id = "local"
            if use_rtk:
                offset = localOffsetFromGpsOffset(rtk_position, local_frame_ref)
            else:
                offset = localOffsetFromGpsOffset(gps_position, local_frame_ref)

            local_position.point.x = offset[0]
            local_position.point.y = offset[1]
            local_position.point.z = offset[2]
            
            bag.write('/suav/dji_osdk_ros/local_position', local_position, local_position.header.stamp)
    finally:
        bag.close()

if __name__ == '__main__':
    use_rtk = False  # Modify this value to use RTK or GPS for local_position calculation
    bag_filename = "example_rosbag.bag"
    create_rosbag(use_rtk, bag_filename)