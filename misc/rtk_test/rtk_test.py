import matplotlib.pyplot as plt
import rosbag
import numpy as np
from sensor_msgs.msg import NavSatFix

def localOffsetFromGpsOffset(target: NavSatFix, origin: NavSatFix):
    deltaLon = target.longitude - origin.longitude
    deltaLat = target.latitude - origin.latitude
    C_EARTH = 6378137.0
    deltaENU = [np.deg2rad(deltaLon) * C_EARTH * np.cos(np.deg2rad(target.latitude)),
                np.deg2rad(deltaLat) * C_EARTH,
                target.altitude - origin.altitude]
    return deltaENU
  
def extract_lla(msg):
    lla = [msg.latitude, msg.longitude, msg.altitude]
    return lla
  
def extract_pos(msg):
    posENU = [msg.point.x, msg.point.y, msg.point.z]
    return posENU

if __name__ == '__main__':
    uav_name = "/suav"
    bag_filename = 'position_2024-07-09-11-58-51.bag'

    localPositionENU = []
    localPositionGPSlla = []
    posGPS = []
    localPositionRTKlla = []
    posRTK = []

    topic_list = [uav_name + "/dji_osdk_ros/local_frame_ref",
                  uav_name + "/dji_osdk_ros/local_position",
                  uav_name + "/dji_osdk_ros/gps_position", 
                  uav_name + "/dji_osdk_ros/rtk_position"
                ]
    
    bag = rosbag.Bag(bag_filename)
    for topic, msg, t in bag.read_messages(topics=topic_list[0]):
        origin = extract_lla(msg)

    for topic, msg, t in bag.read_messages(topics=topic_list[1]):
        value1 = extract_pos(msg)
        if value1 is not None:
            localPositionENU.append([t.to_sec(), value1])

    for topic, msg, t in bag.read_messages(topics=topic_list[2]):
        value2 = extract_lla(msg)
        if value2 is not None:
            localPositionGPSlla.append([t.to_sec(), value2])
            posGPS.append(localOffsetFromGpsOffset(
                NavSatFix(altitude=localPositionGPSlla[-1][1][2], latitude=localPositionGPSlla[-1][1][0], longitude=localPositionGPSlla[-1][1][1]),
                NavSatFix(altitude=origin[2], latitude=origin[0], longitude=origin[1]),
                ))

    for topic, msg, t in bag.read_messages(topics=topic_list[3]):
        value3 = extract_lla(msg)
        if value3 is not None:
            localPositionRTKlla.append([t.to_sec(), value3])
            posRTK.append(localOffsetFromGpsOffset(
                NavSatFix(altitude=localPositionRTKlla[-1][1][2], latitude=localPositionRTKlla[-1][1][0], longitude=localPositionRTKlla[-1][1][1]),
                NavSatFix(altitude=origin[2], latitude=origin[0], longitude=origin[1]),
                ))
      

    fig = plt.figure()
    posUse = [data[1] for data in localPositionENU]
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(np.array(posUse)[0, :], np.array(posUse)[1, :], np.array(posUse)[2, :], '-', color='r', label='locPos')
    ax.plot(np.array(posGPS)[0, :], np.array(posGPS)[1, :], np.array(posGPS)[2, :], '--', color='g', label='GPS')
    ax.plot(np.array(posRTK)[0, :], np.array(posRTK)[1, :], np.array(posRTK)[2, :], '--', color='b', label='RTK')
    ax.set_xlabel('$X$ (m)')
    ax.set_ylabel('$Y$ (m)')
    ax.set_zlabel('$Z$ (m)')
    plt.xlabel('Time (s)')
    plt.ylabel( 'Position (m)')
    plt.legend()
    plt.savefig('pos.png')
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    time1 = [data[0] for data in localPositionENU]
    time2 = [data[0] for data in localPositionGPSlla]
    time3 = [data[0] for data in localPositionRTKlla]
    
    axs[0].plot(time1, np.array(posUse)[:, 0], '-', color = 'r', label='posUse')
    axs[0].plot(time2, np.array(posGPS)[:, 0], '--', color = 'g', label='posGPS')
    axs[0].plot(time3, np.array(posRTK)[:, 0], ':', color = 'b', label='posRTK')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('E-Position (m)')
    axs[0].legend()

    axs[1].plot(time1, np.array(posUse)[:, 1], '-', color = 'r', label='posUse')
    axs[1].plot(time2, np.array(posGPS)[:, 1], '--', color = 'g', label='posGPS')
    axs[1].plot(time3, np.array(posRTK)[:, 1], ':', color = 'b', label='posRTK')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('N-Position (m)')
    axs[1].legend()

    axs[2].plot(time1, np.array(posUse)[:, 2], '-', color = 'r', label='posUse')
    axs[2].plot(time2, np.array(posGPS)[:, 2], '--', color = 'g', label='posGPS')
    axs[2].plot(time3, np.array(posRTK)[:, 2], ':', color = 'b', label='posRTK')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('U-Position (m)')
    axs[2].legend()

    plt.savefig('pos1.png')
    plt.show()
    #plt.close()