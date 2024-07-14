import rosbag
import numpy as np
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Vector3, PointStamped

def localOffsetFromGpsOffset(target: NavSatFix, origin: NavSatFix):
    deltaLon = target.longitude - origin.longitude
    deltaLat = target.latitude - origin.latitude
    C_EARTH = 6378137.0
    return [
        np.deg2rad(deltaLon) * C_EARTH * np.cos(np.deg2rad(target.latitude)),
        np.deg2rad(deltaLat) * C_EARTH,
        target.altitude - origin.altitude
    ]

def getValue(target: NavSatFix):
    return [target.longitude, target.latitude, target.altitude]

bag = rosbag.Bag('2024-07-07-14-31-49.bag')

origin = []

for topic, msg, t in bag.read_messages('/suav/dji_osdk_ros/local_frame_ref'):
    origin = getValue(msg)

print(f'Origin: lla: {origin}')

localPosition = []

for topic, msg, t in bag.read_messages(['/suav/dji_osdk_ros/local_position']):
    localPosition.append([msg.point.x, msg.point.y, msg.point.z])

gpsPosition = []
rtkPosition = []
    
for topic, msg, t in bag.read_messages(['/suav/dji_osdk_ros/gps_position', '/suav/dji_osdk_ros/rtk_position']):
    value = getValue(msg)
    if 'rtk' in topic:
        rtkPosition.append(value)
    if 'gps' in topic:
        gpsPosition.append(value)

print(len(gpsPosition), len(rtkPosition), len(localPosition))

import pickle

dict = {
    'origin': origin,
    'localPosition': localPosition,
    'gpsPosition': gpsPosition,
    'rtkPosition': rtkPosition
}

with open('data.pkl', 'wb') as f:
    pickle.dump(dict, f)