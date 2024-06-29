import math
import numpy as np

RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CLEAR = "\033[2J"


STD_SHAPE = (3,)
GRAVITY = 9.81

def ned2frdRotationMatrix(rollAngle, pitchAngle, yawAngle):
    R_z = np.array([
            [np.cos(yawAngle), np.sin(yawAngle), 0],
            [np.sin(yawAngle), np.cos(yawAngle), 0],
            [0, 0, 1]
        ])
        
    R_y = np.array([
            [np.cos(pitchAngle), 0, -np.sin(pitchAngle)],
            [0, 1, 0],
            [np.sin(pitchAngle), 0, np.cos(pitchAngle)]
        ])
        
    R_x = np.array([
            [1, 0, 0],
            [0, np.cos(rollAngle), np.sin(rollAngle)],
            [0, -np.sin(rollAngle), np.cos(rollAngle)]
        ])

    ned2frdRotationMatrix = np.dot(R_x, np.dot(R_y, R_z))
    return ned2frdRotationMatrix

def frd2nedRotationMatrix(rollAngle, pitchAngle, yawAngle):

    R = np.linalg.inv(ned2frdRotationMatrix(rollAngle, pitchAngle, yawAngle))
    
    return R

def enu2losRotationMatrix(yAxisRotationAngle, zAxisRotationAngle):

    R_y = np.array([
        [np.cos(yAxisRotationAngle), 0, -np.sin(yAxisRotationAngle)],
        [0, 1, 0],
        [np.sin(yAxisRotationAngle), 0, np.cos(yAxisRotationAngle)]
    ])
    
    R_z = np.array([
        [np.cos(zAxisRotationAngle), np.sin(zAxisRotationAngle), 0],
        [-np.sin(zAxisRotationAngle), np.cos(zAxisRotationAngle), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(R_y, R_z)
    
    return R

def los2enuRotationMatrix(yAxisRotationAngle, zAxisRotationAngle):
    return np.linalg.inv(enu2losRotationMatrix(yAxisRotationAngle, zAxisRotationAngle))

def makeArray(ls: list):
    assert len(ls) == 3, 'List length should be 3'
    return np.array(ls)


def CHECK(vec: np.ndarray):
    assert vec.shape == STD_SHAPE, f'Shape should be {STD_SHAPE} but here is {vec.shape}'


def ned2enu(vec):
    CHECK(vec)
    T = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    return T @ vec


def enu2ned(vec):
    CHECK(vec)
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
    cosHalfRoll = math.cos(attitudeAngle[0] / 2)
    cosHalfPitch = math.cos(attitudeAngle[1] / 2)
    cosHalfYaw = math.cos(attitudeAngle[2] / 2)
    sinHalfRoll = math.sin(attitudeAngle[0] / 2)
    sinHalfPitch = math.sin(attitudeAngle[1] / 2)
    sinHalfYaw = math.sin(attitudeAngle[2] / 2)

    quaternion[0] = cosHalfRoll * cosHalfPitch * cosHalfYaw + sinHalfRoll * sinHalfPitch * sinHalfYaw
    quaternion[1] = sinHalfRoll * cosHalfPitch * cosHalfYaw - cosHalfRoll * sinHalfPitch * sinHalfYaw
    quaternion[2] = cosHalfRoll * sinHalfPitch * cosHalfYaw + sinHalfRoll * cosHalfPitch * sinHalfYaw
    quaternion[3] = cosHalfRoll * cosHalfPitch * sinHalfYaw - sinHalfRoll * sinHalfPitch * cosHalfYaw
    return quaternion


def pointString(point):
    assert len(point) == 3, f'Length of point should be 3, but get {len(point)}'
    return f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})"


def rpyString(rpyRad):
    assert len(rpyRad) == 3, f'Length of rpy should be 3, but get {len(rpyRad)}'
    rpyDeg = 180 / np.pi * rpyRad
    return f'(roll: {rpyDeg[0]:.2f}, pitch: {rpyDeg[1]:.2f}, yaw: {rpyDeg[2]:.2f})'