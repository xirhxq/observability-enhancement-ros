from pyquaternion import Quaternion
from collections import deque
import rospy

from Utils import *


class QuaternionBuffer:
    def __init__(self, name='Buffer', maxAge=0.15):
        self.buffer = deque()
        self.name = name
        self.maxAge = maxAge

        self.preT = None
        self.preQ = None

        self.mode = 'NoInterp'

    @property
    def empty(self):
        return len(self.buffer) == 0

    def addMessage(self, msg):
        self.buffer.append((rospy.Time.now().to_sec(), msg))

    def getMessage(self):
        if self.empty:
            return None

        targetTime = rospy.Time.now().to_sec() - self.maxAge
        while self.buffer[0][0] < targetTime:
            self.buffer.popleft()

        q = self.buffer[0][1]
        return q

if __name__ == '__main__':
    qb = QuaternionBuffer()
    rospy.init_node('QuaternionBuffer', anonymous=True)

