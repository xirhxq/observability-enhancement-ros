import numpy as np
from GuidanceLaws.GuidanceLawCommon import GuidanceLawCommon

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

class PN_test:
    def __init__(self):
        self.nn = 3
        self.glc = None
        self.intVcError = 0.0
        self.tStep = 0.02
        self.expectedVc = 5.0
        self.kp = 4.0
        self.ki = 8.0
        self.uLos = np.zeros(3)

    def getU(self, relativePosition, relativeVelocity, velMe):
        self.glc = GuidanceLawCommon(relativePosition, relativeVelocity, velMe)
        enu2los = enu2losRotationMatrix(np.arctan2(-relativePosition[2], np.sqrt(relativePosition[0]**2 + relativePosition[1]**2)), np.arctan2(relativePosition[1],relativePosition[0]))
        losRateLOS = np.dot(enu2los, self.glc.losRate)
        vcError = self.expectedVc - self.glc.closeVelocity
        self.intVcError = (vcError) * self.tStep + self.intVcError

        self.uLos[0] = self.kp * vcError + self.ki * self.intVcError
        self.uLos[1] = self.nn * self.glc.closeVelocity * losRateLOS[2]
        self.uLos[2] = -self.nn * self.glc.closeVelocity * losRateLOS[1]
        print(f"kp = {self.kp}, ki = {self.ki}")
        print(f"close velocity ={self.glc.closeVelocity}" )
        print(f"LOSrate ={losRateLOS}")
        print(f"uLos = {self.uLos}")
        los2enu = los2enuRotationMatrix(np.arctan2(-relativePosition[2], np.sqrt(relativePosition[0]**2 + relativePosition[1]**2)), np.arctan2(relativePosition[1],relativePosition[0]))
        u = np.dot(los2enu, self.uLos)
        return u