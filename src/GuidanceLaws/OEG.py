import numpy as np
from .GuidanceLawCommon import GuidanceLawCommon

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

class OEG:
    def __init__(self):
        self.glc = None
        self.omega = 0.057
        self.xNPRange = 10
        self.intVcError = 0.0
        self.tStep = 0.02
        self.expectedVc = 10.0
        self.kp = 1.0
        self.ki = 0.01

    def getU(self, relativePosition, relativeVelocity, velMe):
        self.glc = GuidanceLawCommon(relativePosition, relativeVelocity, velMe)
        xNP = (self.omega**2 * np.sin(self.omega * self.glc.tGo)) / (self.glc.tGo * (np.sin(self.omega * self.glc.tGo) - self.omega * self.glc.tGo * np.cos(self.omega * self.glc.tGo))) * self.glc.tGo**3
        xNP = np.clip(xNP, -self.xNPRange, self.xNPRange)

        enu2los = enu2losRotationMatrix(np.arctan2(-relativePosition[2], np.sqrt(relativePosition[0]**2 + relativePosition[1]**2)), np.arctan2(relativePosition[1],relativePosition[0]))
        losRateLOS = np.dot(enu2los, self.glc.losRate)
        vcError = self.expectedVc - self.glc.closeVelocity
        self.intVcError = (vcError) * self.tStep + self.intVcError

        uLos = np.array([0, 0, 0])
        uLos[0] = self.kp * vcError + self.ki * self.intVcError
        uLos[1] = xNP * self.glc.closeVelocity * losRateLOS[2]
        uLos[2] = -xNP * self.glc.closeVelocity * losRateLOS[1]
        los2enu = los2enuRotationMatrix(np.arctan2(-relativePosition[2], np.sqrt(relativePosition[0]**2 + relativePosition[1]**2)), np.arctan2(relativePosition[1],relativePosition[0]))
        u = np.dot(los2enu, uLos)
        return u