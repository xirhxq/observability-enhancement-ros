import numpy as np
from GuidanceLaws.GuidanceLawCommon import GuidanceLawCommon
from Utils import *

class OEHG_test:
    def __init__(self):
        self.glc = None
        self.flagTgoInitial = False
        self.omega = 0.057
        self.xNPRange = 10
        self.intVcError = 0.0
        self.tStep = 0.02
        self.expectedVc = 5.0
        self.kp = 4.0
        self.ki = 12.0
        self.uOEGLos = np.zeros(3)

    def getU(self, relativePosition, relativeVelocity, velMe):
        self.glc = GuidanceLawCommon(relativePosition, relativeVelocity, velMe)

        if not self.flagTgoInitial:
            self.tgoInitial = self.glc.tGo
            self.flagTgoInitial = True

        xNP = (self.omega**2 * np.sin(self.omega * self.glc.tGo)) / (self.glc.tGo * (np.sin(self.omega * self.glc.tGo) - self.omega * self.glc.tGo * np.cos(self.omega * self.glc.tGo))) * self.glc.tGo**3
        xNP = np.clip(xNP, -self.xNPRange, self.xNPRange)
        thetaRotation = 0.5 * np.pi * self.glc.tGo / self.tgoInitial

        enu2los = enu2losRotationMatrix(np.arctan2(-relativePosition[2], np.sqrt(relativePosition[0]**2 + relativePosition[1]**2)), np.arctan2(relativePosition[1],relativePosition[0]))
        losRateLOS = np.dot(enu2los, self.glc.losRate)
        vcError = self.expectedVc - self.glc.closeVelocity
        self.intVcError = (vcError) * self.tStep + self.intVcError

        self.uOEGLos[0] = self.kp * vcError + self.ki * self.intVcError
        self.uOEGLos[1] = xNP * self.glc.closeVelocity * losRateLOS[2]
        self.uOEGLos[2] = -xNP * self.glc.closeVelocity * losRateLOS[1]
        print(f"close velocity ={self.glc.closeVelocity}" )
        print(f"kp = {self.kp}, ki = {self.ki}")
        print(f"xNP = {xNP}")
        print(f"LOSrate ={losRateLOS}")
        print(f"uOEGLos = {self.uOEGLos}")
        los2enu = los2enuRotationMatrix(np.arctan2(-relativePosition[2], np.sqrt(relativePosition[0]**2 + relativePosition[1]**2)), np.arctan2(relativePosition[1],relativePosition[0]))
        uOEGDirection = np.dot(los2enu, np.array([0, self.uOEGLos[1],self.uOEGLos[2]]))
        print(f"uOEG = {uOEGDirection}")
        uOEGVelocity = np.dot(los2enu, np.array([self.uOEGLos[0], 0, 0]))
        uDirection = np.cos(thetaRotation) * uOEGDirection + (1 - np.cos(thetaRotation)) * (np.outer(self.glc.normalisedVelMe, self.glc.normalisedVelMe)) @ uOEGDirection + np.sin(thetaRotation) * np.cross(self.glc.normalisedVelMe.flatten(), uOEGDirection.flatten()).reshape(3,)
        print(f"uDirection = {uDirection}")
        print(f"uOEGVelocity = {uOEGVelocity}")
        uV = uOEGVelocity
        u = uDirection + uV

        return u