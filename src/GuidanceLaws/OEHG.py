import numpy as np
from GuidanceLaws.GuidanceLawCommon import GuidanceLawCommon
import math
from Utils import *

class OEHG:
    def __init__(self):
        self.glc = None
        self.omega = 0.057
        self.flagTgoInitial = False
        self.tgoInitial = 0
        self.xNPRange = 10
        self.g = 9.81
        self.intVError = 0.0
        self.tStep = 0.02
        self.expectedV = 5.0
        self.kp = 4.0
        self.ki = 12.0
    
    def getU(self, relativePosition, relativeVelocity, velMe):
        self.glc = GuidanceLawCommon(relativePosition, relativeVelocity, velMe)
        print(f"kp = {self.kp}, ki = {self.ki}")
        if not self.flagTgoInitial:
            self.tgoInitial = self.glc.tGo
            self.flagTgoInitial = True

        vError = self.expectedV - np.linalg.norm(velMe)
        self.intVError = (vError) * self.tStep + self.intVError
        uv = self.kp * vError + self.ki * self.intVError
        meVelocityNED = enu2ned(velMe)
        thetaV = math.atan(meVelocityNED[2]/ np.sqrt(meVelocityNED[0]**2 + meVelocityNED[1]**2))
        phiV = math.atan(meVelocityNED[1] / meVelocityNED[0])
        uTangentialNED = np.array([uv * math.cos(thetaV) * math.cos(phiV), uv * math.cos(thetaV) * math.sin(phiV), uv * math.sin(thetaV)])
        print(f"uTangentialNED = {uTangentialNED}")
        xNP = (self.omega**2 * np.sin(self.omega * self.glc.tGo)) / (self.glc.tGo * (np.sin(self.omega * self.glc.tGo) - self.omega * self.glc.tGo * np.cos(self.omega * self.glc.tGo))) * self.glc.tGo**3
        xNP = np.clip(xNP, -self.xNPRange, self.xNPRange)
        print(f"xNP = {xNP}")
        aOEG = (xNP * self.glc.closeVelocity * np.cross(self.glc.losRate.flatten(), self.glc.normalisedVelMe.flatten())).reshape(3)
        print(f"aOEG = {aOEG}")
        thetaRotation = 0.5 * np.pi * self.glc.tGo / self.tgoInitial
        uNormal = np.cos(thetaRotation) * aOEG + (1 - np.cos(thetaRotation)) * (np.outer(self.glc.normalisedVelMe, self.glc.normalisedVelMe)) @ aOEG + np.sin(thetaRotation) * np.cross(self.glc.normalisedVelMe.flatten(), aOEG.flatten()).reshape(3)
        print(f"uNormal = {uNormal}")
        uTangential = ned2enu(uTangentialNED)
        u = uNormal + uTangential
        return u
