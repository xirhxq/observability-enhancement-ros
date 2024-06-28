import numpy as np
from .GuidanceLawCommon import GuidanceLawCommon

class RAIM:
    def __init__(self):
        self.glc = None
        self.flagRInitial = False
        self.rInitial = 0
        self.flagRAIM = False
        self.normLosRateInitial = 0
        self.normLosRateDownInitial = 0
        self.S0 = 0
        self.switchAim = 0
        self.kUp = 0.8
        self.kDown = 0.3
    
    def getU(self, relativePosition, relativeVelocity, velMe):
        self.glc = GuidanceLawCommon(relativePosition, relativeVelocity, velMe)

        if not self.flagRInitial:
            self.rInitial = self.glc.relativeDistance
            self.flagRInitial = True

        normLosRate = np.linalg.norm(self.glc.losRate)
        uLosRate = self.glc.losRate / normLosRate
        normLosRateUp = np.deg2rad(0.8 * (np.linalg.norm(self.glc.relativePosition) / self.rInitial))
        normLosRateDown = np.deg2rad(0.3 * (np.linalg.norm(self.glc.relativePosition) / self.rInitial))

        if not self.flagRAIM:
            self.normLosRateInitial = normLosRate
            self.normLosRateDownInitial = normLosRateDown
            self.S0 = 1 if self.normLosRateInitial > self.normLosRateDownInitial else 0
            self.flagRAIM = True

        if self.normLosRateInitial < self.normLosRateDownInitial:
            if normLosRate < normLosRateUp:
                if self.switchAim == 0:
                    u = (-self.glc.closeVelocity * (normLosRateUp - normLosRate) * np.cross(uLosRate.flatten(), self.glc.normalisedVelMe.flatten())).reshape(3, 1)
                else:
                    if normLosRate >= normLosRateUp:
                        self.S0 = 1
                    elif normLosRate <= normLosRateDown:
                        self.S0 = 0
                    png = (3 * np.cross(self.glc.losRate.flatten(), self.glc.velMe.flatten())).reshape(3, 1)
                    u = png * self.S0
            else:
                self.switchAim = 1
                if normLosRate >= normLosRateUp:
                    self.S0 = 1
                elif normLosRate <= normLosRateDown:
                    self.S0 = 0
                png = (3 * np.cross(self.glc.losRate.flatten(), self.glc.velMe.flatten())).reshape(3, 1)
                u = png * self.S0
        else:
            if normLosRate >= normLosRateUp:
                self.S0 = 1
            elif normLosRate <= normLosRateDown:
                self.S0 = 0
            png = (3 * np.cross(self.glc.losRate.flatten(), self.glc.velMe.flatten())).reshape(3, 1)
            u = png * self.S0

        return u