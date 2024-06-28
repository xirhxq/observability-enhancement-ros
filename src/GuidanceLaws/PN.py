import numpy as np
from .GuidanceLawCommon import GuidanceLawCommon

class PN:
    def __init__(self):
        self.nn = 3
        self.glc = None
    
    def getU(self, relativePosition, relativeVelocity, velMe):
        self.glc = GuidanceLawCommon(relativePosition, relativeVelocity, velMe)

        u = (-self.nn * self.glc.closeVelocity * np.cross(self.glc.losRate.flatten(), self.glc.normalisedRelativeVelocity.flatten())).reshape(3, 1)

        return u
