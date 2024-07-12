import numpy as np

class GuidanceLawCommon:
    def __init__(self, relativePosition, relativeVelocity, velMe):
        assert relativePosition.shape == (3,), 'Relative position must be a 3, vector'
        assert relativeVelocity.shape == (3,), 'Relative velocity must be a 3, vector'
        assert velMe.shape == (3,), 'My velocity must be a 3, vector'
        
        self.relativePosition = relativePosition
        print(f"relativePositionENU = {self.relativePosition}")
        self.relativeVelocity = relativeVelocity
        print(f"relativelocityENU = {self.relativeVelocity}")
        self.velMe = velMe
        self.velMeScalar = np.linalg.norm(velMe)

        self.relativeDistance = np.linalg.norm(relativePosition)
        self.relativeSpeed = np.linalg.norm(relativeVelocity)

        self.closeVelocity = -np.dot(relativeVelocity, relativePosition) / self.relativeDistance

        self.leadAngle = np.arccos(np.dot(velMe, relativePosition) / self.relativeDistance / self.velMeScalar)

        self.tGo = self.relativeDistance / self.velMeScalar * (1 + self.leadAngle**2 / (2 * (2 * self.nn - 1)))

        self.zeroEffortMiss = relativePosition + relativeVelocity * self.tGo

        self.losRate = (np.cross(velMe, relativePosition) / self.relativeDistance**2)

        self.normalisedRelativeVelocity = relativeVelocity / self.relativeSpeed
        self.normalisedVelMe = velMe / self.velMeScalar
        self.normalisedLos = self.relativePosition/ self.relativeDistance
        self.missDistance = abs(self.relativeDistance**2 * np.linalg.norm(self.losRate) / np.sqrt(self.closeVelocity**2 + self.relativeDistance**2 + np.linalg.norm(self.losRate)**2))

    @property
    def nn(self):
        return 3
