import numpy as np

class EKF:
    def __init__(self, states, measurementNoise):
        self.R = (measurementNoise**2) * np.eye(2)
        self.P = np.diag(np.array([5, 5, 5, 0.1, 0.1, 0.1])**2) # Target position component in ENU and target velocity component in ENU 
        self.x0Truth = states
        self.x = self.x0Truth - np.array([5, 5, 5, 0.1, 0.1, 0.1]) 
        self.ekfUseNum = 1
        self.sigmaV = 0.008

    def newFrame(self, dt, u, z):
        assert isinstance(dt, (int, float)), 'dt must be a scalar'
        assert u.shape == (3,), f'u must be a (3,) vector, but now is {u.shape}'
        assert z.shape == (2,), f'z must be a (2,) vector, but now is {z.shape}'
        if self.ekfUseNum == 1:
            self.x = self.x0Truth - np.array([5, 5, 5, 0.1, 0.1, 0.1])
            print(f"self.x = {self.x}")
        else:
            self.setFGQ(dt)
            self.predict(u)
            self.update(z)
        self.ekfUseNum += 1

    def getMeState(self, data):
        self.meState = data

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.G, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

    def update(self, z):
        assert z.shape == (2, ), f'{z.shape = }, should be (2, )'
        H = self.hJacobianAt(self.x[:3] - self.meState)
        assert H.shape == (2, 6), f'{H.shape = }, should be (2, 6)'
        S = H @ (self.P @ H.T) + self.R
        assert S.shape == (2, 2), f'{S.shape = }, should be (2, 2)'
        K = self.P @ (H.T @ np.linalg.inv(S))
        assert K.shape == (6, 2), f'{K.shape = }, should be (6, 2)'
        self.P = (np.eye(K.shape[0]) - K @ H) @ self.P
        assert self.P.shape == (6, 6), f'{self.P.shape = }, should be (6, 6)'
        dZ = z - self.h(self.x[:3] - self.meState)
        dZ[0] = self.rad_round(dZ[0])
        dZ[1] = self.rad_round(dZ[1])
        print(f"dZ = {np.rad2deg(dZ)}")
        print(f"z = {np.rad2deg(z)}")
        print(f"hx = {np.rad2deg(self.h(self.x[:3] - self.meState))}")
        assert dZ.shape == (2, ), f'{dZ.shape = }, should be (2, )'
        self.x += (K @ dZ).reshape(6)
    
    def rad_round(self, rad):
        if rad < -np.pi:
            return rad + 2 * np.pi
        if rad > np.pi:
            return rad - 2 * np.pi
        return rad

    def setFGQ(self, dt):
        self.F = np.block([[np.eye(3), dt * np.eye(3)], [np.zeros((3, 3)), np.eye(3)]])
        self.G = np.block([[0.5 * dt**2 * np.eye(3)], [dt * np.eye(3)]])
        GG = np.dot(np.block([[dt**2 / 2 * np.eye(3)], [dt * np.eye(3)]]), [[self.sigmaV**2], [self.sigmaV**2], [self.sigmaV**2]])
        self.Q = np.dot(GG, GG.T)

    @staticmethod
    def h(x):
        return np.array([np.arctan2(x[2], np.sqrt(x[0]**2 + x[1]**2)), np.arctan2(x[1], x[0])])

    @staticmethod
    def hJacobianAt(x):
        relativeX = x[0]
        relativeY = x[1]
        relativeZ = x[2]
        relativeDistancePlanar = np.sqrt(relativeX**2 + relativeY**2)
        relativeDistance = np.sqrt(relativeX**2 + relativeY**2 + relativeZ**2)
        H = np.array([[-relativeX * relativeZ / relativeDistance**2 / relativeDistancePlanar, 
                       -relativeY * relativeZ / relativeDistance**2 / relativeDistancePlanar, 
                       relativeDistancePlanar / relativeDistance**2, 0, 0, 0],
                      [-relativeY / relativeDistancePlanar**2, 
                       relativeX / relativeDistancePlanar**2, 0, 0, 0, 0]])
        return H
