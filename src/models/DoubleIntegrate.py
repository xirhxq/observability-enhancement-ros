import numpy as np

class DoubleIntegrate:
    def __init__(self, posInitial=np.array([[0.0], [0.0], [0.0]]), velInitial=np.array([[0.0], [0.0], [0.0]]), method='AdjustAB'):
        self.x = np.vstack([posInitial, velInitial])
        self.A = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0]])
        self.B = np.array([[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0],
                           [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        self.method = method

    def update(self, u, dt):
        assert len(u) == 3, 'u must be a 3x1 vector'
        assert isinstance(dt, (int, float)), 'dt must be a scalar'
        if self.method == 'Euler':
            self.updateEuler(u, dt)
        elif self.method == 'Midpoint':
            self.updateMidpoint(u, dt)
        elif self.method == 'RK4':
            self.updateRK4(u, dt)
        elif self.method == 'AdjustAB':
            self.updateAdjustAB(u, dt)
        else:
            raise ValueError('Invalid method')

    def getPosition(self):
        return self.x[:3]

    def getVelocity(self):
        return self.x[3:]

    def updateEuler(self, u, dt):
        self.x += (np.dot(self.A, self.x) + np.dot(self.B, u)) * dt

    def updateMidpoint(self, u, dt):
        x_mid = self.x + (np.dot(self.A, self.x) + np.dot(self.B, u)) * (dt / 2)
        self.x += (np.dot(self.A, x_mid) + np.dot(self.B, u)) * dt

    def updateRK4(self, u, dt):
        k1 = (np.dot(self.A, self.x) + np.dot(self.B, u)) * dt
        k2 = (np.dot(self.A, (self.x + 0.5 * k1)) + np.dot(self.B, u)) * dt
        k3 = (np.dot(self.A, (self.x + 0.5 * k2)) + np.dot(self.B, u)) * dt
        k4 = (np.dot(self.A, (self.x + k3)) + np.dot(self.B, u)) * dt
        self.x += (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def updateAdjustAB(self, u, dt):
        A_adj = np.array([[0, 0, 0, dt, 0, 0],
                          [0, 0, 0, 0, dt, 0],
                          [0, 0, 0, 0, 0, dt],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]])
        B_adj = np.array([[0.5 * dt**2, 0, 0],
                          [0, 0.5 * dt**2, 0],
                          [0, 0, 0.5 * dt**2],
                          [dt, 0, 0],
                          [0, dt, 0],
                          [0, 0, dt]])
        self.x += A_adj @ self.x + B_adj @ u
