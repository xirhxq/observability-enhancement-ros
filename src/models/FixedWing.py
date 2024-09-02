import numpy as np


class FixedWing:
    def __init__(self, posInitial=np.array([[0.0], [0.0], [0.0]]), attInitialPYR=np.array([[0.0], [0.0], [0.0]]), velocityNorm=0.0, method='RK4'):
        self.x = np.vstack([posInitial, attInitialPYR])
        self.kinematic = None
        self.velocityNorm = velocityNorm
        self.pRollAngle = 100
        self.rollAngleCommand = 0
        self.g = 9.81
        self.method = method

    def setKinematic(self, u):
        self.kinematic = lambda state: np.array([
        [self.velocityNorm * np.cos(state[3, 0]) * np.cos(state[4, 0])],
        [self.velocityNorm * np.cos(state[3, 0]) * np.sin(state[4, 0])],
        [-self.velocityNorm * np.sin(state[3, 0])],
        [(-u[2, 0] + self.g * np.cos(state[3, 0])) / self.velocityNorm],
        [-self.g * np.tan(state[5, 0]) / self.velocityNorm],
        [self.pRollAngle * (self.rollAngleCommand - state[5, 0])]
        ])
        
    def getRollAngle(self, u):
        if u[2, 0] == 0.0:
            self.rollAngleCommand = np.pi / 2
        else:
            self.rollAngleCommand = np.arctan(-u[1, 0] / u[2, 0])

    def update(self, u, dt):
        assert len(u) == 3, 'u must be a 3x1 vector'
        assert isinstance(dt, (int, float)), 'dt must be a scalar'

        if self.method == 'Euler':
            self.updateEuler(u, dt)
        elif self.method == 'Midpoint':
            self.updateMidpoint(u, dt)
        elif self.method == 'RK4':
            self.updateRK4(u, dt)

    def getPosition(self):
        return self.x[:3]

    def getAttitude(self):
        return self.x[3:]

    def getVelocity(self):
        return np.array([self.velocityNorm * np.cos(self.x[4, 0]) * np.cos(self.x[3, 0]), self.velocityNorm * np.sin(self.x[4, 0]) * np.cos(self.x[3, 0]), -self.velocityNorm * np.sin(self.x[3, 0])]).reshape(3, 1)

    def updateEuler(self, u, dt):
        transMatrix1 = np.array([[np.cos(self.x[3, 0]), 0, -np.sin(self.x[3, 0])],
                                 [0, 1, 0],
                                 [np.sin(self.x[3, 0]), 0, np.cos(self.x[3, 0])]])
        transMatrix2 = np.array([[np.cos(self.x[4, 0]), np.sin(self.x[4, 0]), 0],
                                 [-np.sin(self.x[4, 0]), np.cos(self.x[4, 0]), 0],
                                 [0, 0, 1]])
        u = transMatrix1 @ transMatrix2 @ u
        uConsiderGravity = u - np.array([[0.0], [0.0], [-self.g]]) * np.cos(self.x[3, 0])
        self.getRollAngle(uConsiderGravity)
        self.setKinematic(uConsiderGravity)
        self.x[:6] += self.kinematic(self.x[:6]) * dt

    def updateMidpoint(self, u, dt):
        transMatrix1 = np.array([[np.cos(self.x[3, 0]), 0, -np.sin(self.x[3, 0])],
                                 [0, 1, 0],
                                 [np.sin(self.x[3, 0]), 0, np.cos(self.x[3, 0])]])
        transMatrix2 = np.array([[np.cos(self.x[4, 0]), np.sin(self.x[4, 0]), 0],
                                 [-np.sin(self.x[4, 0]), np.cos(self.x[4, 0]), 0],
                                 [0, 0, 1]])
        u = transMatrix1 @ transMatrix2 @ u
        uConsiderGravity = u - np.array([[0.0], [0.0], [-self.g]]) * np.cos(self.x[3, 0])
        self.getRollAngle(uConsiderGravity)
        self.setKinematic(uConsiderGravity)
        xMid = self.x[:6] + self.kinematic(self.x[:6]) * (dt / 2)
        self.x[:6] += self.kinematic(xMid) * dt

    def updateRK4(self, u, dt):
        transMatrix1 = np.array([[np.cos(self.x[3, 0]), 0, -np.sin(self.x[3, 0])],
                                 [0, 1, 0],
                                 [np.sin(self.x[3, 0]), 0, np.cos(self.x[3, 0])]])
        transMatrix2 = np.array([[np.cos(self.x[4, 0]), np.sin(self.x[4, 0]), 0],
                                 [-np.sin(self.x[4, 0]), np.cos(self.x[4, 0]), 0],
                                 [0, 0, 1]])
        u = transMatrix1 @ transMatrix2 @ u
        uConsiderGravity = u - np.array([[0.0], [0.0], [-self.g]]) * np.cos(self.x[3, 0])
        self.getRollAngle(uConsiderGravity)
        self.setKinematic(uConsiderGravity)
        k1 = self.kinematic(self.x[:6]) * dt
        k2 = self.kinematic(self.x[:6] + 0.5 * k1) * dt
        k3 = self.kinematic(self.x[:6] + 0.5 * k2) * dt
        k4 = self.kinematic(self.x[:6] + k3) * dt
        self.x[:6] += (k1 + 2 * k2 + 2 * k3 + k4) / 6
