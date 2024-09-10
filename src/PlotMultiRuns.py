import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PlotMultiRuns:
    def __init__(self, **kwargs):
        self.baseDir = kwargs.get('baseDir', None)
        self.folderName = None
        self.guidanceLaws = []
        self.data = []

    def findGuidanceLaws(self):
        dirs = os.listdir(self.baseDir)
        latestDir = ''
        for directory in dirs:
            if os.path.isdir(os.path.join(self.baseDir, directory)) and directory not in ['.', '..']:
                latestDir = directory
        self.folderName = os.path.join(self.baseDir, latestDir)
        guidanceDirs = os.listdir(os.path.join(self.baseDir, latestDir))
        self.guidanceLaws = [guidance for guidance in guidanceDirs if os.path.isdir(os.path.join(self.baseDir, latestDir, guidance))]
        self.guidanceLaws = [law for law in self.guidanceLaws if law not in ['.', '..']]

    def loadData(self):
        for guidanceLaw in self.guidanceLaws:
            dataPath = os.path.join(self.folderName, guidanceLaw, 'data.pkl')
            if os.path.exists(dataPath):
                with open(dataPath, "rb") as file:
                    loadedData = pickle.load(file)
                self.data.append(loadedData)

    def plotTrajectoriesENU(self):
        colors = ['r', 'b']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(2):
            targetTrajectory = np.zeros((3, len(self.data[i])))
            meTrajectory = np.zeros((3, len(self.data[i])))
            for j in range(len(self.data[i])):
                targetTrajectory[0:3, j] = np.squeeze(self.data[i][j]['targetPosition'])
                meTrajectory[0:3, j] = np.squeeze(self.data[i][j]['mePosition'])
            if i == 0:
                ax.plot(targetTrajectory[0, :], targetTrajectory[1, :], targetTrajectory[2, :], '*', color='k', label="Target")
            ax.plot(meTrajectory[0, :], meTrajectory[1, :], meTrajectory[2, :], color=colors[i], linewidth=2.5, label=f"{self.guidanceLaws[i]}")
        ax.grid(True)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Z Position (m)', fontsize=12)
        ax.legend(loc='best', fontsize=12)
        ax.view_init(13, 3)
        plt.savefig(os.path.join(self.folderName, 'Trajectories3D.png'))
        plt.close(fig)

    def plotTargetPositionErrorENU(self):
        colors = ['r', 'b']
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        for i in range(2):
            time = np.zeros((1,len(self.data[i])))
            ekfState = np.zeros((6,len(self.data[i])))
            targetPosition = np.zeros((3,len(self.data[i])))
            targetPositionError = np.zeros((3,len(self.data[i])))
            for j in range(len(self.data[i])):
                time[0, j] = np.squeeze(self.data[i][j]['t'])
                ekfState[:6, j] = np.squeeze(self.data[i][j]['ekfState'])
                targetPosition[:3, j] = np.squeeze(self.data[i][j]['targetPosition'])
                targetPositionError[:3, j] = np.abs(ekfState[:3, j] - targetPosition)
            ax[0].plot(time[:], targetPositionError[0, :], color = colors[i], linewidth=2.0,
                    label=f'{self.guidanceLaws[i]}')
            ax[1].plot(time[:], targetPositionError[1, :], color = colors[i], linewidth=2.0,
                    label=f'{self.guidanceLaws[i]}')
            ax[2].plot(time[:], targetPositionError[2, :], color = colors[i], linewidth=2.0,
                    label=f'{self.guidanceLaws[i]}')
    
        for num,direction in enumerate(['East position', 'North position', 'Up position']):
            ax[num].set_xlabel('Time (s)')
            ax[num].set_ylabel(direction +' estimated error (m)')
            ax[num].legend(loc='best')
    
        ax[0].set_title('ENU target position estimated error')
        plt.savefig(os.path.join(self.folderName, 'TargetPositionError.png'))
        plt.close(fig)

    def plotTargetVelocityErrorENU(self):
        colors = ['r', 'b']
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        for i in range(2):
            time = np.zeros((1,len(self.data[i])))
            ekfState = np.zeros((6,len(self.data[i])))
            targetVelocity = np.zeros((3,len(self.data[i])))
            targetVelocityError = np.zeros((3,len(self.data[i])))
            for j in range(len(self.data[i])):
                time[0, j] = np.squeeze(self.data[i][j]['t'])
                ekfState[:6, j] = np.squeeze(self.data[i][j]['ekfState'])
                targetVelocity[:3, j] = np.squeeze(self.data[i][j]['targetVelocity'])
                targetVelocityError[:3, j] = np.abs(ekfState[:3, j] - targetVelocity)
            ax[0].plot(time[:], targetVelocityError[0, :], color = colors[i], linewidth=2.0,
                    label=f'{self.guidanceLaws[i]}')
            ax[1].plot(time[:], targetVelocityError[1, :], color = colors[i], linewidth=2.0,
                    label=f'{self.guidanceLaws[i]}')
            ax[2].plot(time[:], targetVelocityError[2, :], color = colors[i], linewidth=2.0,
                    label=f'{self.guidanceLaws[i]}')
    
        for num,direction in enumerate(['East velocity', 'North velocity', 'Up velocity']):
            ax[num].set_xlabel('Time (s)')
            ax[num].set_ylabel(direction +' estimated error (m/s)')
            ax[num].legend(loc='best')
    
        ax[0].set_title('ENU target velocity estimated error')
        plt.savefig(os.path.join(self.folderName, 'TargetVelocityError.png'))
        plt.close(fig)

    def plotMeasurementUseError(self):
        colors = ['r', 'b']
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        for i in range(2):
            timeStep = self.data[i][0]['tStep']
            timeDelay = self.data[i][0]['timeDelay']
            time = np.zeros((1, len(self.data[i])))
            targetPosition = np.zeros((3, len(self.data[i])))
            mePosition = np.zeros((3, len(self.data[i])))
            measurementUse = np.zeros((2, len(self.data[i])))
            measurementError = np.zeros((2, len(self.data[i])))
            relativePosition = np.zeros((3, len(self.data[i])))
            losAngle = np.zeros((2, len(self.data[i])))
            for j in range(len(self.data[i])):
                time[0, j] = np.squeeze(self.data[i][j]['t'])
                targetPosition[:3, j] = np.squeeze(self.data[i][j]['targetPosition'])
                mePosition[:3, j] = np.squeeze(self.data[i][j]['mePositionENU'])
                measurementUse[:2, j] = np.squeeze(self.data[i][j]['measurementUse'])
                relativePosition[:3, j] = targetPosition[:3, j] - mePosition[:3, j]
                losAngle[:2, j] = np.array([
                    np.arctan2(relativePosition[2, j], np.sqrt(relativePosition[0, j] ** 2 + relativePosition[1, j] ** 2)),
                    np.arctan2(relativePosition[1, j], relativePosition[0, j])
                ])
                measurementError[:2, j] = np.abs(measurementUse[:2, j] - losAngle[:2, j])
            ax[0].plot(time[int(timeDelay / timeStep):], np.rad2deg(measurementUse[int(timeDelay / timeStep):, 0]), color = colors[i], linestyle='-', linewidth=2.0,
                            label=f'{self.guidanceLaws[i]}')
            ax[1].plot(time[int(timeDelay / timeStep):], np.rad2deg(measurementUse[int(timeDelay / timeStep):, 1]), color = colors[i], linestyle='-', linewidth=2.0,
                            label=f'{self.guidanceLaws[i]}')
            
            for num, direction in enumerate(['elevation angle', 'azimuth angle']):
                ax[num].set_xlabel('Time (s)')
                ax[num].set_ylabel('Measurement Error of ' + direction + ' (deg)')
                ax[num].legend(loc='best')

        plt.savefig(os.path.join(self.folderName, 'MeasurementError.png'))
        plt.close(fig)

    def plotMissDistance(self):
        colors = ['r', 'b']
        fig,ax = plt.figure()
        for i in range(2):
            time = np.zeros((1, len(self.data[i])))
            missDistance = np.zeros((1, len(self.data[i])))
            for j in range(len(self.data[i])):
                time[0, j] = np.squeeze(self.data[i][j]['t'])
                missDistance[0, j] = np.squeeze(self.data[i][j]['missDistance'])
            print(f"Miss distance of {self.guidacneLaws[i]} is {missDistance[0, -1]}")
            ax.plot(time, missDistance, color = colors[i], label=f'{self.guidanceLaws[i]}', linewidth=2.0)
        ax.xlabel('Time (s)', fontsize=12)
        ax.ylabel('$Miss\ Distance\ (m)$', fontsize=12)
        plt.savefig(os.path.join(self.folderPath, 'MissDistance.png'))
        plt.close()
 
    def plotAll(self):
        self.plotTrajectoriesENU()
        self.plotTargetPositionErrorENU()
        self.plotTargetVelocityErrorENU()
        self.plotMeasurementUseError()
        self.plotMissDistance()

if __name__ == '__main__':
    pmr = PlotMultiRuns()
    pmr.findGuidanceLaws()
    pmr.loadData()
    pmr.plotAll()
