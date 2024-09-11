import os
import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class PlotRepeatRuns:
    def __init__(self, **kwargs):
        self.baseDir = kwargs.get('baseDir', None)
        self.repeatTime = kwargs.get('repeatTime', 1)
        self.folderName = None
        self.guidanceLaw = kwargs.get('guidanceLaw', None)
        self.data = []

    def loadData(self):
        self.folderName = self.baseDir
        for root, dirs, files in os.walk(self.folderName):
            for file in files:
                if file == 'data.pkl':
                    dataPath = os.path.join(root, 'data.pkl')
                    with open(dataPath, "rb") as file:
                        loadedData = pickle.load(file)
                        self.data.append(loadedData)
            print(f"len(self.data) = {len(self.data)}")

    def plotTrajectoriesENU(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(self.repeatTime):
            targetTrajectory = np.zeros((3, len(self.data[i])))
            meTrajectory = np.zeros((3, len(self.data[i])))
            for j in range(len(self.data[i])):
                targetTrajectory[0:3, j] = np.squeeze(self.data[i][j]['targetPosition'])
                meTrajectory[0:3, j] = np.squeeze(self.data[i][j]['mePositionENU'])
            if i == 0:
                ax.plot(targetTrajectory[0, :], targetTrajectory[1, :], targetTrajectory[2, :], '*', color='k', label="Target")
            ax.plot(meTrajectory[0, :], meTrajectory[1, :], meTrajectory[2, :], color=colors[i], linewidth=2.5, label=f"Test {i+1}")
        
        ax.grid(True)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_zlabel('Z Position (m)', fontsize=12)
        ax.legend(loc='best', fontsize=12)
        ax.set_title(f'Trajectory for {self.guidanceLaw}')
        ax.view_init(13, 180)
        plt.savefig(os.path.join(self.folderName, 'Trajectories3D.png'))
        plt.close(fig)

    def plotTargetPositionErrorENU(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        for i in range(self.repeatTime):
            time = np.zeros((1,len(self.data[i])))
            ekfState = np.zeros((6,len(self.data[i])))
            targetPosition = np.zeros((3,len(self.data[i])))
            targetPositionError = np.zeros((3,len(self.data[i])))
            for j in range(len(self.data[i])):
                time[0, j] = np.squeeze(self.data[i][j]['t'])
                ekfState[:6, j] = np.squeeze(self.data[i][j]['ekfState'])
                targetPosition[:3, j] = np.squeeze(self.data[i][j]['targetPosition'])
                targetPositionError[:3, j] = np.abs(ekfState[:3, j] - targetPosition[:3, j])
            ax[0].plot(time[0, :], targetPositionError[0, :], color = colors[i], linewidth=2.0,
                    label=f"Test {i+1}")
            ax[1].plot(time[0, :], targetPositionError[1, :], color = colors[i], linewidth=2.0,
                    label=f"Test {i+1}")
            ax[2].plot(time[0, :], targetPositionError[2, :], color = colors[i], linewidth=2.0,
                    label=f"Test {i+1}")
    
        for num,direction in enumerate(['East position', 'North position', 'Up position']):
            ax[num].set_xlabel('Time (s)')
            ax[num].set_ylabel(direction +' estimated error (m)')
            ax[num].legend(loc='best')
    
        ax[0].set_title(f'ENU target position estimated error for {self.guidanceLaw}')
        plt.savefig(os.path.join(self.folderName, 'TargetPositionError.png'))
        plt.close(fig)

    def plotTargetVelocityErrorENU(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(3, 1, figsize=(8, 10))
        for i in range(self.repeatTime):
            time = np.zeros((1,len(self.data[i])))
            ekfState = np.zeros((6,len(self.data[i])))
            targetVelocity = np.zeros((3,len(self.data[i])))
            targetVelocityError = np.zeros((3,len(self.data[i])))
            for j in range(len(self.data[i])):
                time[0, j] = np.squeeze(self.data[i][j]['t'])
                ekfState[:6, j] = np.squeeze(self.data[i][j]['ekfState'])
                targetVelocity[:3, j] = np.squeeze(self.data[i][j]['targetVelocity'])
                targetVelocityError[:3, j] = np.abs(ekfState[:3, j] - targetVelocity[:3, j])
            ax[0].plot(time[0, :], targetVelocityError[0, :], color = colors[i], linewidth=2.0,
                    label=f"Test {i+1}")
            ax[1].plot(time[0, :], targetVelocityError[1, :], color = colors[i], linewidth=2.0,
                    label=f"Test {i+1}")
            ax[2].plot(time[0, :], targetVelocityError[2, :], color = colors[i], linewidth=2.0,
                    label=f"Test {i+1}")
    
        for num,direction in enumerate(['East velocity', 'North velocity', 'Up velocity']):
            ax[num].set_xlabel('Time (s)')
            ax[num].set_ylabel(direction +' estimated error (m/s)')
            ax[num].legend(loc='best')
    
        ax[0].set_title(f'ENU target velocity estimated error for {self.guidanceLaw}')
        plt.savefig(os.path.join(self.folderName, 'TargetVelocityError.png'))
        plt.close(fig)

    def plotMeasurementUseError(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        for i in range(self.repeatTime):
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
                if measurementUse[1, j] < 0:
                    measurementUse[1, j] += 2 * np.pi
                relativePosition[:3, j] = targetPosition[:3, j] - mePosition[:3, j]
                losAngle[:2, j] = np.array([
                    np.arctan2(relativePosition[2, j], np.sqrt(relativePosition[0, j] ** 2 + relativePosition[1, j] ** 2)),
                    np.arctan2(relativePosition[1, j], relativePosition[0, j])
                ])
                measurementError[:2, j] = np.abs(measurementUse[:2, j] - losAngle[:2, j])
                if measurementError[1, j] > np.pi:
                    measurementError[1, j] -= 2 * np.pi
            ax[0].plot(time[0, int(timeDelay / timeStep):], np.rad2deg(measurementError[0, int(timeDelay / timeStep):]), color = colors[i], linestyle='-', linewidth=2.0,
                            label=f"Test {i+1}")
            ax[1].plot(time[0, int(timeDelay / timeStep):], np.rad2deg(measurementError[1, int(timeDelay / timeStep):]), color = colors[i], linestyle='-', linewidth=2.0,
                            label=f"Test {i+1}")
            
            for num, direction in enumerate(['elevation angle', 'azimuth angle']):
                ax[num].set_xlabel('Time (s)')
                ax[num].set_ylabel('Measurement Error of ' + direction + ' (deg)')
                ax[num].legend(loc='best')
        
        ax[0].set_title(f'Measurement error for {self.guidanceLaw}')
        plt.savefig(os.path.join(self.folderName, 'MeasurementError.png'))
        plt.close(fig)

    def plotMissDistance(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots()
        for i in range(self.repeatTime):
            time = np.zeros((1, len(self.data[i])))
            missDistance = np.zeros((1, len(self.data[i])))
            for j in range(len(self.data[i])):
                time[0, j] = np.squeeze(self.data[i][j]['t'])
                missDistance[0, j] = np.squeeze(self.data[i][j]['missDistance'])
            print(f"Miss distance of Test {i+1} is {missDistance[0, -1]}")
            ax.plot(time[0], missDistance[0], color=colors[i], label=f"Test {i+1}", linewidth=2.0)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Miss Distance (m)', fontsize=12)
        ax.legend()
        plt.savefig(os.path.join(self.folderName, 'MissDistance.png'))
        plt.close()
    
    def createGif(self, num):
        frames = np.arange(start=0, stop=len(self.data[num]), step=1)
        tq = tqdm.tqdm(total=len(frames))
        ekfState = np.zeros((6, len(self.data[num])))
        targetPosition = np.zeros((3, len(self.data[num])))
        mePosition = np.zeros((3, len(self.data[num])))

        for i in range(len(self.data[num])):
            ekfState[0:6, i] = np.squeeze(self.data[num][i]['ekfState'])
            targetPosition[0:3, i] = np.squeeze(self.data[num][i]['targetPosition'])
            mePosition[0:3, i] = np.squeeze(self.data[num][i]['mePositionENU'])

        def getRangesFrom3dArray(arr):
            return np.array([[np.min(arr[i, :]), np.max(arr[i, :])] for i in range(arr.shape[0])])
        
        def combineRanges(rangesList):
            return np.array([
                [
                    np.min([rangesList[i][j][0] for i in range(len(rangesList))]), 
                    np.max([rangesList[i][j][1] for i in range(len(rangesList))])
                ]   for j in range(rangesList[0].shape[0])
            ])

        def getRangesFrom3dArrayList(arrList):
            return combineRanges([getRangesFrom3dArray(arr) for arr in arrList])
        
        ranges = getRangesFrom3dArrayList([ekfState[:3, :], targetPosition, mePosition])

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(ranges[0, 0], ranges[0, 1])
        ax.set_ylim(ranges[1, 0], ranges[1, 1])
        ax.set_zlim(ranges[2, 0], ranges[2, 1])
        
        ax.plot_surface(
            ranges[0][[0, 1, 0, 1]].reshape((2, 2)),
            ranges[1][[0, 0, 1, 1]].reshape((2, 2)),
            np.zeros((2, 2)),
            alpha=0.5
        )  
        
        meGL, = ax.plot([], [], [], 'b', linewidth=2, label=f'Test {num+1}')
        target, = ax.plot([], [], [], 'ro', markersize=8, label='Target')
        estimateTarget, = ax.plot([], [], [], 'k*', markersize=10, label='Estimate Target')

        ax.legend(loc='upper left', fontsize=15)
        ax.set_xlabel('X (m)', fontsize=16)
        ax.set_ylabel('Y (m)', fontsize=16)
        ax.set_zlabel('Z (m)', fontsize=16)

        ax.view_init(elev=10, azim=20)
        
        def update(frame):
            tq.update(1)
            meGL.set_data(mePosition[0, :frame], mePosition[1, :frame])
            meGL.set_3d_properties(mePosition[2, :frame])
            target.set_data([targetPosition[0, frame]], [targetPosition[1, frame]])
            target.set_3d_properties([targetPosition[2, frame]])
            estimateTarget.set_data([ekfState[0, frame]], [ekfState[1, frame]])
            estimateTarget.set_3d_properties([ekfState[2, frame]])
            return meGL, target, estimateTarget

        anim = FuncAnimation(fig, update, frames, interval=50)
        fileName = f'Estimating{num+1}' + '.gif'
        pathStr = os.path.join(self.folderName, fileName)
        anim.save(pathStr, writer='pillow')
        tq.close()

    def plotAll(self):
        self.plotTrajectoriesENU()
        self.plotTargetPositionErrorENU()
        self.plotTargetVelocityErrorENU()
        self.plotMeasurementUseError()
        self.plotMissDistance()
        for num in range(len(self.data)):
            self.createGif(num)

if __name__ == '__main__':
    prr = PlotRepeatRuns()
    prr.loadData()
    prr.plotAll()
