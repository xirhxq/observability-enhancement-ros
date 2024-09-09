#!/usr/bin/env python3

import os
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

class PlotSingleRun:
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', 'M300')
        self.useCamera = kwargs.get('useCamera', False)
        self.baseDir = None
        self.packagePath = kwargs.get('packagePath', None)
        self.guidanceLawName =  kwargs.get('GL', 'OEHG_test')
        self.lastDir = None
        self.folderPath = None
        self.file = None
        self.data = None
        self.gifLoop = 1

    def findLastDir(self):
        self.baseDir = os.path.join(os.getcwd(), 'data') if self.packagePath is None else os.path.join(self.packagePath, 'data')
        dirs = sorted(os.listdir(self.baseDir))
        for d in dirs:
            if os.path.isdir(os.path.join(self.baseDir, d)) and not d.startswith('.'):
                self.lastDir = d
        guidanceDirPath = os.path.join(self.baseDir, self.lastDir, self.guidanceLawName)
        if not os.path.isdir(guidanceDirPath):
            raise ValueError(f"Directory for {self.guidanceLawName} does not exist in {self.lastDir}")
        self.folderPath = os.path.join(self.baseDir, self.lastDir, self.guidanceLawName)
        self.file = os.path.join(self.folderPath, 'data.pkl')

    def loadData(self):
        if self.file is None:
            raise ValueError("Data file path not set. Please run findLastDir method first.")
        with open(self.file, "rb") as file:
            self.data = pickle.load(file)

    def plotRelativeDistanceError(self):
        self.createFigure()
        distanceErrors = np.zeros(len(self.data))
        for i in range(len(self.data)):
            distanceErrors[i] = np.linalg.norm(self.data[i]['relativePosition'] - self.data[i]['estimateState'][:3])
        plt.plot([d['t'] for d in self.data], distanceErrors, 'r', linewidth=2.5)
        self.decorateFigure('Time (s)', 'Error of relative distance (m)', 'RelativeDistanceError.png')

    def plotRelativeVelocityError(self):
        self.createFigure()
        velocityErrors = np.zeros(len(self.data))
        for i in range(len(self.data)):
            velocityErrors[i] = np.linalg.norm(self.data[i]['relativeVelocity'] - self.data[i]['estimateState'][3:])
        plt.plot([d['t'] for d in self.data], velocityErrors, 'r', linewidth=2.5)
        self.decorateFigure('Time (s)', 'Error of relative velocity (m/s)', 'RelativeVelocityError.png')

    def plotTrajectory(self):
        self.createFigure()

        targetTrajectory = np.zeros((3, len(self.data)))
        meTrajectory = np.zeros((3, len(self.data)))

        for i in range(len(self.data)):
            targetTrajectory[0:3, i] = np.squeeze(self.data[i]['targetPosition'])
            meTrajectory[0:3, i] = np.squeeze(self.data[i]['mePositionENU'])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(targetTrajectory[0, :], targetTrajectory[1, :], targetTrajectory[2, :], '*', color='b')
        ax.plot(meTrajectory[0, :], meTrajectory[1, :], meTrajectory[2, :], 'r', linewidth=2.5)

        ax.set_xlabel('$X$ (m)')
        ax.set_ylabel('$Y$ (m)')
        ax.set_zlabel('$Z$ (m)')

        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=30, azim=180)

        plt.title('ENU Position', fontsize=10)
        plt.savefig(self.folderPath + '/Trajectory.png')
        plt.close()
        
    def plotXYZTrajectories(self):
        self.createFigure()

        targetPos = np.zeros((3, len(self.data)))
        mePos = np.zeros((3, len(self.data)))

        time = [d['t'] for d in self.data]
        for i in range(len(self.data)):
            targetPos[0:3, i] = np.squeeze(self.data[i]['targetPosition'])
            mePos[0:3, i] = np.squeeze(self.data[i]['mePositionENU'])

        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

        axs[0].plot(time, mePos[0, :], 'r', linewidth=2, label = 'me position')
        axs[0].plot(time, targetPos[0, :], '--', color=[0.5, 0.5, 0.5, 0.5], label = 'target position')
        axs[0].set_ylabel('E Position', fontsize=10)
        axs[0].legend()

        axs[1].plot(time, mePos[1, :], 'g', linewidth=2, label = 'me position')
        axs[1].plot(time, targetPos[1, :], '--', color=[0.5, 0.5, 0.5, 0.5], label = 'target position')
        axs[1].set_ylabel('N Position', fontsize=10)
        axs[1].legend()

        axs[2].plot(time, mePos[2, :], 'b', linewidth=2, label = 'me position')
        axs[2].plot(time, targetPos[2, :], '--', color=[0.5, 0.5, 0.5, 0.5],  label = 'target position')
        axs[2].set_ylabel('U Position', fontsize=10)
        axs[2].set_xlabel('Time (s)')
        axs[2].legend()

        fig.suptitle('Trajectory Comparison', fontsize=12)

        axs[0].set_title('X Coordinate', fontsize=10)
        axs[1].set_title('Y Coordinate', fontsize=10)
        axs[2].set_title('Z Coordinate', fontsize=10)

        plt.savefig(self.folderPath + '/TrajectoryComparison.png')
        plt.close()
    
    def plotAMy(self):
        self.createFigure()

        uz = np.array([0, 0, 1])
        am = np.zeros((3, len(self.data)))
        vr = np.zeros((3, len(self.data)))
        aMy = np.zeros((1, len(self.data)))

        time = [d['t'] for d in self.data]
        for i in range(len(self.data)):
            am[0:3, i] = np.squeeze(self.data[i]['u'])
            vr[0:3, i] = np.squeeze(self.data[i]['relativeVelocity'])
            aMy[0, i] = np.squeeze(np.dot(am[0:3, i], np.cross(uz, vr[0:3, i])) / np.linalg.norm(np.cross(uz, vr[0:3, i])))
        plt.plot(time[:-300], aMy[0, :-300], 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('$\mathrm{a_{My}\ (m/s^2)}$', fontsize=12)
        plt.savefig(self.folderPath + '/aMy.png')
        plt.close()

    def plotAMz(self):
        self.createFigure()
        uz = np.array([0, 0, 1])
        am = np.zeros((3, len(self.data)))
        vr = np.zeros((3, len(self.data)))
        aMz = np.zeros((1, len(self.data)))
        
        def calculateAMz(i):
            uzCrossVr = np.cross(uz, vr[0:3, i]) / np.linalg.norm(np.cross(uz, vr[0:3, i]))
            aMzSquared = np.linalg.norm(am[0:3, i])**2 - np.dot(am[0:3, i], uzCrossVr)**2
            return np.sqrt(aMzSquared) * np.sign(np.dot(vr[:, i], np.cross(np.cross(uz, vr[:, i]), am[:, i])))

        time = [d['t'] for d in self.data]
        for i in range(len(self.data)):
            am[0:3, i] = np.squeeze(self.data[i]['u'])
            vr[0:3, i] = np.squeeze(self.data[i]['relativeVelocity'])
            aMz[0, i] = np.squeeze(calculateAMz(i))
   
        plt.plot(time[:-300], aMz[0, :-300], 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('$\mathrm{a_{Mz}\ (m/s^2)}$', fontsize=12)
        plt.savefig(self.folderPath + '/aMz.png')
        plt.close()
    
    def plotTgo(self):
        self.createFigure()
        time = [d['t'] for d in self.data]
        tGo = [data['tGo'] for data in self.data]

        plt.plot(time, tGo, 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('$\mathrm{t_{go}\ (s)}$', fontsize=12)
        plt.savefig(os.path.join(self.folderPath, 'tGo.png'))
        plt.close()

    def plotCloseVelocity(self):
        self.createFigure()
        time = [d['t'] for d in self.data]
        closeVelocity = [data['closeVelocity'] for data in self.data]

        plt.plot(time, closeVelocity, 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Close velocity (m/s)', fontsize=12)
        plt.savefig(os.path.join(self.folderPath, 'closeVelocity.png'))
        plt.close()

    
    def plotLeadAngle(self):
        self.createFigure()
        time = [d['t'] for d in self.data]
        leadAngle = np.rad2deg([data['leadAngle'] for data in self.data])

        plt.plot(time, leadAngle, 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Lead Angle (deg)', fontsize=12)
        plt.savefig(os.path.join(self.folderPath, 'leadAngle.png'))
        plt.close()

    def plotZeroEffortMiss(self):
        self.createFigure()
        time = [d['t'] for d in self.data]
        zeroEffortMissNorm = [np.linalg.norm(data['zeroEffortMiss']) for data in self.data]

        plt.plot(time, zeroEffortMissNorm, 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('$Zero\ Effort\ Miss\ (m)$', fontsize=12)
        plt.savefig(os.path.join(self.folderPath, 'ZeroEffortMiss.png'))
        plt.close()

    def plotMissDistance(self):
        self.createFigure()
        time = [d['t'] for d in self.data]
        missDistance = [data['missDistance'] for data in self.data]

        plt.plot(time, missDistance, 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('$Miss\ Distance\ (m)$', fontsize=12)
        plt.savefig(os.path.join(self.folderPath, 'MissDistance.png'))
        plt.close()

    def plotEulerMeAndCommandENU(self):
        self.createFigure()

        time = [d['t'] for d in self.data]
        meRPYENU = np.array([d['meRPYENU'] for d in self.data])
        cmdRPYENU = np.array([d['cmdRPYENU'] for d in self.data])

        str = ['Roll', 'Pitch', 'Yaw']
        colors = ['r', 'g', 'b']

        for i, name in enumerate(str):
            plt.subplot(3, 1, i + 1)
            plt.plot(time, np.rad2deg(meRPYENU[:, i]), colors[i] + '-', linewidth=2, label=name)
            plt.plot(time, np.rad2deg(cmdRPYENU[:, i]), colors[i] + '--', linewidth=2, label=name + ' command')
            plt.xlabel('Time (s)')
            plt.ylabel(str[i] + ' angle (deg)')
            plt.legend()

        plt.savefig(os.path.join(self.folderPath, 'EulerMeAndCommand.png'))
        plt.close()

    def plotLosRate(self):
        self.createFigure()

        time = [d['t'] for d in self.data]
        losRate = np.zeros((3, len(self.data)))
        losRateNorm = np.zeros(len(self.data))

        for i in range(len(self.data)):
            losRate[:, i] = np.squeeze(self.data[i]['lambda'])
            losRateNorm[i] = np.linalg.norm(losRate[:, i])

        plt.plot(time, losRateNorm)
        plt.savefig(os.path.join(self.folderPath, 'losRateNorm.png'))
        plt.close()

    def plotAttitudeMeENU(self):
        self.createFigure()

        time = [d['t'] for d in self.data]
        attitude = np.zeros((3, len(self.data)))

        for i in range(len(self.data)):
            attitude[0:3, i] = np.squeeze(self.data[i]['meRPYENU'])

        plt.subplot(3, 1, 1)
        plt.plot(time, np.rad2deg(attitude[0, :]), 'r', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Roll angle (deg)')

        plt.subplot(3, 1, 2)
        plt.plot(time, np.rad2deg(attitude[1, :]), 'g', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch angle (deg)')

        plt.subplot(3, 1, 3)
        plt.plot(time, np.rad2deg(attitude[2, :]), 'r', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Yaw angle (deg)')
        plt.savefig(os.path.join(self.folderPath, 'AttitudeMeENU.png'))
        plt.close()
    
    def plotVelocityAndAcceleration(self):
        self.createFigure()

        velocity = np.zeros((3, len(self.data)))
        acceleration = np.zeros((3, len(self.data)))

        time = [d['t'] for d in self.data]
        for i in range(len(self.data)):
            velocity[0:3, i] = np.squeeze(self.data[i]['meVelocity'])
            acceleration[0:3, i] = np.squeeze(self.data[i]['u'])

        plt.subplot(3, 1, 1)
        plt.plot(time, velocity[0, :], 'r', linewidth=2)
        plt.plot(time, acceleration[0, :], '--', color=[1.0, 0, 0, 0.5])

        plt.subplot(3, 1, 2)
        plt.plot(time, velocity[1, :], 'g', linewidth=2)
        plt.plot(time, acceleration[1, :], '--', color=[0, 1, 0, 0.5])

        plt.subplot(3, 1, 3)
        plt.plot(time, velocity[2, :], 'b', linewidth=2)
        plt.plot(time, acceleration[2, :], '--', color=[0, 0, 1, 0.5])
        plt.xlabel('Time (s)')

        plt.suptitle('Trajectory Comparison')
        plt.subplot(3, 1, 1)
        plt.title('X Coordinate', fontsize=10)
        plt.subplot(3, 1, 2)
        plt.title('Y Coordinate', fontsize=10)
        plt.subplot(3, 1, 3)
        plt.title('Z Coordinate', fontsize=10)
        plt.savefig(os.path.join(self.folderPath, 'VelocityAndAcceleration.png'))
        plt.close()

    def plotMeasurement(self):
        self.createFigure()

        measurement = np.zeros((2, len(self.data)))

        time = [d['t'] for d in self.data]
        for i in range(len(self.data)):
            measurement[0:2, i] = np.squeeze(self.data[i]['measurement'])

        plt.subplot(2, 1, 1)
        plt.plot(time, np.rad2deg(measurement[0, :]), 'r', linewidth=2.5)
        plt.title('Elevation angle', fontsize=10)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Measurement (deg)', fontsize=12)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, np.rad2deg(measurement[1, :]), 'g', linewidth=2.5)
        plt.title('Azimuth angle', fontsize=10)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Measurement (deg)', fontsize=12)
        
        plt.savefig(os.path.join(self.folderPath, 'Measurement.png'))
        plt.close()

    def plotMeasurementUse(self):
        self.createFigure()
        time = [d['t'] for d in self.data]

        measurementUse = []
      

        for data in self.data:
            if not len(data['measurementUse']) == 0:
                measurementUse.append(data['measurementUse'])

        measurementUseMatrix = np.zeros((2, len(measurementUse)))

        for i in range(len(measurementUse)):
            measurementUseMatrix[0, i] = measurementUse[i][0]
            measurementUseMatrix[1, i] = measurementUse[i][1]
            
        plt.subplot(2, 1, 1)
        plt.plot(time[int(self.data[0]['timeDelay'] / self.data[0]['tStep']) + 1:], np.rad2deg(measurementUseMatrix[0][int(self.data[0]['timeDelay'] / self.data[0]['tStep']) + 1:]), 'r', linewidth=2.5)
        plt.title('Elevation angle', fontsize=10)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Used Measurement (deg)', fontsize=12)
        
        plt.subplot(2, 1, 2)
        plt.plot(time[int(self.data[0]['timeDelay'] / self.data[0]['tStep']) + 1:], np.rad2deg(measurementUseMatrix[1][int(self.data[0]['timeDelay'] / self.data[0]['tStep']) + 1:]), 'g', linewidth=2.5)
        plt.title('Azimuth angle', fontsize=10)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Used Measurement (deg)', fontsize=12)
        
        plt.savefig(os.path.join(self.folderPath, 'MeasurementUse.png'))
        plt.close()
    
    def plotTargetPositionError(self):
        self.createFigure()

        ekfState = np.zeros((6, len(self.data)))
        targetPosition = np.zeros((3, len(self.data)))

        time = [d['t'] for d in self.data]
        for i in range(len(self.data)):
            ekfState[0:6, i] = np.squeeze(self.data[i]['ekfState'])
            targetPosition[0:3, i] = np.squeeze(self.data[i]['targetPosition'])
        
        RT_error = np.abs(ekfState[:3, :] - targetPosition)

        plt.subplot(3, 1, 1)
        plt.plot(time, RT_error[0, :], 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Error of RTx (m)', fontsize=10)

        plt.subplot(3, 1, 2)
        plt.plot(time, RT_error[1, :], 'g', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Error of RTy (m)', fontsize=10)

        plt.subplot(3, 1, 3)
        plt.plot(time, RT_error[2, :], 'b', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Error of RTz (m)', fontsize=10)

        plt.savefig(os.path.join(self.folderPath, 'TargetPositionError.png'))
        plt.close()

    def plotTargetVelocityError(self):
        self.createFigure()

        ekfState = np.zeros((6, len(self.data)))
        targetVelocity = np.zeros((3, len(self.data)))

        time = [d['t'] for d in self.data]
        for i in range(len(self.data)):
            ekfState[0:6, i] = np.squeeze(self.data[i]['ekfState'])
            targetVelocity[0:3, i] = np.squeeze(self.data[i]['targetVelocity'])

        VT_error = np.abs(ekfState[3:, :] - targetVelocity)

        plt.subplot(3, 1, 1)
        plt.plot(time, VT_error[0, :], 'r', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Error of VTx (m/s)', fontsize=10)

        plt.subplot(3, 1, 2)
        plt.plot(time, VT_error[1, :], 'g', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Error of VTy (m/s)', fontsize=10)

        plt.subplot(3, 1, 3)
        plt.plot(time, VT_error[2, :], 'b', linewidth=2.5)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Error of VTz (m/s)', fontsize=10)

        plt.savefig(os.path.join(self.folderPath, 'TargetVelocityError.png'))
        plt.close()

    def plotVelResponse(self):
        self.createFigure()

        velNorm = [d['meVelocityNorm'] for d in self.data]
        time = [d['t'] for d in self.data]
        
        plt.plot(time, velNorm[:], 'r', linewidth=2, label="Velocity Norm")
        plt.legend()
        plt.savefig(os.path.join(self.folderPath, 'velResponse.png'))
        plt.close()

    def plotLookAngleScatterWithTime(self):
        self.createFigure()

        time = [d['t'] for d in self.data]
        lookAngle = np.array([data['lookAngle'] for data in self.data]).T
        elevationAngle, azimuthAngle = np.rad2deg(lookAngle)

        fig, ax = plt.subplots()

        scatter = ax.scatter(
            azimuthAngle, elevationAngle,
            c=time, cmap='coolwarm',
            s=1, alpha=0.5
        )
        ax.set_xlabel('Azimuth Angle (deg)') 
        ax.set_ylabel('Elevation Angle (deg)') 
        cbar = plt.colorbar(scatter, ax=ax) 
        cbar.set_label('Time')

        plt.savefig(os.path.join(self.folderPath, 'LookAngleScatterWithTime.png'))
        plt.close()

    def plotLookAngleWithTime(self):
        self.createFigure()

        time = [d['t'] for d in self.data]
        lookAngle = np.array([data['lookAngle'] for data in self.data]).T
        elevationAngle, azimuthAngle = np.rad2deg(lookAngle)

        fig, ax = plt.subplots()

        ax.plot(time, elevationAngle, 'r', linewidth=2.5, label="elevation angle")
        ax.plot(time, azimuthAngle, 'b', linewidth=2.5, label="azimuth angle")
        ax.set_xlabel('Time (s)') 
        ax.set_ylabel('Look Angles (deg)') 
        ax.legend()

        plt.savefig(os.path.join(self.folderPath, 'LookAngleWithTime.png'))
        plt.close()

    def plotLookAngleWithRelativeDistance(self):
        self.createFigure()

        relativeDistance = [d['relativeDistance'] for d in self.data]
        lookAngle = np.array([data['lookAngle'] for data in self.data]).T
        elevationAngle, azimuthAngle = np.rad2deg(lookAngle)

        fig, ax = plt.subplots()
        ax.plot(relativeDistance, elevationAngle, 'r', linewidth=2.5, label="elevation angle")
        ax.plot(relativeDistance, azimuthAngle, 'b', linewidth=2.5, label="azimuth angle")
        ax.set_xlabel('Relative distance (m)') 
        ax.set_ylabel('Angle (deg)') 
        ax.invert_xaxis()
        ax.legend()

        plt.savefig(os.path.join(self.folderPath, 'LookAngleWithRelativeDistance.png'))
        plt.close()
    
    def plotGimbalAngleAndAttitudeAngle(self):
        self.createFigure()

        time = [d['t'] for d in self.data]
        attitude = np.zeros((3, len(self.data)))
        gimbal = np.zeros((3, len(self.data)))

        for i in range(len(self.data)):
            attitude[0:3, i] = np.squeeze(self.data[i]['meRPYENU'])
            gimbal[0:3, i] = np.squeeze(self.data[i]['meGimbalRPYENURad'])

        plt.subplot(3, 1, 1)
        plt.plot(time, np.rad2deg(attitude[0, :]), 'r', linewidth=2, label='attitude')
        plt.plot(time, np.rad2deg(gimbal[0, :]), 'g', linewidth=2, label='gimbal')
        plt.xlabel('Time (s)')
        plt.ylabel('Roll angle (deg)')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(time, np.rad2deg(attitude[1, :]), 'r', linewidth=2, label='attitude')
        plt.plot(time, np.rad2deg(gimbal[1, :]), 'g', linewidth=2, label='gimbal')
        plt.xlabel('Time (s)')
        plt.ylabel('Pitch angle (deg)')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(time, np.rad2deg(attitude[2, :]), 'r', linewidth=2, label='attitude')
        plt.plot(time, np.rad2deg(gimbal[2, :]), 'g', linewidth=2, label='gimbal')
        plt.xlabel('Time (s)')
        plt.ylabel('Yaw angle (deg)')
        plt.legend()
        plt.savefig(os.path.join(self.folderPath, 'AttitudeAndGimbal.png'))
        plt.close()

    def plotAcceResponse(self):
        self.createFigure()

        acceNED = np.zeros((3, len(self.data)))
        acceResponse = np.zeros((3, len(self.data)))
        acceFusedResponse = np.zeros((3, len(self.data)))

        time = [d['t'] for d in self.data]
        for i in range(len(self.data)):
            acceNED[0:3, i] = np.squeeze(self.data[i]['meAcceCommandNED'])
            acceFusedResponse[0:3, i] = np.squeeze(self.data[i]['meAccelerationFusedNED'])

        plt.subplot(3, 1, 1)
        plt.plot(time, acceNED[0, :], 'r', linewidth=2, label="Acceleration command N")
        plt.plot(time, acceFusedResponse[0, :], '--', color=[1, 0, 0, 0.5], label="Acceleration response N")
        # plt.ylim(-20,20)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(time, acceNED[1, :], 'g', linewidth=2, label="Acceleration command E")
        plt.plot(time, acceFusedResponse[1, :], '--', color=[0, 1, 0, 0.5], label="Acceleration response E")
        # plt.ylim(-20,20)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(time, acceNED[2, :], 'b', linewidth=2, label="Acceleration command D")
        plt.plot(time, acceFusedResponse[2, :], '--', color=[0, 0, 1, 0.5], label="Acceleration response D")
        plt.xlabel('Time (s)')
        # plt.ylim(-20,20)
        plt.legend()
        
        plt.savefig(os.path.join(self.folderPath, 'AccelerationResponse.png'))
        plt.close()

    def plotVelGradientAndAcceleration(self):
        self.createFigure()
        acceResponse = np.zeros((len(self.data), 3))
        velNED = np.zeros((len(self.data), 3))
        time = [d['t'] for d in self.data]

        for i in range(len(self.data)):
            velNED[i, 0:3] = np.squeeze(self.data[i]['meVelocity'])
            acceResponse[i, 0:3] = np.squeeze(self.data[i]['meAccelerationNED'])

        dt = np.gradient(time)
        vel_diff = np.gradient(velNED, axis=0) / dt[:, np.newaxis]

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        titles = ['N', 'E', 'D']

        for i in range(3):
            axs[i].plot(time, vel_diff[:, i], label='Velocity gradient')
            axs[i].plot(time, acceResponse[:, i], label='Acceleration')
            axs[i].set_title(f'{titles[i]}: Velocity gradient vs Acceleration')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Value')
            axs[i].legend()
            axs[i].grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folderPath, 'VelGradientAndAcceleration.png'))
        plt.close()

    def plotPosGradientAndVelocity(self):
        self.createFigure()
        posNED = np.zeros((len(self.data), 3))
        velNED = np.zeros((len(self.data), 3))
        time = [d['t'] for d in self.data]

        for i in range(len(self.data)):
            velNED[i, 0:3] = np.squeeze(self.data[i]['meVelocity'])
            posNED[i, 0:3] = np.squeeze(self.data[i]['mePositionNED'])

        dt = np.gradient(time)
        pos_diff = np.gradient(posNED, axis=0) / dt[:, np.newaxis]

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        titles = ['N', 'E', 'D']

        for i in range(3):
            axs[i].plot(time, pos_diff[:, i], label='Position gradient')
            axs[i].plot(time, velNED[:, i], label='Velocity')
            axs[i].set_title(f'{titles[i]}: Position gradient vs Velocity')
            axs[i].set_xlabel('Time (s)')
            axs[i].set_ylabel('Value')
            axs[i].legend()
            axs[i].grid()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folderPath, 'PosGradientAndVelocity.png'))
        plt.close()
    
    def plotMeasurementUseAndLosAngle(self):
        time = [d['t'] for d in self.data]
        measurementUse = []
        for data in self.data:
            if not len(data['measurementUse']) == 0:
                if data['measurementUse'][1] < 0:
                    data['measurementUse'][1] += 2 * np.pi
                measurementUse.append(data['measurementUse'])
        
        measurementUse = np.array(measurementUse)
        
        num_data_points = len(self.data)
        targetPosition = np.zeros((3, num_data_points))
        mePosition = np.zeros((3, num_data_points))
        relativePosition = np.zeros((3, num_data_points))
        losAngle = np.zeros((2, num_data_points))
        
        for i in range(num_data_points):
            targetPosition[:, i] = np.squeeze(self.data[i]['targetPosition'])
            mePosition[:, i] = np.squeeze(self.data[i]['mePositionENU'])
            relativePosition[:, i] = targetPosition[:, i] - mePosition[:, i]
            losAngle[:, i] = np.array([
                np.arctan2(relativePosition[2, i], np.sqrt(relativePosition[0, i] ** 2 + relativePosition[1, i] ** 2)),
                np.arctan2(relativePosition[1, i], relativePosition[0, i])
            ])
            
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))
        for num, direction in enumerate(['elevation angle', 'azimuth angle']):
            ax[num].plot(time[int(self.data[0]['timeDelay'] / self.data[0]['tStep']):], np.rad2deg(measurementUse[int(self.data[0]['timeDelay'] / self.data[0]['tStep']):, num]), linewidth=2.0,
                        label='Measurement of ' + direction)
            ax[num].plot(time, np.rad2deg(losAngle[num, :]), linewidth=2.0,
                        label='Real ' + direction)
            ax[num].set_xlabel('Time (s)')
            ax[num].set_ylabel('Angle (deg)')
            ax[num].legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.folderPath, 'measurementUseAndLosAngle.png'))
        plt.close()

    def printMissDistance(self):
        time = [d['t'] for d in self.data]
        missDistance = [d['missDistance'] for d in self.data]
        plt.plot(time, missDistance)
        plt.savefig(os.path.join(self.folderPath, 'missDistance.png'))
        plt.close()
        print(f"miss distance = {missDistance[-1]}")

    def createGif(self):
        frames = np.arange(start=0, stop=len(self.data), step=1)
        tq = tqdm.tqdm(total=len(frames))
        ekfState = np.zeros((6, len(self.data)))
        targetPosition = np.zeros((3, len(self.data)))
        mePosition = np.zeros((3, len(self.data)))

        for i in range(len(self.data)):
            ekfState[0:6, i] = np.squeeze(self.data[i]['ekfState'])
            targetPosition[0:3, i] = np.squeeze(self.data[i]['targetPosition'])
            mePosition[0:3, i] = np.squeeze(self.data[i]['mePositionENU'])

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
        
        meGL, = ax.plot([], [], [], 'b', linewidth=2, label=self.guidanceLawName)
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
        fileName = 'Estimating.gif'
        pathStr = os.path.join(self.folderPath, fileName)
        anim.save(pathStr, writer='pillow')
        tq.close()

    def createFigure(self):
        plt.figure()

    def decorateFigure(self, xlabelText, ylabelText, fileName):
        plt.grid(True)
        plt.xlabel(xlabelText, fontsize=12)
        plt.ylabel(ylabelText, fontsize=12)
        plt.savefig(os.path.join(self.folderPath, fileName))
        plt.close()

    def plotAll(self):
        if self.model == 'M300':
            self.printMissDistance()
            self.plotLosRate()
            self.plotMeasurementUseAndLosAngle()
            self.plotLookAngleScatterWithTime()
            self.plotLookAngleWithTime()
            self.plotLookAngleWithRelativeDistance()
            self.plotMeasurementUse()
            self.plotTrajectory()
            self.plotAcceResponse()
            self.plotVelResponse()
            self.plotVelGradientAndAcceleration()
            self.plotPosGradientAndVelocity()
            self.plotCloseVelocity()
            self.plotXYZTrajectories()
            self.plotLeadAngle()
            self.plotTargetPositionError()
            self.plotTargetVelocityError()
            self.plotAttitudeMeENU()
            self.plotEulerMeAndCommandENU()
            self.createGif()
            if self.useCamera:
                self.plotGimbalAngleAndAttitudeAngle()
        else:
            self.plotRelativeDistanceError()
            self.plotRelativeVelocityError()
            self.plotTrajectory()
            if self.model == 'FixedWing':
                self.plotAttitudeMeENU()
            self.plotAMy()
            self.plotAMz()
            self.plotXYZTrajectories()
            self.plotTgo()
            self.plotZeroEffortMiss()
            self.plotVelocityAndAcceleration()
            self.plotMeasurement()
            self.plotMeasurementUse()
            self.plotTargetPositionError()
            self.plotTargetVelocityError()
            self.plotLeadAngle()
            self.createGif()


if __name__ == '__main__':
    psr = PlotSingleRun(guidanceLawName = 'OEHG_test')
    psr.findLastDir()
    psr.loadData()
    psr.plotAll()
 