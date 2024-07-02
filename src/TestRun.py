#!/usr/bin/env python3

import argparse
import copy
import datetime
import os
import pickle
import sys
import time
import threading
from enum import Enum

import numpy as np
import rospy
import rospkg

from EKF import EKF
from GuidanceLaws.OEHG import OEHG
from GuidanceLaws.OEHG_test import OEHG_test
from GuidanceLaws.PN import PN
from GuidanceLaws.PN_test import PN_test
from GuidanceLaws.OEG import OEG
from GuidanceLaws.RAIM import RAIM
from PlotSingleRun import PlotSingleRun

import builtins
from rich.console import Console
from rich import print as rprint

console = Console()

from models.M300.M300 import M300

from Utils import *


class State(Enum):
    INIT = 0
    TAKEOFF = 1
    PREPARE = 2
    GUIDANCE = 3
    LAND = 4
    BACK = 5
    END = 6
    THROTTLE_TEST = 7
    HOVER = 8


def stepEntrance(method):
    def wrapper(self, *args, **kwargs):
        self.stateStartTime = self.getTimeNow()
        return method(self, *args, **kwargs)
    return wrapper



class SingleRun:
    def __init__(self, **kwargs):
        self.algorithmName = 'observability_enhancement'
        self.guidanceLawName = kwargs.get('GL', 'PN')
        self.model = kwargs.get('model', 'useFixedWingModel')
        self.monteCarlo = kwargs.get('monteCarlo', False)
        self.tStep = 0.02
        self.tUpperLimit = 100
        self.loopNum = 1
        self.t = 0
        self.nn = 3

        vel = kwargs.get('vel')
        prepareLength = vel * 1.0  #1m/s:0.5m; 2m/s:1.6m; 
        guidanceLength = kwargs.get('disGuidance')
        totalLength = prepareLength + guidanceLength

        yawDegNED = kwargs.get('yawDegNED') 
        self.yawRadNED = np.deg2rad(yawDegNED)
        self.yawRadENU = yawRadNED2ENU(self.yawRadNED)
        self.targetState = np.array([
            totalLength * math.sin(self.yawRadNED),
            totalLength * math.cos(self.yawRadNED),
            0,
            0,
            0,
            0
        ])
        self.uTarget = np.array([0, 0, 0])

        self.u = None
        self.data = []
        self.z = None
        self.zUse = []
        self.endControlFlag = 0
        self.real = False

        timeStr = kwargs.get('prefix', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

        self.outliers = kwargs.get('outliers', False)
        self.timeDelay = kwargs.get('timeDelay', 0.0)
        self.measurementNoise = np.deg2rad(0.5)

        self.state = State.INIT
        self.taskStartTime = time.time()
        self.stateStartTime = time.time()
        self.taskTime = 0
        self.stateTime = 0

        self.reallyTakeoff = kwargs.get('takeoff', False)

        self.takeoffPointENU = np.array([0.0, 0.0, kwargs.get('height')])
        self.expectedSpeed = vel
        self.initialVelocityENU = np.array([
            self.expectedSpeed * math.sin(self.yawRadNED), 
            self.expectedSpeed * math.cos(self.yawRadNED), 
            0.0
        ])

        rospy.init_node(self.algorithmName, anonymous=True)
        self.me = M300('suav')
        self.spinThread = threading.Thread(target=lambda: rospy.spin())
        self.spinThread.start()

        rospack = rospkg.RosPack()
        self.packagePath = rospack.get_path(self.algorithmName)
        self.folderName = os.path.join(self.packagePath, 'data', timeStr, self.guidanceLawName)
        os.makedirs(self.folderName, exist_ok=True)
        self.ekf = EKF(self.targetState, self.measurementNoise)

        cliOutputFile = open(os.path.join(self.folderName, 'output.txt'), "w")

        def custom_print(*args, **kwargs):
            message = " ".join(map(str, args))
            rprint(message, **kwargs)
            cliOutputFile.write(message + "\n")
            cliOutputFile.flush()

        builtins.print = custom_print

        if self.reallyTakeoff:
            input('Really going to takeoff, input anything to confirm...')

        if self.guidanceLawName == 'PN':
            self.guidanceLaw = PN()
        elif self.guidanceLawName == 'PN_test':
            self.guidanceLaw = PN_test()
        elif self.guidanceLawName == 'OEG':
            self.guidanceLaw = OEG()
        elif self.guidanceLawName == 'RAIM':
            self.guidanceLaw = RAIM()
        elif self.guidanceLawName == 'OEHG':
            self.guidanceLaw = OEHG()
        elif self.guidanceLawName == 'OEHG_test':
            self.guidanceLaw = OEHG_test(expectedVc=kwargs.get('vel'))
        else:
            raise ValueError("Invalid guidance law name")
        
        print(f"Simulation Condition: takeoffHeight = {kwargs.get('height')}, expectedSpeed = {self.expectedSpeed}, targetState = {self.targetState}")
        print(f"GL: {self.guidanceLawName}")

    def getTimeNow(self):
        return time.time()

    @stepEntrance
    def toStepTakeoff(self):
        self.state = State.TAKEOFF
        self.me.setPositionControlMode()

    @stepEntrance
    def toStepPrepare(self):
        self.state = State.PREPARE
        self.me.setVelocityControlMode()

    @stepEntrance
    def toStepGuidance(self):
        self.state = State.GUIDANCE
        self.me.setAttitudeControlMode()

    @stepEntrance
    def toStepBack(self):
        self.state = State.BACK
        self.me.setPositionControlMode()

    @stepEntrance
    def toStepLand(self):
        self.state = State.LAND

    @stepEntrance
    def toStepEnd(self):
        self.state = State.END

    @stepEntrance
    def toStepThrottleTest(self):
        self.state = State.THROTTLE_TEST
        self.throttleMin = 31.4
        self.throttleMax = 31.6
        self.throttle = (self.throttleMax + self.throttleMin) / 2.0
        self.changeTime = 10.0

    @stepEntrance
    def toStepHover(self):
        self.state = State.HOVER

    def stepInit(self):
        if not self.reallyTakeoff:
            self.toStepTakeoff()
            return
        if not self.me.set_local_position() or not self.me.obtain_control() or not self.me.monitored_takeoff():
            self.toStepLand()
            return
        print('Initialization completed')
        self.toStepTakeoff()

    def stepTakeoff(self):
        self.me.positionENUControl(self.takeoffPointENU, self.yawRadENU)
        if self.me.nearPositionENU(self.takeoffPointENU) and self.me.nearSpeed(0.0):
            print(f"stepPrepareInitialMePositionNED = {self.me.mePositionNED}")
            print(f"stepPrepareInitialMeVelocityNED = {self.me.meVelocityNED}")
            self.toStepPrepare()

    def stepPrepare(self):
        self.me.velocityENUControl(self.initialVelocityENU, self.yawRadENU)
        if self.me.nearSpeed(self.expectedSpeed):
            print(f"stepGuidanceInitialMePositionNED = {self.me.mePositionNED}")
            print(f"stepGuidanceInitialMeVelocityNED = {self.me.meVelocityNED}")
            self.toStepGuidance()

    def stepGuidance(self):
        if self.stateTime >= 100.0:
            self.toStepLand()
        elif np.linalg.norm(self.getRelativePosition()) <= 0.1:
            self.toStepHover()
        elif self.me.underHeight(0.1):
            self.toStepLand() 
        else:
            self.getMeasurement()
            if self.loopNum > np.floor(self.timeDelay / self.tStep): 
                if (not len(self.data) == 0 ) and (not len(self.data[-1]['measurementUse']) == 0 ):
                    self.MeasurementFiltering()
                self.ekf.newFrame(self.tStep, self.uTarget, self.zUse)
            print(f"estimate position ENU = {pointString(self.ekf.x[:3])}")
            self.u = self.guidanceLaw.getU(self.getRelativePosition(),
                                        self.getRelativeVelocity(), self.me.getVelocityENU()).reshape(3)
            print(f'uENU = {pointString(self.u)}')
            assert np.all(np.isfinite(self.u)), "u is not finite"
            self.me.acc2attENUControl(self.u, self.yawRadENU)
            self.loopNum += 1
            self.log()

    def stepBack(self):
        self.me.positionENUControl(self.takeoffPointENU)
        if self.me.nearPositionENU(self.takeoffPointENU):
            self.toStepLand()

    def stepLand(self):
        self.me.land()
        if self.me.underHeight(z=0.2):
            self.toStepEnd()

    def stepThrottleTest(self):
        if self.stateTime >= 100.0:
            self.toStepLand()
        elif np.linalg.norm(self.getRelativePosition()) <= 1:
            self.toStepLand()
        elif self.me.underHeight(0.1):
            self.toStepLand() 
        else:
            self.throttle = (self.throttleMin + self.throttleMax) / 2.0
            self.me.hoverThrottle = self.throttle
            print(f'{self.me.hoverThrottle = }')
            if self.stateTime >= self.changeTime:
                if self.me.meAccelerationNED[2] > 0:
                    self.throttleMin = self.throttle
                else:
                    self.throttleMax = self.throttle
                self.changeTime += 10.0
                    
            if self.throttleMax - self.throttleMin < 0.01:
                self.toStepLand()
            self.me.acc2attENUControl(np.zeros(3))

    def stepHover(self):
        self.me.hoverWithYaw(self.yawRadENU)
        if self.stateTime >= 5.0:
            self.toStepLand()

    def controlStateMachine(self):
        if self.state == State.INIT:
            self.stepInit()
        elif self.state == State.TAKEOFF:
            self.stepTakeoff()
        elif self.state == State.PREPARE:
            self.stepPrepare()
        elif self.state == State.GUIDANCE:
            self.stepGuidance()
        elif self.state == State.BACK:
            self.stepBack()
        elif self.state == State.LAND:
            self.stepLand()
        elif self.state == State.END:
            exit(0)
        elif self.state == State.THROTTLE_TEST:
            self.stepThrottleTest()
        elif self.state == State.HOVER:
            self.stepHover()

    def print(self):
        print('-' * 20)
        print(f'UAV {self.me.name}: state {self.state.name}')
        print(f'Total time: {self.taskTime:.2f}, state time: {self.stateTime:.2f}')
        # print(f'Armed: {"YES" if self.me.isArmed() else "NO"}({self.me.status.arming_state})')
        self.me.printMe()

    def run(self):
        if self.model == 'useGroundTruth':
            while self.state != State.END and not rospy.is_shutdown():
                self.taskTime = time.time() - self.taskStartTime
                self.stateTime = time.time() - self.stateStartTime

                self.print()

                self.me.sendHeartbeat()
                self.controlStateMachine()

                time.sleep(self.tStep)
                self.t += self.tStep
        else:
            for t in np.arange(0, self.tUpperLimit + self.tStep, self.tStep):
                self.getMeasurement()
                if self.loopNum > np.floor(self.timeDelay / self.tStep):
                    if (not len(self.data) == 0) and (not len(self.data[-1]['measurementUse']) == 0):
                        self.MeasurementFiltering()
                    if np.linalg.norm(self.getRelativePosition()) > 10 and self.endControlFlag == 0:
                        self.ekf.newFrame(self.tStep, self.uTarget, self.zUse)
                if np.linalg.norm(self.getRelativePosition()) > 10 and self.endControlFlag == 0:
                    self.u = self.guidanceLaw.getU(self.getRelativePosition(), self.getRelativeVelocity(),
                                                   self.me.getVelocityENU())
                else:
                    self.u = np.array([[0], [0], [0]])
                    self.endControlFlag = 1
                assert np.all(np.isfinite(self.u)), "u is not finite"
                self.log()
                self.update(self.u)
                if self.getCloseVelocity() < 0:
                    break

    def saveLog(self):
        self.fileName = os.path.join(self.folderName, 'data.pkl')
        with open(self.fileName, "wb") as file:
            pickle.dump(self.data, file)

        print(f"Data saved to {self.fileName}")

    def update(self, u):
        self.me.update(u, self.tStep)
        self.t += self.tStep
        self.loopNum += 1

    def getMeasurement(self):
        relPos = self.getRelativePosition(True)
        self.z = np.array([np.arctan2(relPos[2], np.sqrt(relPos[0] ** 2 + relPos[1] ** 2)),
                           np.arctan2(relPos[1], relPos[0])]) + np.random.randn() * self.measurementNoise
        if self.outliers:
            self.addOutliers()
        if self.loopNum > np.floor(self.timeDelay / self.tStep):
            if self.timeDelay > 0:
                self.ekf.getMeState(
                    self.data[int(self.loopNum - np.floor(self.timeDelay / self.tStep) - 1)]['mePositionENU'])
                self.zUse = self.data[int(self.loopNum - np.floor(self.timeDelay / self.tStep) - 1)]['measurement']
            else:
                self.ekf.getMeState(self.me.getPositionENU())
                self.zUse = self.z

    def MeasurementFiltering(self):
        if np.abs(self.zUse[0] - self.data[-1]['measurementUse'][0]) > np.deg2rad(8) or \
                np.abs(self.zUse[1] - self.data[-1]['measurementUse'][1]) > np.deg2rad(8):
            self.zUse = self.data[-2]['measurementUse']

    def getRelativePosition(self, real = False):
        if real:
            relativePosition = self.targetState[:3] - self.me.getPositionENU()
        else:
            relativePosition = self.ekf.x[:3] - self.me.getPositionENU()
        return relativePosition

    def getRelativeVelocity(self, real = False):
        if real:
            relativeVelocity = self.targetState[3:] - self.me.getVelocityENU()
        else:
            relativeVelocity = self.ekf.x[3:] - self.me.getVelocityENU()
        return relativeVelocity

    def getCloseVelocity(self):
        return -np.dot(self.getRelativePosition().flatten(), self.getRelativeVelocity().flatten()) / \
            np.linalg.norm(self.getRelativePosition())

    def getLeadAngle(self):
        return np.arccos(self.getCloseVelocity() / np.linalg.norm(self.getRelativeVelocity()))

    def getTgo(self):
        return np.linalg.norm(self.getRelativePosition()) / np.linalg.norm(self.me.getVelocityENU()) * \
            (1 + self.getLeadAngle() ** 2 / (2 * (2 * self.nn) - 1))

    def getZeroEffortMiss(self):
        return self.getRelativePosition() + self.getRelativeVelocity() * self.getTgo()

    def getLambda(self):
        return (np.cross(self.me.getVelocityENU().flatten(), self.getRelativePosition().flatten()) / \
                np.linalg.norm(self.getRelativePosition()) ** 2).reshape(3, 1)

    def getMissDistance(self):
        return np.abs(np.linalg.norm(self.getRelativePosition()) ** 2 * np.linalg.norm(self.getLambda()) /
                      np.sqrt(self.getCloseVelocity() ** 2 + np.linalg.norm(self.getRelativePosition()) ** 2 *
                              np.linalg.norm(self.getLambda()) ** 2))
    
    def getLookAngle(self):
        losDirectionNED = enu2ned(self.getRelativePosition(True).flatten()/np.linalg.norm(self.getRelativePosition(True)))
        print(f"losDirectionNED = {losDirectionNED}")
        rotationMatrix = ned2frdRotationMatrix(self.me.meRPYNED[0], self.me.meRPYNED[1], self.me.meRPYNED[2])
        losDirectionFRD = rotationMatrix @ losDirectionNED
        print(f"losDirectionFRD = {losDirectionFRD}")
        elevationAngle = np.arctan2(losDirectionFRD[2], losDirectionFRD[0])
        azimuthAngle = np.arctan2(losDirectionFRD[1], losDirectionFRD[0])
        return np.array([elevationAngle, azimuthAngle])

    def addOutliers(self):
        if np.abs(self.t - 1000 * self.tStep) < 1e-2 or np.abs(self.t - 2000 * self.tStep) < 1e-2:
            self.z = np.deg2rad([[10], [-70]])

    def log(self):
        currentData = {}
        currentData['t'] = copy.copy(self.t)
        currentData['u'] = copy.copy(self.u)
        currentData['tStep'] = copy.copy(self.tStep)
        currentData['timeDelay'] = copy.copy(self.timeDelay)
        currentData['targetPosition'] = copy.copy(self.targetState[:3])
        currentData['targetVelocity'] = copy.copy(self.targetState[3:])
        currentData['mePositionENU'] = copy.copy(self.me.getPositionENU())
        currentData['mePositionNED'] = copy.copy(enu2ned(self.me.getPositionENU()))
        currentData['meVelocity'] = copy.copy(enu2ned(self.me.getVelocityENU()))
        currentData['meVelocityNorm'] = copy.copy(np.linalg.norm(self.me.getVelocityENU()))
        currentData['lookAngle'] = copy.copy(self.getLookAngle())
        if self.model == 'useGroundTruth':
            currentData['meRPYENU'] = copy.copy(self.me.meRPYENU)
            currentData['meRPYNED'] = copy.copy(self.me.meRPYNED)
            currentData['cmdRPYENU'] = copy.copy(self.me.controlEulerENU)
            currentData['cmdRPYNED'] = copy.copy(self.me.controlEulerNED)
            currentData['meAccelerationNED'] = copy.copy(self.me.getAccelerationNED())
            currentData['meAcceCommandNED'] = copy.copy(self.me.guidanceCommandNED)
        else:
            if self.model == 'useFixWingModel':
                currentData['rollAngleCommand'] = copy.copy(self.me.rollAngleCommand)
        currentData['relativePosition'] = copy.copy(self.getRelativePosition())
        currentData['relativeDistance'] = copy.copy(np.linalg.norm(self.getRelativePosition()))
        currentData['relativeVelocity'] = copy.copy(self.getRelativeVelocity())
        currentData['closeVelocity'] = copy.copy(self.getCloseVelocity())
        currentData['leadAngle'] = copy.copy(self.getLeadAngle())
        currentData['tGo'] = copy.copy(self.getTgo())
        currentData['zeroEffortMiss'] = copy.copy(self.getZeroEffortMiss())
        currentData['lambda'] = copy.copy(self.getLambda())
        currentData['missDistance'] = copy.copy(self.getMissDistance())
        currentData['ekfState'] = copy.copy(self.ekf.x)
        currentData['estimateState'] = copy.copy(
            self.ekf.x - np.concatenate([self.me.getPositionENU(), self.me.getVelocityENU()]))
        currentData['measurement'] = copy.copy(self.z)
        currentData['measurementUse'] = copy.copy(self.zUse)
        
        self.data.append(currentData)


def main(args):
    sr = SingleRun(
        GL='OEHG_test', 
        model='useGroundTruth',
        vel=8.0,
        height=15.0,
        yawDegNED=0,
        disGuidance=20,
        **vars(args)
    )
    sr.run()
    rospy.signal_shutdown('Shutting down')
    sr.spinThread.join()

    print('Run ended, start plotting')

    if len(sr.data) > 0:
        sr.saveLog()

        psr = PlotSingleRun(guidanceLawName='OEHG_test', packagePath=sr.packagePath)
        psr.findLastDir()
        psr.loadData()
        psr.plotAll()

        print('Plotting ended')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ROS node with command line parameters')
    parser.add_argument('--takeoff', action='store_true', help='explicit takeoff command')

    args = parser.parse_args()

    print(args)

    main(args)
