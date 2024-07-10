#!/usr/bin/env python3

import copy
import datetime
import json
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
    BOOST = 9


def stepEntrance(method):
    def wrapper(self, *args, **kwargs):
        self.stateStartTime = self.getTimeNow()
        return method(self, *args, **kwargs)
    return wrapper



class SingleRun:
    def __init__(self, **kwargs):
        rospy.init_node('observability_enhancement', anonymous=True)
        self.algorithmName = 'observability_enhancement'

        self.guidanceOn = rospy.get_param('guidanceOn') == True
        self.guidanceLawName = rospy.get_param('GL', 'PN')

        self.expectedSpeed = rospy.get_param('expectedSpeed')
        self.takeoffHeight = rospy.get_param('takeoffHeight')
        self.yawDegNED = rospy.get_param('yawDegNED')
        self.guidanceLength = rospy.get_param('guidanceLength')

        self.safetyMinHeight = rospy.get_param('safetyMinHeight')
        self.safetyMaxHeight = rospy.get_param('safetyMaxHeight')
        self.safetyMinDescendHeight = rospy.get_param('safetyMinDescendHeight')
        self.safetyMaxDescendVelocity = rospy.get_param('safetyMaxDescendVelocity')
        self.safetyMaxAscendVelocity = rospy.get_param('safetyMaxAscendVelocity')

        self.reallyTakeoff = rospy.get_param('takeoff', False) == True

        self.tStep = rospy.get_param('tStep', 0.02)
        self.tUpperLimit = rospy.get_param('tUpperLimit', 100)

        self.outliers = rospy.get_param('outliers', False) == True
        self.timeDelay = rospy.get_param('timeDelay', 0.0)

        self.throttleTestOn = rospy.get_param('throttleTestOn') == True
        self.throttleTestHeight = rospy.get_param('throttleTestHeight')
        self.throttleTestChangeTime = rospy.get_param('throttleTestChangeTime')
        self.throttleTestMin = rospy.get_param('throttleTestMin')
        self.throttleTestMax = rospy.get_param('throttleTestMax')

        self.useCamera = rospy.get_param('useCamera', False)
        self.cameraPitch = rospy.get_param('cameraPitch', 0.0)

        if self.throttleTestOn:
            self.takeoffHeight = self.throttleTestHeight
        
        self.loopNum = 1
        self.t = 0
        self.nn = 3

        self.yawRadNED = np.deg2rad(self.yawDegNED)
        self.yawRadENU = yawRadNED2ENU(self.yawRadNED)
        self.unitVector = np.array([math.sin(self.yawRadNED), math.cos(self.yawRadNED)])
        self.targetState = np.concatenate([self.guidanceLength * self.unitVector, [20.0], np.zeros(3)])
        self.uTarget = np.array([0, 0, 0])

        self.u = None
        self.data = []
        self.z = None
        self.zUse = []
        self.endControlFlag = 0
        self.real = False

        self.measurementNoise = np.deg2rad(0.5)

        self.state = State.INIT
        self.taskStartTime = time.time()
        self.stateStartTime = time.time()
        self.taskTime = 0
        self.stateTime = 0

        self.takeoffPointENU = np.array([0, 0, self.takeoffHeight])
        self.preparePointENU = np.concatenate([-self.unitVector * self.guidanceLength, np.array([self.takeoffHeight])])
        self.initialVelocityENU = np.array([
            self.expectedSpeed * np.sin(self.yawRadNED), 
            self.expectedSpeed * np.cos(self.yawRadNED), 
            0.0
        ])

        self.cmdAccNED = np.zeros(3)
        self.cmdRPYRadENU = np.zeros(3)
        self.cmdRPYRadNED = np.zeros(3)

        rospy.init_node(self.algorithmName, anonymous=True)
        self.me = M300('suav')
        self.spinThread = threading.Thread(target=lambda: rospy.spin())
        self.spinThread.start()

        rospack = rospkg.RosPack()
        self.packagePath = rospack.get_path(self.algorithmName)
        self.timeStr = kwargs.get('prefix', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        self.folderName = os.path.join(
            self.packagePath, 
            'data', 
            self.timeStr, 
            self.guidanceLawName
        )
        os.makedirs(self.folderName, exist_ok=True)
        self.ekf = EKF(self.targetState, self.measurementNoise)

        cliOutputFile = open(os.path.join(self.folderName, 'output.txt'), "w")

        def custom_print(*args, **kwargs):
            message = " ".join(map(str, args))
            rprint(message, **kwargs)
            cliOutputFile.write(message + "\n")
            cliOutputFile.flush()

        builtins.print = custom_print

        print(f"Simulation Condition: {self.takeoffHeight = }, {self.expectedSpeed = }, {self.targetState = }")
        print(f'{self.guidanceOn = }, {self.throttleTestOn = }')
        print(f'{self.guidanceLawName = }, {self.me.useRTK = }')

        if self.reallyTakeoff:
            input('Really going to takeoff!!! Input anything to confirm...')
        else:
            input('No takeoff, input anything to confirm...')

        if self.guidanceLawName == 'PN':
            self.guidanceLaw = PN()
        elif self.guidanceLawName == 'PN_test':
            self.guidanceLaw = PN_test(expectedVc=self.expectedSpeed)
        elif self.guidanceLawName == 'OEG':
            self.guidanceLaw = OEG()
        elif self.guidanceLawName == 'RAIM':
            self.guidanceLaw = RAIM()
        elif self.guidanceLawName == 'OEHG':
            self.guidanceLaw = OEHG()
        elif self.guidanceLawName == 'OEHG_test':
            self.guidanceLaw = OEHG_test(expectedVc=self.expectedSpeed)
        else:
            raise ValueError("Invalid guidance law name")

        # Save parameters to file
        params = {
            'GL': self.guidanceLawName,
            'expectedSpeed': self.expectedSpeed,
            'takeoffHeight': self.takeoffHeight,
            'yawDegNED': self.yawDegNED,
            'guidanceLength': self.guidanceLength,
            'takeoff': self.reallyTakeoff,
            'tStep': self.tStep,
            'tUpperLimit': self.tUpperLimit,
            'outliers': self.outliers,
            'timeDelay': self.timeDelay
        }

        with open(os.path.join(self.folderName, 'params.json'), 'w') as f:
            json.dump(params, f, indent=4)

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
        self.throttleMin = self.throttleTestMin
        self.throttleMax = self.throttleTestMax
        self.throttle = (self.throttleMax + self.throttleMin) / 2.0
        self.changeTime = self.throttleTestChangeTime
        self.changeStep = self.throttleTestChangeTime
        self.throttleTestAdjustPosition = False

    @stepEntrance
    def toStepHover(self):
        self.state = State.HOVER

    @stepEntrance
    def toStepBoost(self):
        self.state = State.BOOST

    def stepInit(self):
        if not self.me.set_local_position():
            self.toStepLand()
            return
        if not self.reallyTakeoff:
            self.toStepTakeoff()
            return
        if not self.me.obtain_control() or not self.me.monitored_takeoff():
            self.toStepLand()
            return
        print('Initialization completed')
        self.toStepTakeoff()

    def stepTakeoff(self):
        if self.reallyTakeoff:
            self.me.positionENUControl(self.takeoffPointENU, self.yawRadENU)
        print(f'{self.me.mePositionENU = }')
        print(f'{self.takeoffPointENU = }')
        print(f'{self.me.distanceToPointENU(self.takeoffPointENU) = }')
        if self.me.nearPositionENU(self.takeoffPointENU) and self.me.nearSpeed(0):
            if self.throttleTestOn:
                self.toStepThrottleTest()
            elif self.guidanceOn:
                self.toStepPrepare()
            else:
                self.toStepLand()

    def stepPrepare(self):
        if self.reallyTakeoff:
            self.me.positionENUControl(self.preparePointENU, self.yawRadENU)
        print(f'{self.me.mePositionENU = }')
        print(f'{self.preparePointENU = }')
        print(f'{self.me.distanceToPointENU(self.preparePointENU) = }')
        if self.me.nearPositionENU(self.preparePointENU) and self.me.nearSpeed(0):
            self.toStepBoost()

    def stepGuidance(self):
        if self.stateTime >= 100.0:
            self.toStepLand()
            return
        
        if np.linalg.norm(self.getRelativePosition()) <= 0.1:
            self.toStepHover()
            return
        
        if self.me.mePositionENU[2] <= self.targetState[2]:
            self.toStepHover()
            return
        
        if self.me.underHeight(0.5):
            self.toStepLand() 
            return

        if not self.safetyModule():
            self.toStepLand()
            return
            
        self.getMeasurement()
        if self.loopNum > np.floor(self.timeDelay / self.tStep): 
            if (len(self.data) > 1 ) and (not len(self.data[-1]['measurementUse']) == 0 ):
                self.MeasurementFiltering()
            self.ekf.newFrame(self.tStep, self.uTarget, self.zUse)
        print(f"estimate position ENU = {pointString(self.ekf.x[:3])}")
        self.u = self.guidanceLaw.getU(
                    self.getRelativePosition(),
                    self.getRelativeVelocity(), 
                    self.me.getVelocityENU()
                ).reshape(3)
        print(f'uENU = {pointString(self.u)}')
        assert np.all(np.isfinite(self.u)), "u is not finite"
        
        self.cmdAccNED = enu2ned(self.u)
        thrust, self.cmdRPYRadENU = accENUYawENU2EulerENUThrust(
            accENU=self.u, 
            yawRadENU=self.yawRadENU, 
            hoverThrottle=self.me.hoverThrottle
        )
        self.cmdRPYRadNED = rpyENU2NED(self.cmdRPYRadENU)
        
        if self.reallyTakeoff:
            self.me.acc2attENUControl(thrust, self.cmdRPYRadENU)

        self.loopNum += 1
        self.log()

    def stepBack(self):
        if self.reallyTakeoff:
            self.me.positionENUControl(self.takeoffPointENU, self.yawRadENU)
        if self.me.nearPositionENU(self.takeoffPointENU):
            self.toStepLand()

    def stepLand(self):
        self.me.land()
        if self.me.underHeight(z=0.2):
            self.toStepEnd()

    def stepThrottleTest(self):
        if self.stateTime >= 100.0:
            self.toStepLand()
            return
        
        if not self.safetyModule():
            self.toStepLand()
            return
        
        if not self.me.nearPositionENU(self.takeoffPointENU, tol=10):
            self.throttleTestAdjustPosition = True
        
        if self.me.nearPositionENU(self.takeoffPointENU) and self.me.nearSpeed(0):
            self.throttleTestAdjustPosition = False
        
        if self.throttleTestAdjustPosition:
            print('ADJUSTING POSITION!!!')
            if self.reallyTakeoff:
                self.me.positionENUControl(self.takeoffPointENU, self.yawRadENU)
            print(f'{self.me.mePositionENU = }')
            print(f'{self.takeoffPointENU = }')
            print(f'{self.me.distanceToPointENU(self.takeoffPointENU) = }')
        
        self.throttle = (self.throttleMin + self.throttleMax) / 2.0
        print(f'Between {self.throttleMin:.3f} and {self.throttleMax:.3f}: try {self.throttle:.3f}')
        print(f'{self.me.meAccelerationENUFused[2] = }')
        if self.stateTime >= self.changeTime:
            if self.me.meAccelerationENUFused[2] < 0:
                self.throttleMin = self.throttle
            else:
                self.throttleMax = self.throttle
            self.changeTime += self.changeStep
                
        if self.throttleMax - self.throttleMin < 0.01:
            print(f'Throttle test result: between {self.throttleMin} and {self.throttleMax}')
            self.me.hoverThrottle = self.throttleMax
            if self.guidanceOn:
                self.toStepPrepare()
            else:
                self.toStepLand()
            return

        if self.throttleMax - self.throttleTestMin < 0.5:
            print(f'Throttle test failed: range to high')
            self.toStepLand()
            return

        if self.throttleTestMax - self.throttleMin < 0.5:
            print(f'Throttle test failed: range to low')
            self.toStepLand()
            return

        self.me.acc2attENUControl(self.throttle, np.array([0.0, 0.0, self.yawRadENU]))

    def stepHover(self):
        self.me.hoverWithYaw(self.yawRadENU)
        if self.stateTime >= 5.0:
            self.toStepBack()

    def stepBoost(self):
        if self.reallyTakeoff:
            self.me.velocityENUControl(self.initialVelocityENU, self.yawRadENU)
        if np.dot(self.me.mePositionENU[:2], self.unitVector) > 0:
            print(f"stepGuidanceInitialMePositionNED = {self.me.mePositionNED}")
            print(f"stepGuidanceInitialMeVelocityNED = {self.me.meVelocityNED}")
            self.toStepGuidance()

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
        elif self.state == State.BOOST:
            self.stepBoost()

    def print(self):
        console.clear()
        print('-' * 20)
        print(f'UAV {self.me.name}: state {self.state.name}')
        print(f'Total time: {self.taskTime:.2f}, state time: {self.stateTime:.2f}')
        # print(f'Armed: {"YES" if self.me.isArmed() else "NO"}({self.me.status.arming_state})')
        self.me.printMe()

    def safetyModule(self):
        if self.me.underHeight(self.safetyMinHeight):
            print('Safety module: too low, quit guidance...')
            return False
        if self.me.aboveHeight(self.safetyMaxHeight):
            print('Safety module: too high, quit guidance...')
            return False
        if self.me.underHeight(self.safetyMinDescendHeight) and self.me.meVelocityENU[2] < -self.safetyMaxDescendVelocity:
            print('Safety module: too quick desending, quit guidance...')
            return False
        if self.me.meVelocityENU[2] > self.safetyMaxAscendVelocity:
            print('Safety module: too quick asending, quit guidance...')
            return False
        return True

    def run(self):
        while self.state != State.END and not rospy.is_shutdown():
            self.taskTime = time.time() - self.taskStartTime
            self.stateTime = time.time() - self.stateStartTime

            self.print() 

            self.me.sendHeartbeat()
            self.controlStateMachine()

            time.sleep(self.tStep)
            self.t += self.tStep

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
        if self.useCamera:
            lookAngle = self.getLookAngle()
            LOSdirectionCameraFRD = np.array([1, np.tan(lookAngle[1]), np.tan(lookAngle[0])])/np.linalg.norm(np.array([1, np.tan(lookAngle[1]), np.tan(lookAngle[0])]))
            LOSdirectionBodyFRD = camera2bodyFRDrotationMatrix(np.deg2rad(self.cameraPitch)) @ LOSdirectionCameraFRD
            LOSdirectionNED = frd2nedRotationMatrix(self.me.meRPYRadNED) @ LOSdirectionBodyFRD
            LOSdirectionENU = ned2enu(LOSdirectionNED)
            self.z = np.array([np.arctan2(LOSdirectionENU[2], np.sqrt(LOSdirectionENU[0] ** 2 + LOSdirectionENU[1] ** 2)),
                            np.arctan2(LOSdirectionENU[1], LOSdirectionENU[0])])
        else:
            relPos = self.getRelativePosition(True)
            self.z = np.array([np.arctan2(relPos[2], np.sqrt(relPos[0] ** 2 + relPos[1] ** 2)),
                            np.arctan2(relPos[1], relPos[0])]) + np.random.randn() * self.measurementNoise
            print(f"measurementsDeg = {np.rad2deg(self.z)}")
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
        if self.useCamera:
            elevationAngle = self.me.elevationAngle
            azimuthAngle = self.me.azimuthAngle
        else:
            losDirectionNED = enu2ned(self.getRelativePosition(True).flatten()/np.linalg.norm(self.getRelativePosition(True)))
            print(f"losDirectionNED = {losDirectionNED}")
            rotationMatrix = ned2frdRotationMatrix(self.me.meRPYRadNED)
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
        currentData['meRPYENU'] = copy.copy(self.me.meRPYRadENU)
        currentData['meRPYNED'] = copy.copy(self.me.meRPYRadNED)
        currentData['cmdRPYENU'] = copy.copy(self.cmdRPYRadENU)
        currentData['cmdRPYNED'] = copy.copy(self.cmdRPYRadNED)
        currentData['meAccelerationFusedNED'] = copy.copy(enu2ned(self.me.meAccelerationENUFused))
        currentData['meAccelerationNED'] = copy.copy(self.me.getAccelerationNED())
        currentData['meAcceCommandNED'] = copy.copy(self.cmdAccNED)
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


def main():
    sr = SingleRun()
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
    main()
