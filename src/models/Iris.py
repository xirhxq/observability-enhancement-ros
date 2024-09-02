#!/usr/bin/env python3

import math

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleAttitudeSetpoint, VehicleCommand, \
    VehicleLocalPosition, VehicleAttitude, VehicleStatus, SensorCombined
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from ..Utils import *

class Iris(Node):
    def __init__(self, name='') -> None:
        super().__init__('iris', namespace=name)
        self.name = name

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.offboardControlMode = OffboardControlMode(velocity=True)
        self.offboardControlModePub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        
        self.setpoint = TrajectorySetpoint()
        self.setpointPub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
        self.attitudeSetpoint = VehicleAttitudeSetpoint()
        self.attitudeSetpointPub = self.create_publisher(
            VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)
        
        self.cmdPub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        
        self.localPosition = VehicleLocalPosition()
        self.imu = SensorCombined()
        self.hoverThrottle = self.get_parameter('hoverThrottle').get_parameter_value().double_value
        self.mePositionENU = np.zeros(3)
        self.mePositionNED = np.zeros(3)
        self.meVelocityENU = np.zeros(3)
        self.meVelocityNED = np.zeros(3)
        self.meSpeed = 0
        self.meAccelerationENU = np.zeros(3)
        self.meAccelerationNED = np.zeros(3)
        self.meAccelerationFRD = np.zeros(3)
        self.localPositionSub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.localPositionCallback,
            qos_profile
        )

        self.imuSub = self.create_subscription(
            SensorCombined,
            '/fmu/out/sensor_combined',
            self.imuCallback,
            qos_profile
        )

        self.meQuaternionNED = [0, 0, 0, 1]
        self.meRPYRadNED = np.zeros(3)
        self.meRPYRadENU = np.zeros(3)
        self.attitudeSub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitudeCallback,
            qos_profile
        )

        self.status = VehicleStatus(arming_state=100)
        self.statusSub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            lambda msg: setattr(self, 'status', msg),
            qos_profile
        )

        self.velMax = np.array([20.0, 20.0, 10.0])
        self.accMax = np.array([0.5, 0.5, 0.5])
        self.yawRateRadMax = 1

        # self.tStep = 0.02
        # self.kp = 1.0
        # self.ki = 0.01
        # self.vd = 10
        # self.intVError = 0.0
        # self.yawNED = 0

    def localPositionCallback(self, msg):
        self.localPosition = msg
        
        self.mePositionNED = np.array([msg.x, msg.y, msg.z])
        self.meVelocityNED = np.array([msg.vx, msg.vy, msg.vz])
        self.meAccelerationNED = np.array([msg.ax, msg.ay, msg.az])

        self.mePositionENU = ned2enu(self.mePositionNED)
        self.meVelocityENU = ned2enu(self.meVelocityNED)
        self.meAccelerationENU = ned2enu(self.meAccelerationNED)

        self.meSpeed = np.linalg.norm(self.meVelocityNED)

    def imuCallback(self, msg):
        self.imu = msg
        self.meAccelerationFRD = msg.accelerometer_m_s2
        self.meAccelerationimuNED = frd2nedRotationMatrix(self.meRPYRadNED[0], self.meRPYRadNED[1], self.meRPYRadNED[2]) @ (msg.accelerometer_m_s2 + np.array([0, 0, GRAVITY]))

    def attitudeCallback(self, msg):
        self.meQuaternionNED = msg.q
        self.meRPYRadNED = quaternion2euler(self.meQuaternionNED)
        self.meRPYRadENU = np.array([self.meRPYRadNED[0], -self.meRPYRadNED[1], -self.meRPYRadNED[2]])

    def intoOffboardMode(self):
        self.publishCommand(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            param1=1.0, param2=6.0
        )

    def setPositionControlMode(self):
        self.offboardControlMode.position = True

    def setVelocityControlMode(self):
        self.offboardControlMode.position = False
        self.offboardControlMode.velocity = True

    def setAccelerationControlMode(self):
        self.offboardControlMode.position = False
        self.offboardControlMode.velocity = False
        self.offboardControlMode.acceleration = True

    def setAttitudeControlMode(self):
        self.offboardControlMode.position = False
        self.offboardControlMode.velocity = False
        self.offboardControlMode.acceleration = False
        self.offboardControlMode.attitude = True
        self.offboardControlMode.body_rate = False

    def arm(self):
        self.publishCommand(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def isArmed(self):
        return self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED

    def disarm(self):
        self.publishCommand(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)

    def land(self):
        self.publishCommand(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    def printControl(self):
        print('-' * 10 + 'Control' + '-' * 10)
        print('Control Mode: ', end='')
        if self.offboardControlMode.attitude:
            print(
                RED + 
                'att: ' + rpyString(quaternion2euler(self.attitudeSetpoint.q_d)) + 
                ' thrust: ', pointString(self.attitudeSetpoint.thrust_body)
            )
        else:
            if self.offboardControlMode.position:
                print(BLUE + 'pos: ' + pointString(self.setpoint.position))
            elif self.offboardControlMode.velocity:
                print(GREEN + 'vel: ' + pointString(self.setpoint.velocity))
            elif self.offboardControlMode.acceleration:
                print(YELLOW + 'acc: ' + pointString(self.setpoint.acceleration))
        print(RESET)

    def printMe(self):
        print('-' * 10 + 'Me' + '-' * 10)
        print('Position NED: ' + pointString(self.mePositionNED))
        print('Velocity NED: ' + pointString(self.meVelocityNED) + f' speed: {self.meSpeed:.2f}')
        print('Acceleration NED: ' + pointString(self.meAccelerationNED))
        print('Acceleration FRD: ' + pointString(self.meAccelerationFRD))
        print('Euler NED: ' + rpyString(self.meRPYRadNED))

    def sendHeartbeat(self):
        self.offboardControlMode.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboardControlModePub.publish(self.offboardControlMode)

        if self.offboardControlMode.attitude:
            self.attitudeSetpoint.yaw_sp_move_rate = float('nan')
            self.attitudeSetpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.attitudeSetpointPub.publish(self.attitudeSetpoint)
        else:
            self.setpoint.velocity = np.clip(self.setpoint.velocity, -self.velMax, self.velMax).tolist()
            self.setpoint.acceleration = np.clip(self.setpoint.acceleration, -self.accMax, self.accMax).tolist()

            if not self.offboardControlMode.position:
                self.setpoint.position = [float('nan') for _ in range(3)]
            
            if not self.offboardControlMode.velocity:
                self.setpoint.velocity = [float('nan') for _ in range(3)]

            if not self.offboardControlMode.acceleration:
                self.setpoint.acceleration = [float('nan') for _ in range(3)]
            
            self.setpoint.yawspeed = np.clip(self.setpoint.yawspeed, -self.yawRateRadMax, self.yawRateRadMax)
            self.setpoint.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.setpointPub.publish(self.setpoint)

        self.printControl()

    def positionNEDControl(self, posNED, yawNED):
        self.offboardControlMode = OffboardControlMode(position=True)
        self.setpoint.position = posNED.tolist()
        self.setpoint.yaw = float(yawNED)

    def positionENUControl(self, posENU, yawENU):
        self.positionNEDControl(enu2ned(posENU), yawRadENU2NED(yawENU))

    def velocityNEDControl(self, velNED):
        self.offboardControlMode = OffboardControlMode(velocity=True)
        self.setpoint.velocity = velNED.tolist()
        self.setpoint.yaw = float(self.yawNED)

    def velocityENUControl(self, velENU):
        self.velocityNEDControl(enu2ned(velENU))

    def accelerationNEDControl(self, accNED):
        self.offboardControlMode = OffboardControlMode(acceleration=True)
        self.setpoint.acceleration = accNED.tolist()
        self.setpoint.yaw = float(self.yawNED)

    def accelerationENUControl(self, accENU):
        self.accelerationNEDControl(enu2ned(accENU))

    def acc2attENUControl(self, thrust, cmdRPYRadENU):
        self.offboardControlMode = OffboardControlMode(attitude=True)
        controlThrust = thrust
        controlQuaternion = euler2quaternion(rpyENU2NED(cmdRPYRadENU))
        self.attitudeSetpoint.q_d = controlQuaternion
        self.attitudeSetpoint.thrust_body = [0, 0, controlThrust]

    def nearPositionENU(self, pointENU, tol=1):
        return self.distanceToPointENU(pointENU) <= tol

    def nearSpeed(self, speed, tol=0.2):
        return abs(self.meSpeed - speed) <= tol

    def distanceToPointENU(self, pointENU):
        return np.linalg.norm(self.mePositionENU - pointENU)

    def aboveHeight(self, z):
        return self.localPosition.z < -z

    def underHeight(self, z):
        return self.localPosition.z > -z

    def publishCommand(self, command, **params):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.cmdPub.publish(msg)

    def getPositionNED(self):
        return self.mePositionNED

    def getPositionENU(self):
        return self.mePositionENU

    def getVelocityNED(self):
        return self.meVelocityNED

    def getVelocityENU(self):
        return self.meVelocityENU
    
    def getAccelerationNED(self):
        return self.meAccelerationNED
        # return self.meAccelerationimuNED
    
    def getQuaternionNED(self):
        return self.meQuaternionNED

    def getAttitude(self):
        return self.meRPYRadNED

    # def getUv(self):
    #     theta = math.atan(self.meVelocityNED[2]/ np.sqrt(self.meVelocityNED[0]**2 + self.meVelocityNED[1]**2))
    #     phi = math.atan(self.meVelocityNED[1] / self.meVelocityNED[0])
    #     vError = self.vd - np.linalg.norm(self.meVelocityNED)
    #     self.intVcError = (vError) * self.tStep + self.intVError
    #     u = self.kp * (vError) + self.ki * self.intVcError
    #     return np.array([u * math.cos(theta) * math.cos(phi), u * math.cos(theta) * math.sin(phi), u * math.sin(theta)])

    # def acceleration2attitude(self, uENU):
    #     self.yawNED = 0.0
    #     rollMax = pitchMax = math.radians(30.0)
    #     self.guidanceCommandNED = enu2ned(uENU)
    #     print(f"guidanceCommandNED = {pointString(self.guidanceCommandNED)}")
    #     liftAcceleration = -(self.guidanceCommandNED - np.array([0, 0, GRAVITY]))
    #     r1 = math.cos(self.yawNED) * liftAcceleration[0] + math.sin(self.yawNED) * liftAcceleration[1]
    #     r2 = math.sin(self.yawNED) * liftAcceleration[0] - math.cos(self.yawNED) * liftAcceleration[1]
    #     pitch = math.atan2(r1, liftAcceleration[2])
    #     r3 = math.sin(pitch) * r1 + math.cos(pitch) * liftAcceleration[2]
    #     roll = math.atan2(r2, r3)

    #     controlYaw = self.yawNED
    #     controlPitch = max(-pitchMax, min(pitch, pitchMax))
    #     controlRoll = max(-rollMax, min(roll, rollMax))
    #     controlEuler = np.array([controlRoll, controlPitch, controlYaw])
    #     print('Expected Euler NED: ' + rpyString(controlEuler))
    #     controlThrust = -0.708 * abs(liftAcceleration[2] / math.cos(controlRoll) / math.cos(controlPitch) / GRAVITY)
    #     controlThrust = max(-1, min(controlThrust, 1))

    #     controlQuaternion = euler2quaternion(controlEuler)
    #     return controlThrust, controlQuaternion



def main(args=None) -> None:
    rclpy.init(args=args)
    iris = Iris()
    rclpy.spin(iris)
    iris.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
