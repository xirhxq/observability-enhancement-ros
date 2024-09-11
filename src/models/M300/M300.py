#!/usr/bin/env python3

from models.M300.libFlightControl import *
from Utils import *
from QuaternionBuffer import QuaternionBuffer
from sensor_msgs.msg import NavSatFix

class M300:
    def __init__(self, uav_name):
        self.name = uav_name

        # Publishers
        self.ctrl_cmd_pub = rospy.Publisher(uav_name + "/dji_osdk_ros/flight_control_setpoint_generic", Joy, queue_size=10)
        self.gimbal_angle_cmd_pub = rospy.Publisher(uav_name + "/gimbal/gimbal_angle_cmd", Vector3, queue_size=10)
        self.gimbal_speed_cmd_pub = rospy.Publisher(uav_name + "/gimbal/gimbal_speed_cmd", Vector3, queue_size=10)

        # Service Clients
        self.sdk_ctrl_authority_service = rospy.ServiceProxy(uav_name + "/dji_osdk_ros/sdk_control_authority", SDKControlAuthority)
        self.drone_task_service = rospy.ServiceProxy(uav_name + "/dji_osdk_ros/drone_task_control", DroneTaskControl)
        self.set_local_pos_reference = rospy.ServiceProxy(uav_name + "/dji_osdk_ros/set_local_pos_ref", SetLocalPosRef)

        # State variables
        self.current_atti = QuaternionStamped()
        self.current_gimbal_rpy_deg = Point()
        self.current_pos_raw = Point()
        self.current_height = Float32()
        self.current_vo_pos = VOPosition()
        self.current_local_pos = Point()
        self.current_rtk_pos = NavSatFix()
        
        # Buffer
        self.qBuffer = QuaternionBuffer()

        self.flight_status = 255
        self.display_mode = 255

        self.meRTKOrigin = None
        self.meRTKLla = None
        self.meGPSLla = None
        self.mePositionENU = np.zeros(3)
        self.mePositionNED = np.zeros(3)
        self.meVelocityENU = np.zeros(3)
        self.meVelocityNED = np.zeros(3)
        self.meSpeed = 0
        self.meAccelerationENU = np.zeros(3)
        self.meAccelerationNED = np.zeros(3)
        self.meAccelerationImuFRD = np.zeros(3)
        self.meAccelerationFRD = np.zeros(3)
        self.meAccelerationFLU = np.zeros(3)
        self.meAccelerationENUFused = np.zeros(3)
        self.meRPYRadNED = np.zeros(3)
        self.meRPYRadENU = np.zeros(3)
        self.meRPYRadENUDelay = None
        self.meRPYRadNEDDelay = None
        self.hoverThrottle = rospy.get_param('hoverThrottle')
        self.rollSaturationRad = np.deg2rad(rospy.get_param('rollSaturationDeg'))
        self.pitchSaturationRad = np.deg2rad(rospy.get_param('pitchSaturationDeg'))
        self.useRTK = rospy.get_param('useRTK') == True
        self.meGimbalRPYNEDRad = np.zeros(3)

        self.elevationAngle = 0.0
        self.azimuthAngle = 0.0
        
        # Subscribers
        rospy.Subscriber(uav_name + "/dji_osdk_ros/rtk_position", NavSatFix, self.rtk2localPosition_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/gps_position", NavSatFix, self.gpsPosition_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/attitude", QuaternionStamped, self.attitude_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/gimbal_angle", Vector3Stamped, self.gimbal_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/height_above_takeoff", Float32, self.height_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/vo_position", VOPosition, self.vo_pos_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/flight_status", UInt8, self.flight_status_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/display_mode", UInt8, self.display_mode_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/local_position", PointStamped, self.local_position_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/velocity", Vector3Stamped, self.velocity_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/imu", Imu, self.imu_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/acceleration_ground_fused", Vector3Stamped, self.accelerationENU_callback)
        rospy.Subscriber("/spirecv/aruco_detection", TargetsInFrame, self.lookAngle_callback)

    def printMe(self):
        print('-' * 10 + 'Me' + '-' * 10)
        print('Position ENU: ' + pointString(self.mePositionENU))
        print('Velocity NED: ' + pointString(self.meVelocityNED) + f' speed: {self.meSpeed:.2f}')
        print('Acceleration imuFRD: ' + pointString(self.meAccelerationImuFRD))
        print('Acceleration FRD: ' + pointString(self.meAccelerationFRD))
        print('Acceleration NED: ' + pointString(self.meAccelerationNED))
        print('Euler ENU: ' + rpyString(self.meRPYRadENU))
        print('Euler NED: ' + rpyString(self.meRPYRadNED))
        print(f'Hoverthrottle: {self.hoverThrottle:.2f}')
        print(f'gimbalAngleRPYdeg = {vector3String(self.current_gimbal_rpy_deg)}')

    def rtk2localPosition_callback(self, msg):
        self.current_rtk_pos = msg
        self.meRTKLla = [msg.longitude, msg.latitude, msg.altitude]
        if self.meRTKOrigin is not None and self.useRTK:
            self.mePositionENU = localOffsetFromGpsOffset(msg, self.meRTKOrigin)
            self.mePositionNED = enu2ned(self.mePositionENU)

    def gpsPosition_callback(self, msg):
        self.current_gps_pos = msg
        self.meGPSLla = [msg.longitude, msg.latitude, msg.altitude]

    def attitude_callback(self, msg):
        self.current_atti = msg
        q = [msg.quaternion.w, msg.quaternion.x, msg.quaternion.y, msg.quaternion.z]
        self.meRPYRadENU = quaternion2euler(q)
        self.meRPYRadNED = rpyENU2NED(self.meRPYRadENU)
        if rospy.get_param('useCamera'):
            self.qBuffer.addMessage(q)
            qDelay = self.qBuffer.getMessage()
            if qDelay != None:
                print('With camera, using delay IMU information with qBuffer......')
                self.meRPYRadENUDelay = quaternion2euler(qDelay)
                self.meRPYRadNEDDelay = rpyENU2NED(self.meRPYRadENUDelay)

    def gimbal_callback(self, msg):
        self.current_gimbal_rpy_deg.x = msg.vector.y
        self.current_gimbal_rpy_deg.y = msg.vector.x
        self.current_gimbal_rpy_deg.z = msg.vector.z
        self.meGimbalRPYNEDRad = np.radians(np.array([self.current_gimbal_rpy_deg.x, self.current_gimbal_rpy_deg.y, self.current_gimbal_rpy_deg.z]))
        self.meGimbalRPYENURad = rpyNED2ENU(self.meGimbalRPYNEDRad)

    def lookAngle_callback(self, msg: TargetsInFrame):
        if len(msg.targets) > 0:
            self.azimuthAngle = msg.targets[0].los_ax
            self.elevationAngle = msg.targets[0].los_ay
            
    def height_callback(self, msg):
        self.current_height = msg

    def vo_pos_callback(self, msg):
        self.current_vo_pos = msg

    def flight_status_callback(self, msg):
        self.flight_status = msg.data

    def display_mode_callback(self, msg):
        self.display_mode = msg.data

    def local_position_callback(self, msg: PointStamped):
        self.current_local_pos = msg.point
        if not self.useRTK:
            self.mePositionENU = np.array([msg.point.x, msg.point.y, msg.point.z])
            self.mePositionNED = enu2ned(self.mePositionENU)

    def velocity_callback(self, msg: Vector3Stamped):
        self.meVelocityENU = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        self.meVelocityNED = enu2ned(self.meVelocityENU)
        self.meSpeed = np.linalg.norm(self.meVelocityNED)

    def imu_callback(self, msg: Imu):
        self.meAccelerationImuFRD = np.array([msg.linear_acceleration.x, -msg.linear_acceleration.y, -msg.linear_acceleration.z])
        self.meAccelerationFRD = frd2nedRotationMatrix(self.meRPYRadNED) @ self.meAccelerationImuFRD
        self.meAccelerationNED = self.meAccelerationFRD + np.array([0, 0, GRAVITY])
        self.meAccelerationENU = ned2enu(self.meAccelerationNED)

    def accelerationENU_callback(self, msg: Vector3Stamped):
        self.meAccelerationENUFused = np.array([msg.vector.x, msg.vector.y, msg.vector.z])

    def takeoff_land(self, task):
        # rospy.wait_for_service(self.drone_task_service.resolved_name)
        try:
            droneTaskControl = DroneTaskControlRequest()
            droneTaskControl.task = task

            response = self.drone_task_service(droneTaskControl)

            if not response.result:
                rospy.logerr("takeoff_land fail")
                return False

            rospy.loginfo("Takeoff/Land Success!")
            return True
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False

    def obtain_control(self):
        # rospy.wait_for_service(self.sdk_ctrl_authority_service.resolved_name)
        try:
            authority = SDKControlAuthorityRequest()
            authority.control_enable = 1

            response = self.sdk_ctrl_authority_service(authority)

            if not response.result:
                rospy.logerr("obtain control failed!")
                return False

            rospy.loginfo("obtain control successful!")
            return True
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False

    def set_local_position(self):
        if self.useRTK:
            if self.meRTKLla is not None:
                self.meRTKOrigin = self.current_rtk_pos
                rospy.loginfo("set rtk origin successful!")
            else:
                rospy.logerr("set rtk origin failed!")
                return False
        else:
            self.meRTKOrigin = 'unused'
        # rospy.wait_for_service(self.set_local_pos_reference.resolved_name)
        try:
            response = self.set_local_pos_reference()

            if not response.result:
                rospy.logerr("set local position failed!")
                return False

            rospy.loginfo("set local position successful!")
            return True
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)
            return False

    def monitored_takeoff(self):
        start_time = rospy.Time.now()

        if not self.takeoff_land(DroneTaskControlRequest.TASK_TAKEOFF):
            return False

        rospy.sleep(0.01)

        while self.flight_status != DJISDK.FlightStatus.STATUS_ON_GROUND and \
              self.display_mode != DJISDK.DisplayMode.MODE_ENGINE_START and \
              rospy.Time.now() - start_time < rospy.Duration(5):
            rospy.sleep(0.01)

        if rospy.Time.now() - start_time > rospy.Duration(5):
            rospy.logerr("Takeoff failed. Motors are not spinning.")
            return False
        else:
            rospy.loginfo("Motor Spinning ...")

        start_time = rospy.Time.now()
        while self.flight_status != DJISDK.FlightStatus.STATUS_IN_AIR and \
              (self.display_mode != DJISDK.DisplayMode.MODE_ASSISTED_TAKEOFF or \
               self.display_mode != DJISDK.DisplayMode.MODE_AUTO_TAKEOFF) and \
              rospy.Time.now() - start_time < rospy.Duration(20):
            rospy.sleep(0.01)

        if rospy.Time.now() - start_time > rospy.Duration(20):
            rospy.logerr("Takeoff failed. Aircraft is still on the ground, but the motors are spinning.")
            return False
        else:
            rospy.loginfo("Ascending...")

        start_time = rospy.Time.now()
        while self.display_mode == DJISDK.DisplayMode.MODE_ASSISTED_TAKEOFF or \
              self.display_mode == DJISDK.DisplayMode.MODE_AUTO_TAKEOFF and \
              rospy.Time.now() - start_time < rospy.Duration(20):
            rospy.sleep(0.01)

        if self.display_mode != DJISDK.DisplayMode.MODE_P_GPS or \
           self.display_mode != DJISDK.DisplayMode.MODE_ATTITUDE:
            rospy.loginfo("Successful takeoff!")
            return True
        else:
            rospy.logerr("Takeoff finished, but the aircraft is in an unexpected mode. Please connect DJI GO.")
            return False

    def hoverAtHeight(self, zENU=0.0):
        control_cmd = Joy()
        control_cmd.axes = [0, 0, zENU, 0, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_BODY]
        self.ctrl_cmd_pub.publish(control_cmd)

    def hoverWithYaw(self, yawRadENU):
        control_cmd = Joy()
        control_cmd.axes = [0, 0, 0, yawRadENU, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def bodyVelVelYawRate(self, velE, velN, velU, yawRateRad):
        control_cmd = Joy()
        control_cmd.axes = [velE, velN, velU, yawRateRad, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_BODY]
        self.ctrl_cmd_pub.publish(control_cmd)

    def groundVelVelYaw(self, velE, velN, velU, yawRadENU):
        control_cmd = Joy()
        control_cmd.axes = [velE, velN, velU, yawRadENU, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def velocityENUControl(self, velENU, yawRadENU):
        self.groundVelVelYaw(*velENU.tolist(), yawRadENU)

    def groundVelVelYawRate(self, velE, velN, velU, yawRateRad):
        control_cmd = Joy()
        control_cmd.axes = [velE, velN, velU, yawRateRad, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def groundPosPosYaw(self, posDiffE, posDiffN, posU, yawRadENU):
        control_cmd = Joy()
        control_cmd.axes = [posDiffE, posDiffN, posU, yawRadENU, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_POSITION | DJISDK.Control.HORIZONTAL_POSITION | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)
    
    def positionENUControl(self, posENU, yawRadENU):
        posXYDiff = posENU[:2] - self.mePositionENU[:2]
        self.groundPosPosYaw(*posXYDiff.tolist(), posENU[2], yawRadENU)

    def groundVelPosYaw(self, velE, velN, posU, yawRadENU):
        control_cmd = Joy()
        control_cmd.axes = [velE, velN, posU, yawRadENU, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_POSITION | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def groundPosPosYawRate(self, posDiffE, posDiffN, posU, yawRateRad):
        control_cmd = Joy()
        control_cmd.axes = [posDiffE, posDiffN, posU, yawRateRad, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_POSITION | DJISDK.Control.HORIZONTAL_POSITION | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def uav_velocity_yaw_rate_ctrl(self, pos_diff, yaw_diff):
        KP, YAW_KP = 0.2, 0.2
        vel = Vector3()
        vel.x = pos_diff.x * KP
        vel.y = pos_diff.y * KP
        vel.z = pos_diff.z * KP
        yaw_rate = yaw_diff * YAW_KP
        self.saturate_vel(vel, Vector3(0.1, 0.1, 0.2))
        rospy.loginfo(f"Velo cmd: {vel}")
        self.groundVelVelYawRate(vel.x, vel.y, vel.z, yaw_rate)

    def uav_control_to_point_facing_it(self, ctrl_cmd):
        yaw_diff = self.angle2d(self.current_pos_raw, ctrl_cmd) - self.meRPYRadENU[2]
        yaw_diff = self.rad_round(yaw_diff)
        if self.dis2d(ctrl_cmd, self.current_pos_raw) <= 1:
            yaw_diff = 0
        rospy.loginfo(f"Yaw diff: {yaw_diff}")
        self.uav_velocity_yaw_rate_ctrl(self.minus(ctrl_cmd, self.current_pos_raw), yaw_diff)

    def uav_control_to_point_with_yaw(self, ctrl_cmd, yaw):
        yaw_diff = yaw - self.meRPYRadENU[2]
        yaw_diff = self.rad_round(yaw_diff)
        if self.dis2d(ctrl_cmd, self.current_pos_raw) <= 1:
            yaw_diff = 0
        rospy.loginfo(f"Yaw diff: {yaw_diff}")
        self.uav_velocity_yaw_rate_ctrl(self.minus(ctrl_cmd, self.current_pos_raw), yaw_diff)

    def uav_control_body(self, ctrl_cmd, yaw_rate=0.0):
        self.bodyVelVelYawRate(ctrl_cmd.x, ctrl_cmd.y, ctrl_cmd.z, yaw_rate)

    def send_gimbal_angle_ctrl_cmd(self, roll, pitch, yaw):
        v = Vector3()
        v.x = roll
        v.y = pitch
        v.z = yaw
        self.gimbal_angle_cmd_pub.publish(v)

    def send_gimbal_speed_ctrl_cmd(self, roll, pitch, yaw):
        v = Vector3()
        v.x = roll
        v.y = pitch
        v.z = yaw
        self.gimbal_speed_cmd_pub.publish(v)

    def get_time_now(self):
        return rospy.Time.now().to_sec()

    def enough_time_after(self, t0, duration):
        return self.get_time_now() - t0 >= duration

    def nearPositionENU(self, pointENU, tol=1):
        return self.distanceToPointENU(pointENU) <= tol

    def nearSpeed(self, speed, tol=0.2):
        return abs(self.meSpeed - speed) <= tol

    def distanceToPointENU(self, pointENU):
        return np.linalg.norm(self.mePositionENU - pointENU)

    def aboveHeight(self, z):
        return self.mePositionNED[2] < -z

    def underHeight(self, z):
        return self.mePositionNED[2] > -z
    
    def setPositionControlMode(self):
        pass

    def setVelocityControlMode(self):
        pass

    def setAttitudeControlMode(self):
        pass

    def acc2attENUControl(self, thrust, rpyRadENU):
        control_cmd = Joy()
        rpyRadENU = self.saturateRPYRad(rpyRadENU)
        print(f"rpySaturateDeg = {np.rad2deg(rpyRadENU)}")
        control_cmd.axes = [rpyRadENU[0], rpyRadENU[1], thrust, rpyRadENU[2], DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_THRUST | DJISDK.Control.HORIZONTAL_ANGLE | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_BODY]
        self.ctrl_cmd_pub.publish(control_cmd)
    
    def saturateRPYRad(self, rpyRadENU):
        rpyRadENU[0] = max(-self.rollSaturationRad, min(self.rollSaturationRad, rpyRadENU[0]))
        rpyRadENU[1] = max(-self.pitchSaturationRad, min(self.pitchSaturationRad, rpyRadENU[1]))
        return rpyRadENU

    def sendHeartbeat(self):
        pass

    def land(self):
        self.takeoff_land(DroneTaskControlRequest.TASK_LAND)

    def getVelocityENU(self):
        return self.meVelocityENU
    
    def getPositionENU(self):
        return self.mePositionENU
    
    def getAccelerationNED(self):
        return self.meAccelerationNED
        

if __name__ == '__main__':
    fc = M300('suav')