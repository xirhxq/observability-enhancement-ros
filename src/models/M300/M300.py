#!/usr/bin/env python3

from models.M300.libFlightControl import *
from Utils import *

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

        # Subscribers
        rospy.Subscriber(uav_name + "/dji_osdk_ros/attitude", QuaternionStamped, self.attitude_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/gimbal_angle", Vector3Stamped, self.gimbal_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/height_above_takeoff", Float32, self.height_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/vo_position", VOPosition, self.vo_pos_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/flight_status", UInt8, self.flight_status_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/display_mode", UInt8, self.display_mode_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/local_position", PointStamped, self.local_position_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/velocity", Vector3Stamped, self.velocity_callback)
        rospy.Subscriber(uav_name + "/dji_osdk_ros/imu", Imu, self.imu_callback)

        # State variables
        self.current_atti = QuaternionStamped()
        self.current_euler_angle = Point()
        self.current_gimbal_angle = Point()
        self.current_pos_raw = Point()
        self.current_height = Float32()
        self.current_vo_pos = VOPosition()
        self.current_local_pos = Point()
        self.yaw_offset = 0.0
        self.position_offset = Point()
        self.EMERGENCY = False

        self.yawRadENU = math.pi / 2

        self.flight_status = 255
        self.display_mode = 255

        self.mePositionENU = np.zeros(3)
        self.mePositionNED = np.zeros(3)
        self.meVelocityENU = np.zeros(3)
        self.meVelocityNED = np.zeros(3)
        self.meSpeed = 0
        self.meAccelerationENU = np.zeros(3)
        self.meAccelerationNED = np.zeros(3)
        self.meAccelerationFRD = np.zeros(3)
        self.meAccelerationFLU = np.zeros(3)
        self.meRPYNED = np.zeros(3)
        self.meRPYENU = np.zeros(3)

    def printMe(self):
        print('-' * 10 + 'Me' + '-' * 10)
        print('Position NED: ' + pointString(self.mePositionNED))
        print('Velocity NED: ' + pointString(self.meVelocityNED) + f' speed: {self.meSpeed:.2f}')
        print('Acceleration NED: ' + pointString(self.meAccelerationNED))
        print('Acceleration FRD: ' + pointString(self.meAccelerationFRD))
        print('Euler NED: ' + rpyString(self.meRPYNED))

    def to_euler_angle(self, quat):
        q = [quat.x, quat.y, quat.z, quat.w]
        euler = tf.transformations.euler_from_quaternion(q)
        self.current_euler_angle.x, self.current_euler_angle.y, self.current_euler_angle.z = euler
        return self.current_euler_angle

    def attitude_callback(self, msg):
        self.current_atti = msg
        self.current_euler_angle = self.to_euler_angle(msg.quaternion)
        self.meRPYENU = np.array([self.current_euler_angle.x, self.current_euler_angle.y, self.current_euler_angle.z])
        self.meRPYNED = np.array([self.meRPYENU[1], self.meRPYENU[0], rad_round(math.pi / 2 - self.meRPYENU[2])])

    def gimbal_callback(self, msg):
        self.current_gimbal_angle.x = msg.vector.y
        self.current_gimbal_angle.y = msg.vector.x
        self.current_gimbal_angle.z = msg.vector.z

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
        self.mePositionENU = np.array([msg.point.x, msg.point.y, msg.point.z])
        self.mePositionNED = enu2ned(self.mePositionENU)

    def velocity_callback(self, msg: Vector3Stamped):
        self.meVelocityENU = np.array([msg.vector.x, msg.vector.y, msg.vector.z])
        self.meVelocityNED = enu2ned(self.meVelocityENU)
        self.meSpeed = np.linalg.norm(self.meVelocityNED)

    def imu_callback(self, msg: Imu):
        self.meAccelerationImuFRD = np.array([msg.linear_acceleration.x, -msg.linear_acceleration.y, -msg.linear_acceleration.z])
        self.meAccelerationNED = frd2nedRotationMatrix(self.meRPYNED[0], self.meRPYNED[1], self.meRPYNED[2]) @ (self.meAccelerationImuFRD + np.array([0, 0, GRAVITY]))
        self.meAccelerationENU = ned2enu(self.meAccelerationNED)


    def takeoff_land(self, task):
        rospy.wait_for_service(self.drone_task_service.resolved_name)
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
        rospy.wait_for_service(self.sdk_ctrl_authority_service.resolved_name)
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
        rospy.wait_for_service(self.set_local_pos_reference.resolved_name)
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

    def m210_hold_ctrl(self, z=0.0):
        control_cmd = Joy()
        control_cmd.axes = [0, 0, z, 0, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_BODY]
        self.ctrl_cmd_pub.publish(control_cmd)

    def m210_adjust_yaw(self, yaw):
        control_cmd = Joy()
        control_cmd.axes = [0, 0, 0, yaw, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def m210_body_vel_yaw_rate_ctrl(self, x, y, z, yaw_rate):
        control_cmd = Joy()
        control_cmd.axes = [x, y, z, yaw_rate, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_BODY]
        self.ctrl_cmd_pub.publish(control_cmd)

    def m210_velocity_yaw_ctrl(self, vx, vy, vz, yaw):
        control_cmd = Joy()
        control_cmd.axes = [vx, vy, vz, yaw, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def velocityENUControl(self, velENU):
        self.m210_velocity_yaw_ctrl(*velENU.tolist(), self.yawRadENU)

    def m210_velocity_yaw_rate_ctrl(self, vx, vy, vz, yaw_rate):
        control_cmd = Joy()
        control_cmd.axes = [vx, vy, vz, yaw_rate, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def m210_position_yaw_ctrl(self, x, y, z, yaw):
        control_cmd = Joy()
        control_cmd.axes = [x, y, z, yaw, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_POSITION | DJISDK.Control.HORIZONTAL_POSITION | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)
    
    def positionENUControl(self, posENU):
        self.m210_position_yaw_ctrl(*posENU.tolist(), self.yawRadENU)

    def m210_velocity_position_yaw_ctrl(self, x, y, z, yaw):
        control_cmd = Joy()
        control_cmd.axes = [x, y, z, yaw, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_POSITION | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def m210_position_yaw_rate_ctrl(self, x, y, z, yaw):
        control_cmd = Joy()
        control_cmd.axes = [x, y, z, yaw, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_POSITION | DJISDK.Control.HORIZONTAL_POSITION | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def uav_velocity_yaw_rate_ctrl(self, pos_diff, yaw_diff):
        vel = Vector3()
        vel.x = pos_diff.x * KP
        vel.y = pos_diff.y * KP
        vel.z = pos_diff.z * KP
        yaw_rate = yaw_diff * YAW_KP
        self.saturate_vel(vel, Vector3(0.1, 0.1, 0.2))
        rospy.loginfo(f"Velo cmd: {vel}")
        self.m210_velocity_yaw_rate_ctrl(vel.x, vel.y, vel.z, yaw_rate)

    def uav_control_to_point_facing_it(self, ctrl_cmd):
        yaw_diff = self.angle2d(self.current_pos_raw, ctrl_cmd) - self.current_euler_angle.z
        yaw_diff = self.rad_round(yaw_diff)
        if self.dis2d(ctrl_cmd, self.current_pos_raw) <= 1:
            yaw_diff = 0
        rospy.loginfo(f"Yaw diff: {yaw_diff}")
        self.uav_velocity_yaw_rate_ctrl(self.minus(ctrl_cmd, self.current_pos_raw), yaw_diff)

    def uav_control_to_point_with_yaw(self, ctrl_cmd, yaw):
        yaw_diff = yaw - self.current_euler_angle.z
        yaw_diff = self.rad_round(yaw_diff)
        if self.dis2d(ctrl_cmd, self.current_pos_raw) <= 1:
            yaw_diff = 0
        rospy.loginfo(f"Yaw diff: {yaw_diff}")
        self.uav_velocity_yaw_rate_ctrl(self.minus(ctrl_cmd, self.current_pos_raw), yaw_diff)

    def uav_control_body(self, ctrl_cmd, yaw_rate=0.0):
        self.m210_body_vel_yaw_rate_ctrl(ctrl_cmd.x, ctrl_cmd.y, ctrl_cmd.z, yaw_rate)

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

    def acc2attENUControl(self, accENU):
        controlThrust, controlEulerENU = self.acceleration2attitude(accENU)
        control_cmd = Joy()
        control_cmd.axes = [controlEulerENU[0], controlEulerENU[1], controlThrust, controlEulerENU[2], DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_THRUST | DJISDK.Control.HORIZONTAL_ANGLE | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_BODY]
        self.ctrl_cmd_pub.publish(control_cmd)

    def acceleration2attitude(self, uENU):
        self.yawNED = 0.0
        rollMax = pitchMax = math.radians(30.0)
        self.guidanceCommandNED = enu2ned(uENU)
        print(f"guidanceCommandNED = {pointString(self.guidanceCommandNED)}")
        liftAcceleration = -(self.guidanceCommandNED - np.array([0, 0, GRAVITY]))
        r1 = math.cos(self.yawNED) * liftAcceleration[0] + math.sin(self.yawNED) * liftAcceleration[1]
        r2 = math.sin(self.yawNED) * liftAcceleration[0] - math.cos(self.yawNED) * liftAcceleration[1]
        pitch = math.atan2(r1, liftAcceleration[2])
        r3 = math.sin(pitch) * r1 + math.cos(pitch) * liftAcceleration[2]
        roll = math.atan2(r2, r3)
        controlYaw = self.yawNED
        controlPitch = max(-pitchMax, min(pitch, pitchMax))
        controlRoll = max(-rollMax, min(roll, rollMax))
        controlEulerNED = np.array([controlRoll, controlPitch, controlYaw])
        print('Expected Euler NED: ' + rpyString(controlEulerNED))
        
        controlEulerENU = np.array([controlPitch, controlRoll, rad_round(math.pi/2 - controlYaw)])
        controlThrust = 31.0 * abs(liftAcceleration[2] / math.cos(controlRoll) / math.cos(controlPitch) / GRAVITY)
        return controlThrust, controlEulerENU
    
    def sendHeartbeat(self):
        pass

    def land(self):
        self.takeoff_land(DroneTaskControlRequest.TASK_LAND)

    def getVelocityENU(self):
        return self.meVelocityENU
    
    def getPositionENU(self):
        return self.mePositionENU
    
    def getAttitude(self):
        return self.meRPYNED
    
    def getAccelerationNED(self):
        return self.meAccelerationNED
        

if __name__ == '__main__':
    fc = M300('suav')