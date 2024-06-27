#!/usr/bin/env python3

from libFlightControl import *

class FLIGHT_CONTROL:
    def __init__(self, uav_name):
        self.fc_nh = rospy.init_node('flight_control', anonymous=True)
        self.uav_name = uav_name

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

        self.flight_status = 255
        self.display_mode = 255

        rospy.spin()

    def to_euler_angle(self, quat):
        q = [quat.x, quat.y, quat.z, quat.w]
        euler = tf.transformations.euler_from_quaternion(q)
        self.current_euler_angle.x, self.current_euler_angle.y, self.current_euler_angle.z = euler
        return self.current_euler_angle

    def attitude_callback(self, msg):
        self.current_atti = msg
        self.current_euler_angle = self.to_euler_angle(msg.quaternion)

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

    def takeoff_land(self, task):
        try:
            droneTaskControl = DroneTaskControl()
            droneTaskControl.request.task = task
            response = self.drone_task_service(droneTaskControl)
            if response.result:
                rospy.loginfo("Takeoff/Land Success!")
                return True
            else:
                rospy.logerr("Takeoff/Land failed")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def obtain_control(self):
        try:
            authority = SDKControlAuthority()
            authority.request.control_enable = 1
            response = self.sdk_ctrl_authority_service(authority)
            if response.result:
                rospy.loginfo("Obtain control successful!")
                return True
            else:
                rospy.logerr("Obtain control failed!")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def monitored_takeoff(self):
        start_time = rospy.Time.now()

        if not self.takeoff_land(DroneTaskControl.Request.TASK_TAKEOFF):
            return False

        rospy.sleep(0.01)
        rospy.spin_once()

        while self.flight_status != DJISDK.FlightStatus.STATUS_ON_GROUND and \
              self.display_mode != DJISDK.DisplayMode.MODE_ENGINE_START and \
              rospy.Time.now() - start_time < rospy.Duration(5):
            rospy.sleep(0.01)
            rospy.spin_once()

        if rospy.Time.now() - start_time > rospy.Duration(5):
            rospy.logerr("Takeoff failed. Motors are not spinning.")
            return False
        else:
            rospy.loginfo("Motor Spinning ...")
            rospy.spin_once()

        start_time = rospy.Time.now()
        while self.flight_status != DJISDK.FlightStatus.STATUS_IN_AIR and \
              (self.display_mode != DJISDK.DisplayMode.MODE_ASSISTED_TAKEOFF or \
               self.display_mode != DJISDK.DisplayMode.MODE_AUTO_TAKEOFF) and \
              rospy.Time.now() - start_time < rospy.Duration(20):
            rospy.sleep(0.01)
            rospy.spin_once()

        if rospy.Time.now() - start_time > rospy.Duration(20):
            rospy.logerr("Takeoff failed. Aircraft is still on the ground, but the motors are spinning.")
            return False
        else:
            rospy.loginfo("Ascending...")
            rospy.spin_once()

        start_time = rospy.Time.now()
        while self.display_mode == DJISDK.DisplayMode.MODE_ASSISTED_TAKEOFF or \
              self.display_mode == DJISDK.DisplayMode.MODE_AUTO_TAKEOFF and \
              rospy.Time.now() - start_time < rospy.Duration(20):
            rospy.sleep(0.01)
            rospy.spin_once()

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

    def m210_velocity_yaw_rate_ctrl(self, vx, vy, vz, yaw_rate):
        control_cmd = Joy()
        control_cmd.axes = [vx, vy, vz, yaw_rate, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_VELOCITY | DJISDK.Control.HORIZONTAL_VELOCITY | DJISDK.Control.YAW_RATE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

    def m210_position_yaw_ctrl(self, x, y, z, yaw):
        control_cmd = Joy()
        control_cmd.axes = [x, y, z, yaw, DJISDK.Control.STABLE_ENABLE | DJISDK.Control.VERTICAL_POSITION | DJISDK.Control.HORIZONTAL_POSITION | DJISDK.Control.YAW_ANGLE | DJISDK.Control.HORIZONTAL_GROUND]
        self.ctrl_cmd_pub.publish(control_cmd)

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

    def set_local_position(self):
        try:
            response = self.set_local_pos_reference()
            return response.result
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False
        

if __name__ == '__main__':
    fc = FLIGHT_CONTROL('suav')