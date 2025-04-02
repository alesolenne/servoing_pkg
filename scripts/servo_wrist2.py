#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from std_srvs.srv import SetBool
from servoing_pkg.kinematics_wrist import *
from servoing_pkg.tools import *
from pytransform3d.rotations import quaternion_from_matrix, axis_angle_from_quaternion
from servoing_pkg.srv import set_stiffness, set_stiffnessRequest

# Constants
JOINT_NAMES_ARM = [
    'robot_arm_joint1', 'robot_arm_joint2', 'robot_arm_joint3',
    'robot_arm_joint4', 'robot_arm_joint5', 'robot_arm_joint6', 'robot_arm_joint7'
]
JOINT_NAMES_WRIST = [
    'qbmove2_motor_1_joint', 'qbmove2_motor_2_joint', 'qbmove2_shaft_joint',
    'qbmove2_deflection_virtual_joint', 'qbmove2_stiffness_preset_virtual_joint'
]
STIFFNESS_MAX = 1.0
ERROR_TRANSLATION_THRESHOLD = 0.005
ERROR_ORIENTATION_THRESHOLD = 0.01
CONTROL_FREQUENCY = 10000.0

def check_q_goal(q_goal):
    """Check if the robot and wrist have reached the desired position."""
    while True:
        joint_states = rospy.wait_for_message('/robot/joint_states', JointState)
        wrist_states = rospy.wait_for_message('/robot_arm/gripper/qbmove2/joint_states', JointState)

        if joint_states.name == JOINT_NAMES_ARM and wrist_states.name == JOINT_NAMES_WRIST:
            q_arm_actual = np.array(joint_states.position).reshape((7, 1))
            q_wrist_actual = np.array(wrist_states.position[2]).reshape((1, 1))
            q_actual = np.vstack((q_arm_actual, q_wrist_actual))
            e_q = q_goal - q_actual

            if np.linalg.norm(e_q[0:8, :], 2) < 0.01:
                return q_actual
        rospy.sleep(0.01)

def feedback(trans_0B, rot_0B, trans_0V, rot_0V, q):
    """PBVS Visual servoing control law.

    Parameters
    ----------
    trans_ij : [3x1] np.ndarray
               translation vector of j with respect to i
    
    rot_ij : [3x3] np.ndarray
             rotation matrix from i to j
            
    Returns
    -------
    dq : [7x1] np.ndarray
         PBVS control law joints velocity

    e: [6x1] np.ndarray
       Position and orientation error vector
    """
    
    (J, T_0E) = get_jacobian(q)
    T_0B = hom_matrix(trans_0B, rot_0B)
    T_0V = hom_matrix(trans_0V, rot_0V)

    t_0V, t_0B = T_0V[:3, 3], T_0B[:3, 3]
    R_0V, R_0B = T_0V[:3, :3], T_0B[:3, :3]

    R_e = np.matmul(R_0B, np.transpose(R_0V))
    q_e = quaternion_from_matrix(R_e)
    q_e_axisangle = axis_angle_from_quaternion(q_e)
    r, theta = q_e_axisangle[:3], q_e_axisangle[3]

    e_t = t_0B - t_0V
    e_o = np.sin(theta) * r
    e = np.vstack((e_t, e_o)).reshape((6, 1))

    I, Z = np.identity(3), np.zeros((3, 3))
    L_e = -0.5 * sum(np.matmul(skew_matrix(R_0B[:, i]), skew_matrix(R_0V[:, i])) for i in range(3))
    L = np.block([[I, Z], [Z, np.linalg.pinv(L_e)]])

    J_pinv = np.linalg.pinv(J)
    K_p, K_o = 1000 * np.identity(3), 1000 * np.identity(3)
    K = np.block([[K_p, Z], [Z, K_o]])

    q_dot = np.linalg.multi_dot([J_pinv, L, K, e])
    return q_dot, e

def initialize_state():
    """Initialize the robot and wrist state."""
    while True:
        joint_states = rospy.wait_for_message('/robot/joint_states', JointState)
        wrist_states = rospy.wait_for_message('/robot_arm/gripper/qbmove2/joint_states', JointState)

        if joint_states.name == JOINT_NAMES_ARM and wrist_states.name == JOINT_NAMES_WRIST:
            q_arm = np.array(joint_states.position).reshape((7, 1))
            q_wrist = np.array(wrist_states.position[2]).reshape((1, 1))
            return np.vstack((q_arm, q_wrist))
        rospy.sleep(0.01)

def call_service(service_name, service_type, request):
    """Call a ROS service and handle exceptions."""
    rospy.wait_for_service(service_name)
    try:
        client = rospy.ServiceProxy(service_name, service_type)
        response = client(request)
        return response
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call to {service_name} failed: {e}")
        return None

def servoing():
    """Perform visual servoing."""
    pub_arm = rospy.Publisher(
        "/robot/arm/position_joint_trajectory_controller/command",
        JointTrajectory, queue_size=10
    )
    pub_wrist = rospy.Publisher(
        "/robot/gripper/qbmove2/control/qbmove2_position_and_preset_trajectory_controller/command",
        JointTrajectory, queue_size=10
    )

    rate = rospy.Rate(CONTROL_FREQUENCY)
    q = initialize_state()
    x = 0.0

    while not rospy.is_shutdown():
        try:
            (t_0V, q_0V) = listener.lookupTransform('/robot_arm_link0', '/tool_extremity', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.sleep(0.01)
            continue

        t_0B, q_0B = np.array(t_0B), np.array(q_0B)
        t_0V, q_0V = np.array(t_0V), np.array(q_0V)

        q_dot, e = feedback(t_0B, q_0B, t_0V, q_0V, q)
        q = q + q_dot / CONTROL_FREQUENCY

        q = check_joints_position(q)
        q_dot = check_joints_velocity(q_dot)

        arm_str = JointTrajectory()
        arm_str.header = Header(stamp=rospy.Time.now())
        arm_str.joint_names = JOINT_NAMES_ARM
        arm_point = JointTrajectoryPoint(
            positions=q[:7, 0].tolist(),
            velocities=q_dot[:7, 0].tolist(),
            accelerations=[0] * 7,
            time_from_start=rospy.Duration(x)
        )
        arm_str.points.append(arm_point)

        wrist_str = JointTrajectory()
        wrist_str.header = Header(stamp=rospy.Time.now())
        wrist_str.joint_names = ['qbmove2_shaft_joint', 'qbmove2_stiffness_preset_virtual_joint']
        wrist_point = JointTrajectoryPoint(
            positions=[q[7, 0], STIFFNESS_MAX],
            velocities=[q_dot[7, 0], 0.0],
            accelerations=[0, 0],
            time_from_start=rospy.Duration(x)
        )
        wrist_str.points.append(wrist_point)

        pub_arm.publish(arm_str)
        pub_wrist.publish(wrist_str)

        q = check_q_goal(q)

        norm_e_t = np.linalg.norm(e[:3, :], 2)
        norm_e_o = np.linalg.norm(e[3:, :], 2)
        rospy.loginfo(f"Translation error norm: {norm_e_t}")
        rospy.loginfo(f"Orientation error norm: {norm_e_o}")

        if norm_e_t < ERROR_TRANSLATION_THRESHOLD and norm_e_o < ERROR_ORIENTATION_THRESHOLD:
            rospy.loginfo("Visual servoing completed!")
            return True

        x += 1 / CONTROL_FREQUENCY
        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('controller')
    listener = tf.TransformListener()

    while True:
        try:
            (t_0B, q_0B) = listener.lookupTransform('/robot_arm_link0', '/object', rospy.Time(0))
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue

    rospy.loginfo("Starting grasp phase")
    response = call_service("grasp_tool_task", SetBool, True)
    if response:
        rospy.loginfo(response.message)

    rospy.loginfo("Setting wrist stiffness to maximum")
    stiffness_request = set_stiffnessRequest(stiffness_value=STIFFNESS_MAX)
    response = call_service("stiffness_service", set_stiffness, stiffness_request)
    if response and response.result:
        rospy.loginfo("Stiffness set successfully")

    rospy.loginfo("Starting visual servoing")
    if servoing():
        rospy.loginfo("Starting pick and throw phase")
        response = call_service("place_tool_task", SetBool, True)
        if response:
            rospy.loginfo(response.message)
        rospy.loginfo("Task completed!")
    else:
        rospy.logerr("Visual servoing failed!")