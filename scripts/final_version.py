#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header, Bool
from geometry_msgs.msg import Pose
import asyncio
from servoing_pkg.kinematics import *
from servoing_pkg.sym_kin import *
from servoing_pkg.tools import *
from pytransform3d.rotations import axis_angle_from_matrix
from servoing_pkg.msg import servo_init

def initialize_state(use_wrist):
    """Initialize the robot and wrist state."""
    while True:
        joint_states = rospy.wait_for_message(ROBOT_JOINTS_STATE, JointState)
        if use_wrist:
            wrist_states = rospy.wait_for_message(WRIST_JOINTS_STATE, JointState)

        if joint_states.name == ROBOT_JOINTS_NAME and (not use_wrist or wrist_states.name == WRIST_JOINTS_NAME):
            q_arm = np.array(joint_states.position).reshape((7, 1))
            if use_wrist:
                q_wrist = np.array(wrist_states.position[2]).reshape((1, 1))
                q = np.vstack((q_arm, q_wrist))
                return q
            else:
                return q_arm

def check_q_goal(q_goal, use_wrist):
    """Check if the robot and wrist have reached the desired position."""
    while True:
        joint_states = rospy.wait_for_message(ROBOT_JOINTS_STATE, JointState)
        if use_wrist:
            wrist_states = rospy.wait_for_message(WRIST_JOINTS_STATE, JointState)

        if joint_states.name == ROBOT_JOINTS_NAME and (not use_wrist or wrist_states.name == WRIST_JOINTS_NAME):
            q_arm_actual = np.array(joint_states.position).reshape((7, 1))
            if use_wrist:
                q_wrist_actual = np.array(wrist_states.position[2]).reshape((1, 1))
                q_actual = np.vstack((q_arm_actual, q_wrist_actual))
                e_q = q_goal - q_actual

            else:
                q_actual = q_arm_actual
                e_q = q_goal - q_arm_actual

            if np.linalg.norm(e_q, 2) < 0.01:
                return q_actual
        
def feedback(trans_0B, rot_0B, trans_0V, rot_0V, trans_EV, rot_EV, q, use_wrist):
    """PBVS Visual servoing control law.

    Parameters
    ----------
    trans_0B : Translation vector of the object with respect to the robot base.

    rot_0B : Rotation matrix of the object with respect to the robot base.

    trans_EV : Translation vector of the tool with respect to the end-effector.

    rot_EV : Rotation matrix of the tool with respect to the end-effector.

    q : np.ndarray
        Current joint positions of the robot.

    use_wrist : bool
                Whether to use the wrist in the calculations.
            
    Returns
    -------
    dq : [7x1] np.ndarray
         PBVS control law joints velocity

    e: [6x1] np.ndarray
       Position and orientation error vector
    """
    
    # Compute the Jacobian and the transformation matrix from base to end-effector

    # (J, T_0E) = get_jacobian(q, use_wrist)
    # print("Jacobian: ", J)
    # print("Transformation matrix: ", T_0E)

    if use_wrist:

        J_8 = jac_sym_wrist(q)
        # T = kyn_sym_wrist(q)

        # # Transform about x of pi rad and y of pi/2 rad
        # T_aux = np.array([[0, 0, 1, 0],
        #                   [0, -1, 0, 0],
        #                   [1, 0, 0, 0],
        #                   [0, 0, 0, 1]])
        # T_0E = T @ T_aux

    else:

        J_8 = jac_sym(q)
        # T_0E = kin_sym(q)

        
    T_0B = hom_matrix(trans_0B, rot_0B)
    T_0V = hom_matrix(trans_0V, rot_0V)
    T_EV = hom_matrix(trans_EV, rot_EV)
    # T_0V = T_0E @ T_EV

    # Extract translation and rotation components from the transformation matrices
    t_0V, t_0B = T_0V[:3, 3], T_0B[:3, 3]
    R_0V, R_0B = T_0V[:3, :3], T_0B[:3, :3]

    # Compute the relative rotation matrix and convert it to quaternion and axis-angle representation
    R_e = R_0B @ R_0V.T
    r, theta = axis_angle_from_matrix(R_e)[:3], axis_angle_from_matrix(R_e)[3]
 
    # Compute the translation and orientation error vectors
    e_t = (t_0B - t_0V)
    e_o = np.sin(theta)*r   
    norm_e_t = np.linalg.norm(e_t, 2)
    norm_e_o = np.linalg.norm(e_o, 2)
    e_t = e_t/norm_e_t
    # e_o = e_o/norm_e_o 
    e = np.vstack((e_t, e_o)).reshape((6, 1))
    

    # Construct the interaction matrix for orientation control    
    I, Z = np.identity(3), np.zeros((3, 3))
    L_e = -0.5 * sum(np.matmul(skew_matrix(R_0B[:, i]), skew_matrix(R_0V[:, i])) for i in range(3))
    L = np.block([[I, Z], [Z, np.linalg.pinv(L_e)]])

    # Compute the pseudo-inverse of the Jacobian
    T_aux = np.array([[0, 0, 1, 0],
                    [0, -1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])
    T_EV = np.matmul(T_aux, T_EV)
    t_EV = T_EV[0:3, 3]
    Adj = np.block([[I, -skew_matrix(t_EV)], [Z, I]])
    J = Adj @ J_8 
    J_pinv = np.linalg.pinv(J)
    
    # Define proportional gain matrices for position and orientation control
    if(norm_e_t < 0.1):
        K_p = 2.0 * np.identity(3)*norm_e_t
    else:
        K_p = 1 * np.identity(3)*0.1
    # if(norm_e_o < 0.2):
    #     K_o = 1 * np.identity(3)*norm_e_o
    # else:
    K_o = 1 * np.identity(3)
    K = np.block([[K_p, Z], [Z, K_o]])

    # Compute the joint velocity command using the PBVS control law
    q_dot = np.linalg.multi_dot([J_pinv, L, K, e])

    # Return the computed joint velocities and the error vector
    return q_dot, e , norm_e_t, norm_e_o

def servoing(use_wrist, object_name, q, time):
    """Perform visual servoing."""
    global count
    global control_state 
    # Temporary variable for the time
    x = 0.0


    # Get the transformation between the object and the robot base
    while True:
        try:
            (t_0B, q_0B) = listener.lookupTransform(ROBOT_ARM_LINK0, '/' + object_name, rospy.Time(0))
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue


    # Get the transformation between the end-effector and the tool
    while True:
        try:
            (t_0V, q_0V) = listener.lookupTransform(ROBOT_ARM_LINK0, '/right_hand_ee_link', rospy.Time(0))
            break

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue     # Se ci sono errori prova di nuovo a prendere le tf
        
    while True:
        try:
            (t_EV, q_EV) = listener.lookupTransform('/wrist_base',  '/right_hand_ee_link', rospy.Time(0))
            break
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue   
        
    # Get the control law and error state
    q_dot, e, nt , no = feedback(t_0B, q_0B, t_0V, q_0V, t_EV, q_EV, q, use_wrist)    
    arm_str = JointTrajectory()    
    point = JointTrajectoryPoint(
        positions=q[:7, 0].tolist(),
        velocities=[0] * 7,
        accelerations=[0] * 7,
        time_from_start=rospy.Duration(1)
    )
    # arm_str.points.append(point)
    # Update the joint state
    q = q + q_dot/CONTROL_FREQUENCY
    x += 1/CONTROL_FREQUENCY

    # Check for limits
    q = check_joints_position(q, use_wrist)
    q_dot = check_joints_velocity(q_dot, use_wrist)

    # x =  1/CONTROL_FREQUENCY

    # Publish the command for the arm and wrist
    
    
    
    arm_str.header = Header(stamp=rospy.Time.now())
    arm_str.joint_names = ROBOT_JOINTS_NAME
    arm_point = JointTrajectoryPoint(
        positions=q[:7, 0].tolist(),
        velocities=q_dot[:7, 0].tolist(),
        accelerations=[0] * 7,
        time_from_start=rospy.Duration(x)
    )
    arm_str.points.append(arm_point)
    pub_arm.publish(arm_str)

    if use_wrist:
        wrist_str = JointTrajectory()
        wrist_str.header = Header(stamp=rospy.Time.now())
        wrist_str.joint_names = [WRIST_JOINTS_NAME[2], WRIST_JOINTS_NAME[4]]
        wrist_point = JointTrajectoryPoint(
            positions=[q[7, 0], STIFFNESS_MAX],
            velocities=[q_dot[7, 0], 0.0],
            accelerations=[0, 0],
            time_from_start=rospy.Duration(x)
        )
        wrist_str.points.append(wrist_point)
        pub_wrist.publish(wrist_str)

    # Check for goal

    # norm_e_t = np.linalg.norm(e[:3, :], 2)
    # norm_e_o = np.linalg.norm(e[3:, :], 2)

    errors = Pose()
    errors.position.x = e[0,:]*nt
    errors.position.y = e[1,:]*nt
    errors.position.z = e[2,:]*nt
    errors.orientation.w = 0.0
    errors.orientation.x = e[3,:]*no
    errors.orientation.y = e[4,:]*no
    errors.orientation.z = e[5,:]*no
    pub_err.publish(errors)
    count = count + 1
 
        

    rospy.loginfo(f"Translation error norm: {nt}")
    rospy.loginfo(f"Orientation error norm: {no}")

    if nt < ERROR_TRANSLATION_THRESHOLD and no < ERROR_ORIENTATION_THRESHOLD:
        ret = Bool()
        ret.data = True
        pub_finish_servo.publish(ret)
        sub_joints.unregister()
        sub_wrist.unregister()
        control_state = 0
        rospy.loginfo("Visual servoing completed!")
        return

def callback_joints(joint_states):

    global use_wrist
    global q_arm
    global q_wrist
    global object
    if(use_wrist == True):
        if joint_states.name == ROBOT_JOINTS_NAME:

            q_arm = np.array(joint_states.position).reshape((7, 1))
            # mutex.acquire()
            if(q_wrist != 0):
                q = np.vstack((q_arm, q_wrist))
                # mutex.release()
                # print(q)
                servoing(True, object, q, joint_states.header.stamp)
        
    else:
        if joint_states.name == ROBOT_JOINTS_NAME:

            q_arm = np.array(joint_states.position).reshape((7, 1))

            servoing(False, "object", q_arm, joint_states.header.stamp)
 
def callback_wrist(wrist_states):
    global use_wrist
    global q_wrist
    global mutex
    if(use_wrist == True):
        # await mutex.acquire()
        q_wrist = np.array(wrist_states.position[2]).reshape((1, 1))
        # mutex.release()tart

def start_callback(start):
    global object
    global sub_joints 
    global sub_wrist
    global control_state
    if(control_state == 0):
        if(start.data):
            object = start.object_name
            sub_joints = rospy.Subscriber("/robot/joint_states", JointState, callback_joints)
            sub_wrist = rospy.Subscriber(WRIST_JOINTS_STATE, JointState, callback_wrist)
            control_state = 1

if __name__ == '__main__':
    mutex = asyncio.Lock()
    # Initialize the ROS node
    rospy.init_node('controller')
    listener = tf.TransformListener()
    object = ""
    count = 0
    q_arm = 0
    q_wrist = 0
    control_state = 0
    # Get parameters from the parameter server
    use_wrist = rospy.get_param('~use_wrist', False)
    simulator = rospy.get_param('~simulator', True)
    n_objects = rospy.get_param('~n_objects', 1)

    # Constants
    if simulator:
        ROBOT_JOINTS_NAME = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        ROBOT_ARM_LINK0 = '/panda_link0'
        TCP_LINK = '/panda_link8'
        ROBOT_CONTROLLER_NAME = '/position_joint_trajectory_controller/command'
        ROBOT_JOINTS_STATE = '/joint_states'
    else:
        ROBOT_JOINTS_NAME = ['robot_arm_joint1', 'robot_arm_joint2', 'robot_arm_joint3', 'robot_arm_joint4', 'robot_arm_joint5', 'robot_arm_joint6', 'robot_arm_joint7']
        ROBOT_ARM_LINK0 = '/robot_arm_link0'
        TCP_LINK = '/wrist_base_link'
        ROBOT_CONTROLLER_NAME = '/robot/arm/position_joint_trajectory_controller/command'
        ROBOT_JOINTS_STATE = '/robot/joint_states'

    if use_wrist:
        WRIST_JOINTS_NAME = ['qbmove2_motor_1_joint', 'qbmove2_motor_2_joint', 'qbmove2_shaft_joint', 'qbmove2_deflection_virtual_joint', 'qbmove2_stiffness_preset_virtual_joint']
        WRIST_CONTROLLER_NAME = '/robot/gripper/qbmove2/control/qbmove2_position_and_preset_trajectory_controller/command'
        WRIST_JOINTS_STATE = '/robot/gripper/qbmove2/joint_states'
        TCP_LINK = '/wrist_base_link'

    STIFFNESS_MAX = 1.0
    ERROR_TRANSLATION_THRESHOLD = 0.005
    ERROR_ORIENTATION_THRESHOLD = 0.01
    CONTROL_FREQUENCY = 1.0

    pub_arm = rospy.Publisher(ROBOT_CONTROLLER_NAME, JointTrajectory, queue_size=10)
    pub_finish_servo = rospy.Publisher("/servo_finish_task", Bool, queue_size= 10)
    pub_err = rospy.Publisher("/servo_error",Pose,queue_size=10)
    sub_joints = 0
    sub_wrist = 0
    start_sub = rospy.Subscriber("/servo_start_task", servo_init, start_callback)

    if use_wrist:
            pub_wrist = rospy.Publisher(WRIST_CONTROLLER_NAME, JointTrajectory, queue_size=10)

    rospy.spin()