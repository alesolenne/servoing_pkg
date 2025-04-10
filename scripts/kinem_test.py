#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from std_srvs.srv import SetBool

from servoing_pkg.kinematics import *
from servoing_pkg.tools import *
from servoing_pkg.plots import *
from pytransform3d.rotations import axis_angle_from_matrix
# from servoing_pkg.srv import set_stiffness, set_stiffnessRequest

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

def initialize_state(use_wrist):
    """Initialize the robot and wrist state."""
    while True:
        joint_states = rospy.wait_for_message(ROBOT_JOINTS_STATE, JointState)
        print(joint_states)
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
        
def feedback(q, use_wrist):
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
    (J, T_0E) = get_jacobian(q, use_wrist)
    T = np.array([
        [0,0,1,0],
        [0,-1,0,0],
        [1,0,0,0],
        [0, 0, 0, 1]
    ])
    T_DH = np.matmul(T_0E,T )
    print("dh")
    print(T_DH)

    return

def servoing(use_wrist):
      """Perform visual servoing."""

      # Get the initial joint state
      q = initialize_state(use_wrist)

      # Temporary variable for the time
      x = 0.0

      rate = rospy.Rate(CONTROL_FREQUENCY)

      while not rospy.is_shutdown():
         
        # Get the control law and error state
        feedback(q, use_wrist)        

        rate.sleep()

if __name__ == '__main__':

    # Initialize the ROS node
    rospy.init_node('controller')
    listener = tf.TransformListener()

    # Get parameters from the parameter server
    use_wrist = rospy.get_param('~use_wrist', True)
    enable_plotting = rospy.get_param('~enable_plotting', False)
    simulator = rospy.get_param('~simulator', False)

    # Constants
    if simulator:
        ROBOT_JOINTS_NAME = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        ROBOT_ARM_LINK0 = '/panda_link0'
        ROBOT_ARM_LINK8 = '/panda_link8'
        ROBOT_CONTROLLER_NAME = '/position_joint_trajectory_controller/command'
        ROBOT_JOINTS_STATE = '/joint_states'
    else:
        ROBOT_JOINTS_NAME = ['robot_arm_joint1', 'robot_arm_joint2', 'robot_arm_joint3', 'robot_arm_joint4', 'robot_arm_joint5', 'robot_arm_joint6', 'robot_arm_joint7']
        ROBOT_ARM_LINK0 = '/robot_arm_link0'
        ROBOT_ARM_LINK8 = '/robot_arm_link8'
        ROBOT_CONTROLLER_NAME = '/robot/arm/position_joint_trajectory_controller/command'
        ROBOT_JOINTS_STATE = '/robot/joint_states'

    if use_wrist:
        WRIST_JOINTS_NAME = ['qbmove2_motor_1_joint', 'qbmove2_motor_2_joint', 'qbmove2_shaft_joint', 'qbmove2_deflection_virtual_joint', 'qbmove2_stiffness_preset_virtual_joint']
        WRIST_CONTROLLER_NAME = '/robot/gripper/qbmove2/control/qbmove2_position_and_preset_trajectory_controller/command'
        WRIST_JOINTS_STATE = '/robot/gripper/qbmove2/joint_states'

    STIFFNESS_MAX = 1.0
    ERROR_TRANSLATION_THRESHOLD = 0.005
    ERROR_ORIENTATION_THRESHOLD = 0.01
    CONTROL_FREQUENCY = 1000.0

    if enable_plotting:
        # Preallocate arrays for plotting
        estimated_duration = 120  # seconds (adjust based on expected runtime)
        max_steps = int(CONTROL_FREQUENCY * estimated_duration)

        q_plot = np.zeros((max_steps, 7))
        dq_plot = np.zeros((max_steps, 7))
        e_plot = np.zeros((max_steps, 6))
        i= 0

    pub_arm = rospy.Publisher(ROBOT_CONTROLLER_NAME, JointTrajectory, queue_size=10)

    if use_wrist:
            pub_wrist = rospy.Publisher(WRIST_CONTROLLER_NAME, JointTrajectory, queue_size=10)


    rospy.loginfo("Starting visual servoing")
    if servoing(use_wrist):       
        rospy.loginfo("Starting place phase")

    else:
        rospy.logerr("Visual servoing failed!")
    