import numpy as np
import rospy
from typing import Tuple

def get_tf_mat(i: int, dh: np.ndarray) -> np.ndarray:
    """Calculate DH modified transformation matrix from DH parameters for the i-th joint.
    
    Parameters
    ----------
    i : int
        Index of the selected joint.
    dh : np.ndarray
        Matrix of DH parameters for n joints (shape: [n, 4]).
            
    Returns
    -------
    np.ndarray
        Homogeneous transformation matrix of DH for the i-th joint (shape: [4, 4]).
    """
    a, d, alpha, theta = dh[i]
    q = theta

    T = np.array([
        [np.cos(q), -np.sin(q), 0, a],
        [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
        [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
        [0, 0, 0, 1]
    ])

    return T

def get_jacobian(joint_angles: np.ndarray, use_wrist: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the geometric Jacobian for Panda robot with optional wrist inclusion.
    
    Parameters
    ----------
    joint_angles : np.ndarray
        Joint state vector (shape: [n,]).
    use_wrist : bool, optional
        Whether to include the wrist in the calculations, by default True.
            
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Geometric Jacobian (shape: [6, n]) and end-effector transformation matrix (shape: [4, 4]).
    """
    dh_params = get_dh_params(joint_angles, use_wrist)

    T_EE = np.eye(4)
    for i in range(len(dh_params)):
        # T_EE = np.matmul(T_EE, get_tf_mat(i, dh_params))
        T_EE = T_EE @ get_tf_mat(i, dh_params)

    J = np.zeros((6, len(dh_params)))
    T = np.eye(4)
    for i in range(len(dh_params)):
        # T = np.matmul(T, get_tf_mat(i, dh_params))
        T = T @ get_tf_mat(i, dh_params)

        p = T_EE[:3, 3] - T[:3, 3]
        z = T[:3, 2]

        J[:3, i] = np.cross(z, p)
        J[3:, i] = z

    if use_wrist:
        return J[:, :8], T_EE
    else:
        return J[:, :7], T_EE

def get_dh_params(joint_angles: np.ndarray, use_wrist: bool) -> np.ndarray:
    """Generate DH parameters for the Panda robot."""
    if use_wrist:
        return np.array([
            [0, 0.333, 0, joint_angles[0]],
            [0, 0, -np.pi / 2, joint_angles[1]],
            [0, 0.316, np.pi / 2, joint_angles[2]],
            [0.0825, 0, np.pi / 2, joint_angles[3]],
            [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
            [0, 0, np.pi / 2, joint_angles[5]],
            [0.088, 0.161, np.pi / 2, joint_angles[6]],
            [0.105, 0, -np.pi / 2, joint_angles[7]]
        ], dtype=np.float64)
    else:
        return np.array([
            [0, 0.333, 0, joint_angles[0]],
            [0, 0, -np.pi / 2, joint_angles[1]],
            [0, 0.316, np.pi / 2, joint_angles[2]],
            [0.0825, 0, np.pi / 2, joint_angles[3]],
            [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
            [0, 0, np.pi / 2, joint_angles[5]],
            [0.088, 0, np.pi / 2, joint_angles[6]],
            [0, 0.107, 0, 0]
        ], dtype=np.float64)

def check_joints_position(q: np.ndarray, use_wrist: bool = True) -> np.ndarray:
    """Enforce physical limits for Panda joint positions with optional wrist inclusion."""
    q_limits = get_joint_limits(use_wrist)

    for i, (lower, upper) in enumerate(q_limits):
        if not (lower <= q[i] <= upper):
            q[i] = np.clip(q[i], lower, upper)
            rospy.logwarn("Joint %s at position limit %s", i + 1, q[i])
    
    return q

def check_joints_velocity(q_dot: np.ndarray, use_wrist: bool = True) -> np.ndarray:
    """Enforce physical limits for Panda joint velocities with optional wrist inclusion."""
    q_dot_limits = get_velocity_limits(use_wrist)

    for i, (lower, upper) in enumerate(q_dot_limits):
        if not (lower <= q_dot[i] <= upper):
            q_dot[i] = np.clip(q_dot[i], lower, upper)
            rospy.logwarn("Joint %s at velocity limit %s", i + 1, q_dot[i])
    
    return q_dot

def get_joint_limits(use_wrist: bool) -> np.ndarray:
    """Return joint position limits."""
    if use_wrist:
        return np.array([
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973],
            [-2.0, 2.0]
        ], dtype=np.float64)
    else:
        return np.array([
            [-2.8973, 2.8973],
            [-1.7628, 1.7628],
            [-2.8973, 2.8973],
            [-3.0718, -0.0698],
            [-2.8973, 2.8973],
            [-0.0175, 3.7525],
            [-2.8973, 2.8973]
        ], dtype=np.float64)

def get_velocity_limits(use_wrist: bool) -> np.ndarray:
    """Return joint velocity limits."""
    if use_wrist:
        return np.array([
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.6100, 2.6100],
            [-2.6100, 2.6100],
            [-2.6100, 2.6100],
            [-5.5, 5.5]
        ], dtype=np.float64)
    else:
        return np.array([
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.1750, 2.1750],
            [-2.6100, 2.6100],
            [-2.6100, 2.6100],
            [-2.6100, 2.6100]
        ], dtype=np.float64)