import numpy as np
import rospy

def get_tf_mat(i, dh):
    """Calculate DH modified transformation matrix from DH parameters for the i joint.
    
    Parameters
    ----------
    dh : [nx4] np.ndarray
         Matrix of DH parameters for n joints
    i : int
        Index of the selected joints
            
    Returns
    -------
    T : [4x4] np.ndarray
        Homogeneus transformation matrix of DH for the i joint    
    """    
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    T = np.array([[np.cos(q), -np.sin(q), 0, a],
                     [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                     [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                     [0, 0, 0, 1]])

    return T

def get_jacobian(joint_angles):
    """Calculate the geometric Jacobian for Panda robot using DH modified convenction and Direct Kinematics.
    
    Parameters
    ----------
    joint_angles : [7x1] np.ndarray
                   Joint state vector
            
    Returns
    -------
    J : [6x7] np.ndarray
        Geometric Jacobian for Panda robot   
    """    
    dh_params = np.array([[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi / 2, joint_angles[1]],
                 [0, 0.316, np.pi / 2, joint_angles[2]],
                 [0.0825, 0, np.pi / 2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
                 [0, 0, np.pi / 2, joint_angles[5]],
                 [0.088, 0, np.pi / 2, joint_angles[6]],
                 [0, 0.107, 0, 0]], dtype=np.float64)

    T_EE = np.identity(4)
    for i in range(8):
        T_EE = np.matmul(T_EE, get_tf_mat(i, dh_params))

    J = np.zeros((6, 7))
    T = np.identity(4)
    for i in range(7):
        T = np.matmul(T, get_tf_mat(i, dh_params))

        p = T_EE[:3, 3] - T[:3, 3]
        z = T[:3, 2]

        J[:3, i] = np.cross(z, p)
        J[3:, i] = z

    return J[:, :7], T_EE

def check_joints_position(q):
    """Physical limits for Panda joint position.
    """
    q_limits = np.array([[-2.8973, 2.8973],
                         [-1.7628, 1.7628],
                         [-2.8973, 2.8973],
                         [-3.0718, -0.0698],
                         [-2.8973, 2.8973],
                         [-0.0175, 3.7525],
                         [-2.8973, 2.8973]], dtype=np.float64)
    

    for i in range(7):
        if (q[i] > q_limits[i][0] and q[i] < q_limits[i][1]):
            continue
        else:  #saturazione
            if q[i] < q_limits[i][0]:
                q[i] = q_limits[i][0]
                rospy.logwarn("Giunto %s al limite di posizione %s", i+1, q_limits[i][0])
            else:
                q[i] = q_limits[i][1]
                rospy.logwarn("Giunto %s al limite di posizione %s" ,i+1 ,q_limits[i][1])
    
    # rospy.loginfo("Posizione di controllo: (%s, %s, %s, %s ,%s, %s, %s)" , q[0][0], q[1][0], q[2][0], q[3][0], q[4][0], q[5][0], q[6][0])
    return q

def check_joints_velocity(q_dot):
    """Physical limits for Panda joint velocity.
    """
    q_dot_limits = np.array([[-2.1750, 2.1750],
                             [-2.1750, 2.1750],
                             [-2.1750, 2.1750],
                             [-2.1750, 2.1750],
                             [-2.6100, 2.6100],
                             [-2.6100, 2.6100],
                             [-2.6100, 2.6100]], dtype=np.float64)
    
    for i in range(7):
        if (q_dot[i] > q_dot_limits[i][0] and q_dot[i] < q_dot_limits[i][1]):
            continue
        else:  #saturazione
            if q_dot[i] < q_dot_limits[i][0]:
                q_dot[i] = q_dot_limits[i][0]
                rospy.logwarn("Giunto %s al limite di velocita %s", i+1, q_dot_limits[i][0])
            else:
                q_dot[i] = q_dot_limits[i][1]
                rospy.logwarn("Giunto %s al limite di velocita %s", i+1, q_dot_limits[i][1])
    
    # rospy.loginfo("Velocita di controllo: (%s, %s, %s, %s ,%s, %s, %s)" , q_dot[0][0], q_dot[1][0], q_dot[2][0], q_dot[3][0], q_dot[4][0], q_dot[5][0], q_dot[6][0])
    return q_dot
