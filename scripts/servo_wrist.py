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
from pytransform3d.rotations import  quaternion_from_matrix, axis_angle_from_quaternion
from servoing_pkg.srv import set_stiffness, set_stiffnessRequest

def check_q_goal(q_goal):
# Funzione per vedere se il robot col polso hanno fisicamente raggiunto la posizione q_goal
    while 1:
        while 1: 
            joint_states = rospy.wait_for_message('/robot/joint_states', JointState)
            wrist_states = rospy.wait_for_message('/robot_arm/gripper/qbmove2/joint_states', JointState)
            if (joint_states.name == ['robot_arm_joint1', 'robot_arm_joint2', 'robot_arm_joint3', 'robot_arm_joint4', 'robot_arm_joint5', 'robot_arm_joint6', 'robot_arm_joint7'] and 
                wrist_states.name == ['qbmove2_motor_1_joint', 'qbmove2_motor_2_joint', 'qbmove2_shaft_joint', 'qbmove2_deflection_virtual_joint', 'qbmove2_stiffness_preset_virtual_joint']):
                q_arm_actual = np.array(joint_states.position).reshape((7,1))     # Stato letto dal joint states del robot
                q_wrist_actual = np.array(wrist_states.position[2]).reshape((1,1))     # Stato iniziale letto dal joint states del polso
                q_actual = np.vstack((q_arm_actual, q_wrist_actual))                                     # Stato robot + polso
                break
            else:
                continue # Aspetta lo stato del robot

        e_q = q_goal-q_actual

        if (np.linalg.norm(e_q[0:8,:], 2)<0.01):
            q = q_actual # Prendo lo stato attuale del robot
            return q   # Prossimo comando
        else:
            continue # Il robot non ha raggiunto q_goal, aspetta

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
    (J, T_0E) = get_jacobian(q)      # Calcolo del Jacobiano geometrico e della posizione di panda link 8 rispetto allo 0
    
    # Calcolo T_0B
    T_0B = hom_matrix(trans_0B, rot_0B)
    T_0V = hom_matrix(trans_0V, rot_0V)
    #T_0V = np.linalg.multi_dot([T_0E, T_EV])


    # Calcolo vettore errore
    t_0V = T_0V[:3,3]
    t_0B = T_0B[:3,3]
    R_0V = T_0V[:3,:3]
    R_0B = T_0B[:3,:3]

    R_e = np.matmul(R_0B, np.transpose(R_0V))
    q_e = quaternion_from_matrix(R_e)
    q_e_axisangle = axis_angle_from_quaternion(q_e)
    r = [q_e_axisangle[0], q_e_axisangle[1], q_e_axisangle[2]]
    theta = [q_e_axisangle[3]]
    
    e_t = (t_0B - t_0V)
    e_o = np.sin(theta)*r        # Errore di orientamento
    e = np.array([e_t, e_o], dtype='float').reshape((6,1))
    
    # Calcolo matrice L
    I = np.identity(3)
    Z = np.zeros((3,3))
    L_e = -0.5*(np.matmul(skew_matrix(R_0B[0:3,0]), skew_matrix(R_0V[0:3,0])) + np.matmul(skew_matrix(R_0B[0:3,1]), skew_matrix(R_0V[0:3,1])) + np.matmul(skew_matrix(R_0B[0:3,2]), skew_matrix(R_0V[0:3,2])))
    L_inv = np.linalg.pinv(L_e)

    L = np.block([[I, Z], [Z, L_inv]])

    # Calcolo inversa
    J_trans = np.transpose(J)
    J_1 = np.matmul(J, J_trans)
    J_pinv = np.matmul(J_trans, np.linalg.inv(J_1))
    
    # Calcolo legge di controllo
    K_p = np.array([[    1000,   0.0,    0.0],
                  [   0.0,    1000,    0.0 ],
                  [   0.0,  0.0,       1000]])
    K_o = 1000*np.identity(3)
    K = np.block([[K_p, Z], [Z, K_o]])
    q_dot = np.linalg.multi_dot([J_pinv, L, K, e])

    return q_dot, e 

def servoing():

    pub_arm = rospy.Publisher("/robot/arm/position_joint_trajectory_controller/command", JointTrajectory, queue_size=10)
    pub_wrist = rospy.Publisher("/robot/gripper/qbmove2/control/qbmove2_position_and_preset_trajectory_controller/command", JointTrajectory, queue_size=10)
    
    f = 10000.0      # Frequenza di spin del nodo
    x = 0.0          # Tempo per comando controllore
    i = 0            # Indice temporale

    rate = rospy.Rate(f)
    
    initialized = False
    pick = False

    while not rospy.is_shutdown():

        # Acquisisco lo stato iniziale, solo primo giro di braccio e polso
        while not initialized:
                joint_states = rospy.wait_for_message('/robot/joint_states', JointState)
                wrist_states = rospy.wait_for_message('/robot_arm/gripper/qbmove2/joint_states', JointState)
                if (joint_states.name == ['robot_arm_joint1', 'robot_arm_joint2', 'robot_arm_joint3', 'robot_arm_joint4', 'robot_arm_joint5', 'robot_arm_joint6', 'robot_arm_joint7'] and 
                    wrist_states.name == ['qbmove2_motor_1_joint', 'qbmove2_motor_2_joint', 'qbmove2_shaft_joint', 'qbmove2_deflection_virtual_joint', 'qbmove2_stiffness_preset_virtual_joint']):
                    q = np.array(joint_states.position).reshape((7,1))     # Stato iniziale letto dal joint states del robot
                    q_wrist = np.array(wrist_states.position[2]).reshape((1,1))     # Stato iniziale letto dal joint states del polso
                    q = np.vstack((q, q_wrist))                                     # Stato robot + polso
                    initialized = True
                    break
                else:
                    continue # Aspetta che arrivi stato iniziale

        # Acquisisco la trasformazione della ventosa rispetto al robot ad ogni ciclo
        try:
            (t_0V, q_0V) = listener.lookupTransform('/robot_arm_link0', '/tool_extremity', rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue     # Se ci sono errori prova di nuovo a prendere le tf

        # Conversione in vettori
        t_0B = np.array(t_0B)
        q_0B = np.array(q_0B)
        t_0V = np.array(t_0V)
        q_0V = np.array(q_0V)
         
        # Calcolo del nuovo stato con integrazione discreta e legge di controllo PBVS
        (q_dot, e) = feedback(t_0B, q_0B, t_0V, q_0V, q)             # Legge di Controllo PBVS, errore traslazione e orientamento
        q = q + q_dot*1/f                                            # Calcolo lo stato successivo

        # Check se limiti fisici del robot rispettati
        q = check_joints_position(q)
        q_dot = check_joints_velocity(q_dot)

        # Conversione dei dati per controllore di posizione
        q_tolist = [q[0][0], q[1][0], q[2][0], q[3][0], q[4][0], q[5][0], q[6][0]]
        dq_tolist = [q_dot[0][0], q_dot[1][0], q_dot[2][0], q_dot[3][0], q_dot[4][0], q_dot[5][0], q_dot[6][0]]
        ddq_tolist = [0,0,0,0,0,0,0]

        q_wrist_tolist = [q[7][0], 1.0] # Secondo valore rappresenta la stiffness del polso, ipotizzo controllo a massima rigidezza
        dq_wrist_tolist = [q_dot[7][0], 0.0]
        ddq_wrist_tolist = [0, 0]

        # Pubblica sul topic del controllore del robot il comando
        arm_str = JointTrajectory()
        arm_str.header = Header()
        arm_str.header.stamp = rospy.Duration(0)
        arm_str.joint_names = ['robot_arm_joint1', 'robot_arm_joint2', 'robot_arm_joint3', 'robot_arm_joint4', 'robot_arm_joint5', 'robot_arm_joint6', 'robot_arm_joint7']
        arm_point = JointTrajectoryPoint()
        arm_point.positions = q_tolist
        arm_point.velocities = dq_tolist
        arm_point.accelerations = ddq_tolist
        x = x + 1/f
        arm_point.time_from_start = rospy.Duration(x)
        arm_str.points.append(arm_point)

        # Pubblica sul topic del controllore del polso il comando
        wrist_str = JointTrajectory()
        wrist_str.header = Header()
        wrist_str.header.stamp = rospy.Duration(0)
        wrist_str.joint_names = ['qbmove2_shaft_joint', 'qbmove2_stiffness_preset_virtual_joint']
        wrist_point = JointTrajectoryPoint()
        wrist_point.positions = q_wrist_tolist
        wrist_point.velocities = dq_wrist_tolist
        wrist_point.accelerations = ddq_wrist_tolist

        wrist_point.time_from_start = rospy.Duration(x)
        wrist_str.points.append( wrist_point)
        
        pub_arm.publish(arm_str)        # Comando al controllore del robot
        pub_wrist.publish(wrist_str)    # Comando al controllore del polso

        q = check_q_goal(q)             # Controlla quando robot e polso hanno raggiunto la q desiderata e usala come nuovo stato

        # Criterio di arresto di servoing
        norm_e_t = np.linalg.norm(e[0:3,:], 2)
        norm_e_o = np.linalg.norm(e[4:7,:], 2)
        rospy.loginfo('La norma dell\'errore di traslazione: %s' %norm_e_t)
        rospy.loginfo('La norma dell\'errore di orientamento: %s' %norm_e_o)

        if (norm_e_t < 0.005 and norm_e_o < 0.01): # Criterio di raggiungimento regime
            pick = True
            return pick
        print("----------------------------------------------------------------------------------------------")

        i = i + 1 # Avanzamento indice temporale

        rate.sleep()

if __name__ == '__main__':
    
    rospy.init_node('controller')
    listener = tf.TransformListener()

    # Prendo la posa dell'oggetto rispetto alla base del robot
    
    while 1:
        try:
            (t_0B, q_0B) = listener.lookupTransform('/robot_arm_link0', '/object', rospy.Time(0))
            break

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            continue     # Se ci sono errori prova di nuovo a prendere le tf
    
    # Chiamata al primo servizio per afferrare il tool e andare in posizione di visual servoing
    
    rospy.loginfo("Inizio della fase di grasp del tool")
    rospy.wait_for_service("tool_grasp_task_wrist")

    try:
      client = rospy.ServiceProxy("grasp_tool_task", SetBool)
      request = True
      response = client(request)
      rospy.loginfo(response.message)

    except rospy.ServiceException as e:
      rospy.loginfo("Chiamata al servizio fallita: %s" %e)

    rospy.wait_for_service("stiffness_service")

    # Chiamo servizio per impostare rigidezza polso al massimo
    try:
      client = rospy.ServiceProxy("stiffness_service", set_stiffness)
      request = set_stiffnessRequest()
      request.stiffness_value = 1.0 
      response = client(request)

      if response.result:
        rospy.loginfo("Stiffness impostata al massimo")
      else:
        rospy.loginfo("Errore nell'impostare la stiffness")

    except rospy.ServiceException as e:
      rospy.loginfo("Chiamata al servizio fallita: %s" %e)

    rospy.loginfo("Inizio della fase di visual sevoing")

    # Inizio della fase di visual servoing
    check_servoing = servoing()

    # Inizio della fase di pick e throw
    if check_servoing:
        rospy.loginfo('Visual servoing completato!')
        # Chiamata al secondo servizio per il pick e throw
        rospy.loginfo("Inizio della fase di pick and throw dell'oggetto")
        rospy.wait_for_service("place_tool_task")

        try:

            client = rospy.ServiceProxy("place_tool_task", SetBool)
            request = True
            response = client(request)
            rospy.loginfo(response.message)

        except rospy.ServiceException as e:
            rospy.loginfo("Chiamata al servizio fallita: %s"%e)

        rospy.loginfo("Task completato!")

    else:
        rospy.loginfo('Errore nel Servoing!')
