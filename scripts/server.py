#!/usr/bin/env python3

from std_srvs.srv import SetBool, SetBoolResponse
import rospy
from servoing_pkg.srv import grasp, graspResponse
from servoing_pkg.srv import task, taskResponse

def grasp_request(req):
    # request = SetBoolRequest()
    # request.data = True
    response= SetBoolResponse()
    response.success = True
    response.message = "Grasp completato"
    return response

def throw_request(req):
    # request = SetBoolRequest()
    # request.data = True
    response= graspResponse()
    response.success = True
    response.message = "Throw completato"
    return response

def place_request(req):
    response = SetBoolResponse()
    response.success = True
    response.message = "Place completo"
    return response

def task_request(req):
    response = taskResponse()
    response.success = True
    response.message = "Home completo"
    return response

if __name__ == "__main__":
    rospy.init_node('server')
    s = rospy.Service('grasp_task_wrist_tool_service', SetBool, grasp_request)
    s = rospy.Service('throw_task_wrist_tool_servo_service', grasp, throw_request)
    s = rospy.Service('replace_task_wrist_tool_service', SetBool, place_request)
    s = rospy.Service('joint_config_service', task, task_request)

    print("Fake server online.")
    rospy.spin()
