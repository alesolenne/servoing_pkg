#!/usr/bin/env python3

from std_srvs.srv import SetBool, SetBoolResponse, SetBoolRequest
import rospy

def grasp_request(req):
    request = SetBoolRequest()
    request.data = True
    response= SetBoolResponse()
    response.success = True
    response.message = "Grasp completato"
    return response

def throw_request(req):
    request = SetBoolRequest()
    request.data = True
    response= SetBoolResponse()
    response.success = True
    response.message = "Throw completato"
    return response

if __name__ == "__main__":
    rospy.init_node('server')
    s = rospy.Service('grasp_tool_task', SetBool, grasp_request)
    s = rospy.Service('throw_tool_task', SetBool, throw_request)
    print("Fake server online.")
    rospy.spin()
