#!/usr/bin/env python
import rospy
from std_msgs.msg import Bool

def callback(data):
    rospy.loginfo("Finisched task!!")
    
def listener():

    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("/servo_finish_task", Bool, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()