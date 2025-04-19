#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from servoing_pkg.sym_kin import *

def publisher_node():
    rospy.init_node('test_publisher', anonymous=True)
    pub = rospy.Publisher('test_topic', String, queue_size=10)

    while not rospy.is_shutdown():
        message = "Hello, ROS!"
        c = jac_sym(0.1,0.2,0.0,-0.0,0.0,0.0,0.0)
        print(c)
        pub.publish(message)

if __name__ == '__main__':
    try:
        publisher_node()
    except rospy.ROSInterruptException:
        pass